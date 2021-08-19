import cv2 as cv
import torch
import torch.nn as nn
import torchvision as torchcv
import torch.utils.data as torchdata
import matplotlib.pyplot as plt
import albumentations as albu
import numpy as np
import os
import PIL.Image as pimg
import time
import moco.loader
import moco.builder as moco_builder
import math
import shutil
import segmentation_models_pytorch as smp
import cv2
import matplotlib.pyplot as plt
from util import metrics
from util import loss
from util import run
import pickle
from util import save
import random


def visualize_assist(**kwargs):
    n = len(kwargs)
    plt.figure()
    for i, (name, img) in enumerate(kwargs.items()):
        plt.subplot(1, n, i + 1)
        plt.title(name)
        plt.imshow(img)
    plt.show()


def visualize_assist2(**kwargs):
    n = len(kwargs)
    plt.figure()
    for i, (name, img) in enumerate(kwargs.items()):
        plt.subplot(n // 2 + n % 2, 2, i + 1)
        plt.title(name)
        plt.imshow(img)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                        wspace=0.2, hspace=0.8)
    plt.show()


def visualize_dataset(dataset):
    filenamex, imagex, maskx = dataset[0]
    imagex = dataset.get_original_image(0)
    visualize_assist(image=imagex, original=maskx[1])


def to_tensor(x, **kwargs):
    if isinstance(x, torch.Tensor):
        return x.permute(2, 0, 1).to(torch.float)
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


def pet_augmentation():
    transform_list = [
        albu.Resize(320, 320),
        albu.HorizontalFlip(p=0.5),
        albu.MultiplicativeNoise(p=0.7,multiplier=(0.8, 1.2),elementwise=True),
        albu.GaussianBlur(p=0.5,blur_limit=3)
    ]
    return albu.Compose(transform_list)


def pet_augmentation_valid():
    transform_list = [
        albu.Resize(320, 320),
    ]
    return albu.Compose(transform_list)

class ArvnDataset_Pet_Constrastive(torchdata.Dataset):
    def __init__(self,
                 image_src: str,
                 classes: list = None,
                 augmentation: callable = None,
                 preprocessing: callable = None, ):
        if classes is None:
            classes = []
        self.data_name = []
        self.classes = classes
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        for i in range(len(self.classes)):
            file_list = os.listdir(image_src + "/" + self.classes[i])
            index_list = [i for _ in range(len(file_list))]
            path_list = [image_src + "/" + self.classes[i] for _ in range(len(file_list))]
            join_list = zip(file_list, index_list, path_list)
            self.data_name.extend(join_list)
        np.random.shuffle(self.data_name)

    def __len__(self):
        return len(self.data_name)

    def __getitem__(self, item):
        image = pimg.open(self.data_name[item][2] + "/" + self.data_name[item][0]).convert("RGB")
        image = np.array(image)
        image2 = image.copy()
        label = self.data_name[item][1]
        if self.augmentation:
            sp = self.augmentation(image=image)
            sp2 = self.augmentation(image=image2)
            image = sp['image']
            image2 = sp2['image']
        if self.preprocessing:
            image = self.preprocessing(image=image)['image']
            image2 = self.preprocessing(image=image2)['image']

        return [image, image2], label


class PetNet_V2(nn.Module):
    def __init__(self, num_classes=128):
        super(PetNet_V2, self).__init__()
        self.encoder = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            classes=2
        ).encoder
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.flatten = nn.Flatten()
        self.fullconnect = nn.Linear(in_features=512, out_features=128)

    def forward(self, x):
        x = self.encoder(x)[-1]
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fullconnect(x)
        return x


class SegDataset(torch.utils.data.Dataset):
    CLASSES = ['tissue', 'pancreas']

    def __init__(
            self,
            dir_img,
            dir_mask,
            classes=None,
            augmentation=None,
            preprocessing=None,
            maxsize=65536
    ):
        self.ids = os.listdir(dir_img)
        random.shuffle(self.ids)
        self.images_fps = [os.path.join(dir_img, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(dir_mask, image_id) for image_id in self.ids]

        # convert str names to class values on masks
        self.class_values = [i for i in range(len(classes))]

        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.maxsize = maxsize

    def get_original_image(self, i):
        fname = self.images_fps[i]
        image = cv2.imread(fname)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def __getitem__(self, i):
        fname = self.images_fps[i]
        # read data
        image = cv2.imread(fname)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)

        # extract certain classes from mask (e.g. cars)
        masks = [mask == v for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        # apply preprocessing
        if self.preprocessing:
            image = self.preprocessing(image=image)['image']
            mask = to_tensor(mask)

        fname = fname.split('/')[-1]
        return fname, image, mask

    def __len__(self):
        # return len(self.ids)
        return min(self.maxsize, len(self.ids))

def load_unet_weights(unet, root, verbose=False):
    backbone = torch.load(root)
    backbone_state_dict = backbone.state_dict()
    state = unet.state_dict()
    keys = [k for k, _ in state.items()]
    for k in keys:
        if k in backbone.state_dict():
            try:
                if 'segmentation_head' not in k:
                    state[k] = backbone.state_dict()[k]
            except Exception as e:
                print(f'mismatch error = {e}')
            if verbose:
                print(f'transfer {k}')

    unet.load_state_dict(state)
    return unet

def visualize_output(dataset, model, modelp, idx):
    filenamex, imagex, maskx = dataset[idx]
    imagex = dataset.get_original_image(idx)
    maskx[1] = maskx[1]
    maskp = np.array([maskx[1], maskx[1], maskx[1]])
    maskp = maskp.transpose(1, 2, 0)
    tensorx = torch.tensor(dataset[idx][1])
    tensorx = tensorx.unsqueeze(0)
    softmax2d = torch.nn.Softmax2d()
    tensory = model(tensorx)
    tensory = softmax2d(tensory).squeeze()
    output_y = tensory[1]
    output_y = output_y.detach().numpy()
    output_yp = np.array([output_y, output_y, output_y])
    output_yp = output_yp.transpose(1, 2, 0)
    output_ypf = np.where(output_yp > 0.5, 1.0, 0.0)
    visualize_assist2(image=imagex, original_mask=maskp,
                      predict_prob=output_yp, predict_mask=output_ypf)


DEVICE = "cuda"
SAVE_INTERVAL = 1
root = r'C:\Users\huang\Desktop\wen\MRP\MRP'
experiment = 'ss-test'
save_root = os.path.join(root, 'results/' + experiment)
public_save_root = os.path.join(root, 'results')

unet = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    classes=2,
    activation=None,
)
unet = load_unet_weights(unet,"model-exp10-ss.pth")
preproc_fn = smp.encoders.get_preprocessing_fn("resnet34")
train_dataset = SegDataset(
    r"D:\liver2\liver2\train-150",
    r"D:\liver2\liver2\train\masks",
    augmentation=pet_augmentation_valid(),
    preprocessing=get_preprocessing(preproc_fn),
    classes=['tissue', 'pancreas'],
    maxsize=99999
)
valid_dataset = SegDataset(
    r"D:\liver2\liver2\test\imgs",
    r"D:\liver2\liver2\test\masks",
    augmentation=pet_augmentation_valid(),
    preprocessing=get_preprocessing(preproc_fn),
    classes=['tissue', 'pancreas'],
    maxsize=99999
)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True,pin_memory=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=2, shuffle=True,pin_memory=True)
data_root = r'D:\liver2\liver2'
lr = 3e-5
x_train_dir = os.path.join(data_root, 'train-150')
y_train_dir = os.path.join(data_root, 'train/masks')
x_test_dir = os.path.join(data_root, 'test/imgs')
y_test_dir = os.path.join(data_root, 'test/masks')
loss_unet = loss.DiceLoss(weight=0.2, activation='softmax2d', ignore_channels=[0]) + loss.FocalLoss()
optimizer_unet = torch.optim.Adam(unet.parameters(), lr=lr)
metrics = [
    metrics.SMPIoU(threshold=0.5, ignore_channels=[0], activation='softmax2d'),
    metrics.Fscore(threshold=0.5, ignore_channels=[0], activation='softmax2d'),
]

train_epoch = run.TrainEpoch(
    model=unet,
    loss=loss_unet,
    metrics=metrics,
    optimizer=optimizer_unet,
    device=DEVICE,
    verbose=True,
)

valid_epoch = run.ValidEpoch(
    model=unet,
    loss=loss_unet,
    metrics=metrics,
    device=DEVICE,
    verbose=True,
)
train_record = []
valid_record = []
epochs = 100
for epoch in range(epochs):
    optimizer_unet.param_groups[0]['lr'] = lr * (math.pow(0.96, epoch))
    print(f"current {epoch} lr={optimizer_unet.param_groups[0]['lr']}")
    train_logs = train_epoch.run(train_loader)
    train_record.append(train_logs)
    torch.save(unet, "model-exp11.pth")

    with open('exp-11-train.txt', 'wb') as f:
        pickle.dump(train_record, f)

print("VALIDATING...")
valid_logs = valid_epoch.run(valid_loader)
valid_record.append(valid_logs)
with open('exp-11-valid.txt', 'wb') as f:
    pickle.dump(valid_record, f)
