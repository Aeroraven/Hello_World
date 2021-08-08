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
import segmentation_models_pytorch as smp


def pil_visualize(**kwargs):
    n = len(kwargs)
    plt.figure()
    for i, (name, image) in enumerate(kwargs.items()):
        plt.subplot(1, n, i + 1)
        plt.imshow(image)
        plt.title(name)
    plt.show()


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


def pet_argumentation():
    transform_list = [
        albu.Resize(100, 100),
        albu.HorizontalFlip(p=0.5),
        albu.ToSepia(p=0.2),
        albu.ToGray(p=0.3),
        albu.RandomRotate90(p=0.5),
        albu.VerticalFlip(p=0.2)
    ]
    return albu.Compose(transform_list)


class ArvnDataset_Pet(torchdata.Dataset):
    def __init__(self,
                 image_src: str,
                 classes: list = None,
                 argumentation: callable = None,
                 preprocessing: callable = None, ):
        if classes is None:
            classes = []
        self.data_name = []
        self.classes = classes
        self.argumentation = argumentation
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
        label = self.data_name[item][1]
        if self.argumentation:
            sp = self.argumentation(image=image)
            image = sp['image']
        if self.preprocessing:
            image = self.preprocessing(image=image)['image']
        image = torch.tensor(image)
        return image, label


class PetNet(nn.Module):
    def __init__(self):
        super(PetNet, self).__init__()
        self.encoder = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            classes=2
        ).encoder
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.flatten = nn.Flatten()
        self.fullconnect = nn.Linear(in_features=512, out_features=2)

    def forward(self, x):
        x = self.encoder(x)[-1]
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fullconnect(x)
        return x


data_dir = r"D:\PetImages"
train_dir = data_dir
preproc_fn = smp.encoders.get_preprocessing_fn("resnet34")
train_dataset = ArvnDataset_Pet(train_dir, ["Cat", "Dog"], pet_argumentation(), get_preprocessing(preproc_fn))

image, label = train_dataset[0]
model = PetNet().to("cpu")
image = image.unsqueeze(0)
image = torch.cat([image, image, image], dim=0)
y = model(image)
print(y)
print(y.shape)
