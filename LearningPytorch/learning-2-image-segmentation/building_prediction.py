# Stage 2 period 4
import torch
import cv2 as cv
import torchvision as torchcv
import torch.utils.data as torchdata
import os
import numpy as np
import matplotlib.pyplot as plt
import albumentations as albu
import segmentation_models_pytorch as smp


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def visualize_help(**kwargs):
    n = len(kwargs)
    plt.figure()
    for i, (name, image) in enumerate(kwargs.items()):
        plt.subplot(1, n, i + 1)
        plt.imshow(image)
        plt.title(name)
    plt.show()


def dataset_argumentation():
    transform_list = [
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5)
    ]
    return albu.Compose(transform_list)


def dataset_preprocessing(preproc_fn):
    transform_list = [
        albu.Lambda(preproc_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor)
    ]
    return albu.Compose(transform_list)


class LvDataset(torchdata.Dataset):
    def __init__(self,
                 img_dir: str,
                 mask_dir: str,
                 class_idx: list,
                 argumentation: callable = None,
                 preprocessing: callable = None):
        self.data_file_list = os.listdir(img_dir)
        if len(self.data_file_list) >= 500:
            self.data_file_list = self.data_file_list[:500]
        self.data_path_list = [img_dir + '/' + x for x in self.data_file_list]
        self.mask_path_list = [mask_dir + '/' + x for x in self.data_file_list]
        self.classes = class_idx
        self.argumentation = argumentation
        self.preprocessing = preprocessing

    def __getitem__(self, item: int):
        image = cv.imread(self.data_path_list[item])
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        mask = cv.imread(self.mask_path_list[item], 0)
        masks = [(mask == x) for x in self.classes]
        mask_extracted = np.stack(masks, axis=-1).astype('float')

        if self.argumentation:
            arg_sample = self.argumentation(image=image, mask=mask_extracted)
            image, mask_extracted = arg_sample['image'], arg_sample['mask']
        if self.preprocessing:
            arg_sample = self.preprocessing(image=image, mask=mask_extracted)
            image, mask_extracted = arg_sample['image'], arg_sample['mask']

        return image, mask_extracted

    def __len__(self):
        return len(self.data_file_list)


model = smp.Unet(
    encoder_name='resnet34',
    encoder_weights='imagenet',
    classes=1,
    activation='sigmoid'
)
preproc_fn = smp.encoders.get_preprocessing_fn('resnet34', 'imagenet')
x_train_path = r"D:\liver2\liver2\train\imgs"
y_train_path = r"D:\liver2\liver2\train\masks"
x_test_path = r"D:\liver2\liver2\test\imgs"
y_test_path = r"D:\liver2\liver2\test\masks"

train_dataset = LvDataset(x_train_path, y_train_path, [1], dataset_argumentation(), dataset_preprocessing(preproc_fn))
test_dataset = LvDataset(x_test_path, y_test_path, [1], dataset_argumentation(), dataset_preprocessing(preproc_fn))

train_loader = torchdata.DataLoader(train_dataset, batch_size=5, shuffle=True, num_workers=0)
test_loader = torchdata.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

best_model = torch.load('./model_est.pth')

n = np.random.choice(len(test_loader))
image, mask  = test_dataset[n]
mask = mask.squeeze()
x_tensor = torch.from_numpy(image).to("cpu").unsqueeze(0)
pred_mask = best_model.predict(x_tensor)
pred_mask = pred_mask.squeeze().cpu().numpy().round()
image = image.transpose(1,2,0).astype("float32")
visualize_help(image=image,original_mask=mask,predicted_mask=pred_mask)