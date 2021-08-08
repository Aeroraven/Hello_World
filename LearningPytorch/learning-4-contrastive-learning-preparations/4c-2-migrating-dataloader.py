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
        albu.VerticalFlip(p=0.2),
        albu.GaussianBlur(blur_limit=10,p=0.5,always_apply=False)
    ]
    return albu.Compose(transform_list)


class ArvnDataset_Pet_Constrastive(torchdata.Dataset):
    def __init__(self,
                 image_src: str,
                 classes: list = None,
                 argumentation: callable = None,
                 preprocessing: callable = None,):
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
        image2 = image.copy()
        label = self.data_name[item][1]
        if self.argumentation:
            sp = self.argumentation(image=image)
            sp2 = self.argumentation(image=image2)
            image = sp['image']
            image2 = sp2['image']
        if self.preprocessing:
            image = self.preprocessing(image=image)['image']
            image2 = self.preprocessing(image=image2)['image']

        return [image,image2], label


data_dir = r"D:\PetImages"
train_dir = data_dir
train_dataset = ArvnDataset_Pet_Constrastive(train_dir, ["Cat", "Dog"], pet_argumentation())

train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=50,pin_memory=True,shuffle=False,drop_last=True)
y=None
for i,(image,_)in enumerate(train_dataloader):
    y=image
    break