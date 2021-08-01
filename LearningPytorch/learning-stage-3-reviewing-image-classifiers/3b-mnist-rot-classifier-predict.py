import torch
import torchvision as torchcv
import torch.utils.data as torchdata
import torch.nn as nn
import cv2 as cv
import matplotlib.pyplot as plt
import os
import time
import albumentations as albu
import numpy as np
import PIL.Image as pimg


def pil_visualize(**kwargs):
    size = len(kwargs)
    plt.figure()
    for i, (name, image) in enumerate(kwargs.items()):
        plt.subplot(1, size, i + 1)
        plt.title(name)
        plt.imshow(image)
    plt.show()


mnist_dataset = torchcv.datasets.MNIST(
    root="data",
    download=True,
    train=True,
    transform=torchcv.transforms.ToTensor()
)


def dataset_visualize(dataset, idx):
    vect = image = dataset[idx][0].squeeze().numpy()
    vect = np.rot90(vect, 2)
    vect_st = np.array([vect, vect, vect]) * 255
    vect_st = vect_st.astype("uint8")
    vect_st = vect_st.transpose(1, 2, 0)
    pil_visualize(image=vect_st)


class MNISTRot(torchdata.Dataset):
    def __init__(self, mnist_datset_base, size):
        self.ds = mnist_datset_base
        self.size = size

    def __getitem__(self, item):
        x = np.random.randint(len(self.ds))
        rot = np.random.randint(4)
        label = self.ds[x][1]
        vect = self.ds[x][0].squeeze().numpy()
        vect = np.rot90(vect, rot)
        vect = torch.tensor(vect.copy())
        vect = vect.unsqueeze(0)
        label = int(label) + rot * 10
        return vect, label

    def __len__(self):
        return self.size


class CNN_MNIST(nn.Module):
    def __init__(self):
        super(CNN_MNIST, self).__init__()
        self.layer_conv1 = nn.Conv2d(1, 8, 5)  # 24*24
        self.layer_actv1 = nn.ReLU()
        self.layer_pool1 = nn.MaxPool2d(2, 2)  # 12*12
        self.layer_conv2 = nn.Conv2d(8, 64, 3)  # 10*10
        self.layer_actv2 = nn.ReLU()
        self.layer_pool2 = nn.MaxPool2d(2, 2)  # 5*5
        self.layer_fc1 = nn.Linear(64 * 5 * 5, 1024)
        self.layer_fcr1 = nn.ReLU()
        self.layer_fc2 = nn.Linear(1024, 256)
        self.layer_fcr2 = nn.ReLU()
        self.layer_fc3 = nn.Linear(256, 40)

    def forward(self, x):
        x = self.layer_conv1(x)
        x = self.layer_actv1(x)
        x = self.layer_pool1(x)
        x = self.layer_conv2(x)
        x = self.layer_actv2(x)
        x = self.layer_pool2(x)
        x = x.view(-1, 64 * 5 * 5)
        x = self.layer_fc1(x)
        x = self.layer_fcr1(x)
        x = self.layer_fc2(x)
        x = self.layer_fcr2(x)
        x = self.layer_fc3(x)
        return x


# model = CNN_MNIST().to("cpu")
model = torch.load("./model_mnist_rot.pth")

path = r"C:\Users\Null\Desktop\Internship\Demo\images\hand.jpg"
image = pimg.open(path).convert("RGB")
image = np.array(image)
transforms = [
    albu.Resize(28, 28)
]
transformx = albu.Compose(transforms)
image = transformx(image=image)['image']
image = image.transpose(2, 0, 1)
image = image / 256
image = image.astype("float32")
image = torch.tensor(image.copy())[0]
image = image.unsqueeze(0)
image = image.unsqueeze(0)
softmax = nn.Softmax(dim=1)
pred_logits = model(image)
pred_probs = softmax(pred_logits)
pred_ans = torch.argmax(pred_probs, dim=1)
ans_rot = int(pred_ans / 10)
ans_fig = int(pred_ans % 10)
stat = ""
if ans_rot == 0:
    stat = "无旋转"
elif ans_rot == 1:
    stat = "逆时针旋转90"
elif ans_rot == 2:
    stat = "逆时针旋转180"
elif ans_rot == 3:
    stat = "逆时针旋转270"
else:
    stat = str(ans_rot)
print(f"书写的数字是 {ans_fig},旋转情况可能是 {stat}")
print("Softmax层输出")
print(pred_probs)
