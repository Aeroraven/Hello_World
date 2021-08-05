# Stage 3 Period 1
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


def pil_visualizex(image, text):
    plt.figure()
    plt.text(0, -20, text, fontsize=15)
    plt.imshow(image)
    plt.show()


class PetClassifier(nn.Module):
    def __init__(self):
        super(PetClassifier, self).__init__()
        self.layer_conv1 = nn.Conv2d(3, 6, 5)  # 2~98 => 96*96
        self.layer_actv1 = nn.ReLU()
        self.layer_pool1 = nn.MaxPool2d(2, 2)  # 48*48
        self.layer_conv2 = nn.Conv2d(6, 16, 5)  # 2~46 => 44*44
        self.layer_actv2 = nn.ReLU()
        self.layer_pool2 = nn.MaxPool2d(2, 2)  # 16*22*
        self.layer_flatten = nn.Flatten()
        self.layer_fc1 = nn.Linear(16 * 22 * 22, 1024)
        self.layer_fcr1 = nn.ReLU()
        self.layer_fc2 = nn.Linear(1024, 256)
        self.layer_fcr2 = nn.ReLU()
        self.layer_fc3 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.layer_conv1(x)
        x = self.layer_actv1(x)
        x = self.layer_pool1(x)
        x = self.layer_conv2(x)
        x = self.layer_actv2(x)
        x = self.layer_pool2(x)
        x = x.view(-1, 16 * 22 * 22)
        x = self.layer_fc1(x)
        x = self.layer_fcr1(x)
        x = self.layer_fc2(x)
        x = self.layer_fcr2(x)
        x = self.layer_fc3(x)
        return x


model = torch.load("./model_pet.pth")
path = r"C:\Users\Null\Desktop\Internship\Demo\images\pig.jpg"
argu = [
    albu.Resize(100, 100)
]
argu_fn = albu.Compose(argu)
image = pimg.open(path).convert("RGB")
imagex = pimg.open(path).convert("RGB")
image = np.array(image)
sample = argu_fn(image=image)
image = sample["image"].transpose(2, 0, 1)
image = torch.tensor(image) / 256
image = image.unsqueeze(0)
x = model(image)
softmax = torch.nn.Softmax(dim=1)
prob = softmax(x)
prob_py = prob.detach().numpy().tolist()[0]
stat = f"有{int(prob_py[0] * 10000) / 100}%的概率更像猫，\n有{int(prob_py[1] * 10000) / 100}%的概率更像狗"
if prob_py[0] > prob_py[1]:
    stat += "（更像猫）"
else:
    stat += "（更像狗）"

plt.rcParams['font.sans-serif'] = ['SimHei']
pil_visualizex(imagex, stat)
