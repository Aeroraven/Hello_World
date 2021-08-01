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
    for i,(name,image) in enumerate(kwargs.items()):
        plt.subplot(1,size,i+1)
        plt.title(name)
        plt.imshow(image)
    plt.show()


mnist_dataset = torchcv.datasets.MNIST(
    root="data",
    download=True,
    train=True,
    transform=torchcv.transforms.ToTensor()
)


def dataset_visualize(dataset,idx):
    vect = image=dataset[idx][0].squeeze().numpy()
    vect = np.rot90(vect,2)
    vect_st = np.array([vect,vect,vect])*255
    vect_st = vect_st.astype("uint8")
    vect_st = vect_st.transpose(1,2,0)
    pil_visualize(image=vect_st)

class MNISTRot(torchdata.Dataset):
    def __init__(self,mnist_datset_base,size):
        self.ds = mnist_datset_base
        self.size = size

    def __getitem__(self, item):
        x = np.random.randint(len(self.ds))
        rot = np.random.randint(4)
        label = self.ds[x][1]
        vect = self.ds[x][0].squeeze().numpy()
        vect = np.rot90(vect,rot)
        vect = torch.tensor(vect.copy())
        vect = vect.unsqueeze(0)
        label = int(label)+rot*10
        return vect, label

    def __len__(self):
        return self.size

class CNN_MNIST(nn.Module):
    def __init__(self):
        super(CNN_MNIST,self).__init__()
        self.layer_conv1 = nn.Conv2d(1,8,5) #24*24
        self.layer_actv1 = nn.ReLU()
        self.layer_pool1 = nn.MaxPool2d(2,2) #12*12
        self.layer_conv2 = nn.Conv2d(8,64,3) #10*10
        self.layer_actv2 = nn.ReLU()
        self.layer_pool2 = nn.MaxPool2d(2,2) #5*5
        self.layer_fc1 = nn.Linear(64*5*5,1024)
        self.layer_fcr1 = nn.ReLU()
        self.layer_fc2 = nn.Linear(1024,256)
        self.layer_fcr2 = nn.ReLU()
        self.layer_fc3 = nn.Linear(256,40)

    def forward(self,x):
        x = self.layer_conv1(x)
        x = self.layer_actv1(x)
        x = self.layer_pool1(x)
        x = self.layer_conv2(x)
        x = self.layer_actv2(x)
        x = self.layer_pool2(x)
        x = x.view(-1,64*5*5)
        x = self.layer_fc1(x)
        x = self.layer_fcr1(x)
        x = self.layer_fc2(x)
        x = self.layer_fcr2(x)
        x = self.layer_fc3(x)
        return x

# model = CNN_MNIST().to("cpu")
model = torch.load("./model_mnist_rot.pth")
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
train_dataset = MNISTRot(mnist_dataset,10000)
train_dataloader = torchdata.DataLoader(train_dataset,batch_size=100)

def model_train(dataloader,model,optim_fn,loss_fn):
    size = len(dataloader.dataset)
    num_acc = 0
    num_acct = 0
    tot_loss = 0
    time_start = time.time()
    for batch,(dX,dY) in enumerate(dataloader):
        dX = dX.to("cpu")
        dY = dY.to("cpu")
        pY = model(dX)
        loss = loss_fn(pY,dY)
        optim_fn.zero_grad()
        loss.backward()
        optim_fn.step()
        tot_loss += loss.item()
        num_acc = torch.eq(dY, pY.argmax(dim=1)).sum().float().item()
        num_acct = num_acct + num_acc
        time_cur = time.time()
        print(f"Batch {batch} of {len(dataloader)} ends, time_elapsed={int((time_cur - time_start) * 100) / 100}s,"
              f" time_remaining={int(int(time_cur - time_start) / (batch + 1) * (len(dataloader) - batch - 1) * 100) / 100}s,"
              f" loss={loss.item()},"
              f" accuracy={num_acc / dX.shape[0]}")
    return tot_loss, num_acct / size

accuvm = 0.9943
for i in range(150):
    lossv, accuv = model_train(train_dataloader,model,optimizer,loss)
    print(f"Epoch {i} ends, Loss={lossv}, Acc={accuv}")
    if accuv>accuvm:
        accuvm=accuv
        torch.save(model,r"./model_mnist_rot-2.pth")
        print("Model Saved")
