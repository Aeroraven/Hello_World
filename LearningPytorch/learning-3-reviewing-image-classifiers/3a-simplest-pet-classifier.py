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


def pet_augmentation():
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
                 augmentation: callable = None):
        if classes is None:
            classes = []
        self.data_name = []
        self.classes = classes
        self.augmentation = augmentation
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
        # image = cv.imread(self.data_name[item][2] + "/" + self.data_name[item][0])
        image = pimg.open(self.data_name[item][2] + "/" + self.data_name[item][0]).convert("RGB")
        image = np.array(image)
        label = self.data_name[item][1]
        # try:
        #     image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        # except cv.error:
        #     print(self.data_name[item][2] + "/" + self.data_name[item][0])
        if self.augmentation:
            sp = self.augmentation(image=image)
            image = sp['image']
        image = np.array(image)
        image = image.transpose(2, 0, 1)
        image = torch.ByteTensor(image) / 256
        return image, label


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
        x = x.view(-1,16*22*22)
        x = self.layer_fc1(x)
        x = self.layer_fcr1(x)
        x = self.layer_fc2(x)
        x = self.layer_fcr2(x)
        x = self.layer_fc3(x)
        return x


def model_train(dataloader: torchdata.DataLoader,
                model,
                loss_fn,
                opt_fn):
    size = len(dataloader.dataset)
    batches = 0
    tot_loss = 0
    num_acc = 0
    num_acct = 0
    time_start = time.time()
    for batch, (X, Y) in enumerate(dataloader):
        batches = batches + 1
        X = X.to("cpu")
        Y = Y.to("cpu")
        pY = model(X)
        loss = loss_fn(pY, Y)
        opt_fn.zero_grad()
        loss.backward()
        opt_fn.step()
        tot_loss = tot_loss + loss.item()
        num_acc = torch.eq(Y, pY.argmax(dim=1)).sum().float().item()
        num_acct = num_acct + num_acc
        time_cur = time.time()
        print(f"Batch {batch} of {len(dataloader)} ends, time_elapsed={int((time_cur-time_start)*100)/100}s,"
              f" time_remaining={int(int(time_cur-time_start)/(batch+1)*(len(dataloader)-batch-1)*100)/100}s,"
              f" loss={loss.item()},"
              f" accuracy={num_acc / X.shape[0]}")
    return tot_loss, num_acct / size

time_st = time.time()
data_dir = r"C:\Users\Null\Desktop\Internship\Demo\data\PetImages"
train_dir = data_dir
train_dataset = ArvnDataset_Pet(train_dir, ["Cat", "Dog"], pet_augmentation())
train_dataloader = torchdata.DataLoader(train_dataset, batch_size=50)
# model = PetClassifier().to("cpu")
model = torch.load("./model_pet.pth")
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
accuvm = 0
for i in range(10):
    print(f"Epoch {i} starts")
    lossv, accuv = model_train(train_dataloader, model, loss, optimizer)
    print(f"Epoch {i} ends, Loss={lossv}, Acc={accuv}")
    if accuv > accuvm:
        accuvm = accuv
        torch.save(model, "./model_pet.pth")
        print("Model Saved")

print(f"Time Elapsed:{int(time.time()-time_st)}s")
