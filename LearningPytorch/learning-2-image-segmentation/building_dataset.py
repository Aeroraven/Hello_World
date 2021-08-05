# Stage 2 period 1

import cv2 as cv
import torchvision as torchcv
import torch.utils.data as torchdata
import os
import numpy as np
import matplotlib.pyplot as plt

def visualize_help(**kwargs):
    n = len(kwargs)
    plt.figure()
    for i,(name,image) in enumerate(kwargs.items()):
        plt.subplot(1,n,i+1)
        plt.imshow(image)
    plt.show()

class LvDataset(torchdata.Dataset):
    def __init__(self,img_dir:str,mask_dir:str,class_idx:list):
        self.data_file_list = os.listdir(img_dir)
        self.data_path_list = [img_dir+'/'+x for x in self.data_file_list]
        self.mask_path_list = [mask_dir+'/'+x for x in self.data_file_list]
        self.classes = class_idx

    def __getitem__(self, item:int):
        image = cv.imread(self.data_path_list[item])
        image = cv.cvtColor(image,cv.COLOR_BGR2RGB)
        mask = cv.imread(self.mask_path_list[item],0)
        masks=[(mask==x) for x in self.classes]
        mask_extracted = np.stack(masks,axis=-1).astype('float')
        return image,mask_extracted

    def __len__(self):
        return len(self.data_file_list)

x_train_path = r"D:\liver2\liver2\test\imgs"
y_train_path = r"D:\liver2\liver2\test\masks"
train_dataset = LvDataset(x_train_path,y_train_path,[1])

rnd_x,rnd_y = train_dataset[24]
visualize_help(image=rnd_x,mask=rnd_y)