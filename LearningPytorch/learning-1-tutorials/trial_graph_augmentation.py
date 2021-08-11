# Graphic augmentation - Composed augmentation
import albumentations as albu
import cv2 as cv
import torch
import matplotlib.pyplot as plt

arg_img = cv.imread("./images/augmentation_test.jpg", cv.IMREAD_COLOR)
arg_effects = [
    albu.HorizontalFlip(p=1),
    albu.VerticalFlip(p=1),
    # albu.ToSepia(p=1),
    albu.VerticalFlip(p=1),
    # albu.RandomRotate90(p=1)
    albu.ColorJitter(brightness=0.1,contrast=0.1,saturation=0.01,hue=0.1,p=1)
]

arg_composed = albu.Compose(arg_effects)
arg_img = arg_composed(image=arg_img)['image']
arg_img = cv.cvtColor(arg_img,cv.COLOR_BGR2RGB)
# cv.imshow("augmentation Test", arg_img)
# cv.waitKey(delay=0)
plt.imshow(arg_img)

# Transformation - To Tensor
# def to_tensor(x, **kwargs):
#     if isinstance(x, torch.Tensor):
#         return x.permute(2, 0, 1).to(torch.float)
#     return x.transpose(2, 0, 1).astype('float32')
#
#
# arg_img = cv.imread("./images/augmentation_test.jpg", cv.IMREAD_COLOR)
# arg_effects = [
#     albu.Lambda(image=to_tensor)
# ]
# arg_composed = albu.Compose(arg_effects)
# arg_img = arg_composed(image=arg_img)['image']
#

# # Graphic augmentation - Different Channel Orders Between OpenCV & PIL
# arg_img = cv.imread("./images/augmentation_test.jpg",cv.IMREAD_COLOR)
# arg_img = cv.cvtColor(arg_img,cv.COLOR_BGR2RGB)
# plt.imshow(arg_img)
