import torch
import torchvision
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as albu

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    classes="4"
)
print(model.encoder)
