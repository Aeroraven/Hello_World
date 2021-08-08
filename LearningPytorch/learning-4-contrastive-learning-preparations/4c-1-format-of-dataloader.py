import torchvision.transforms as transforms
import torch
import moco.loader
import moco.builder
import torchvision.datasets as datasets
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
augmentation = [
    transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
]
traindir = r"D:\PetImages"
train_dataset = datasets.ImageFolder(
        traindir,
        moco.loader.TwoCropsTransform(transforms.Compose(augmentation)))
x = train_dataset[0]
print(x)
train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=3, shuffle=False,
        num_workers=0, pin_memory=True, sampler=None, drop_last=True)
y = None
for i,(images,_) in enumerate(train_loader):
    y = images
    break