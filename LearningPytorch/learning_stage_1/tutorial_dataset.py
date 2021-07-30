import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

train_data = torchvision.datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor()
) # Load the dataset
rows, cols = 4, 4 # Size of a Matplotlib Subplot
figrows, figcols = 28, 28 # Size of a training data in MNIST
figure = plt.figure(figsize=(figrows,figcols))
for i in range(rows*cols):
    idx = np.random.randint(0,len(train_data)) # Randomly choose a data item
    data_img,data_label = train_data[idx] # Select the active data item
    figure.add_subplot(rows,cols,i+1) # Set the active subplot
    plt.imshow(data_img.squeeze(),cmap="gray") # Set the image
    plt.title(str(data_label)) # Set tht title
plt.show()

# Tensor Operation - Squeeze
tensor_x = torch.ones((3,3,1,))
tensor_y = torch.ones((3,3,1,)).squeeze()
print(tensor_x)
print(tensor_y)

# Prepare with DataLoader
train_dataloader = torch.utils.data.DataLoader(
    train_data,
    batch_size=100,
    shuffle=True
)

# Python Iterables
t = iter(np.linspace(1,10,10).tolist())
while True:
    try:
        x = next(t)
        print(x)
    except StopIteration:
        break
