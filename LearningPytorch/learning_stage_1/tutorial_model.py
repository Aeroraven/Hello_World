import torch
import torchvision
import numpy as np

train_dataset = torchvision.datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor()
)

# Layers - Flatten Layer
flatten = torch.nn.Flatten()
flatten2 = torch.nn.Flatten(start_dim=0)
imgx,labelx = train_dataset[0]
imgxfl = flatten(imgx)
imtxfl2 = flatten2(imgx)
print(imgx.shape)
print(imgxfl.shape)
print(imtxfl2.shape)

# Layers - Linear Layer
flatten = torch.nn.Flatten()
imgx,labelx = train_dataset[0]
flatten_out = flatten(imgx)
linear = torch.nn.Linear(in_features=28*28,out_features=512)
linear_output = linear(flatten_out)
print(linear_output.shape)

# Layers - ReLU Layer
lst=[[np.random.normal() for i in range(5)] for j in range(5)]
tensor_x=torch.tensor(lst)
print(tensor_x)
print(torch.nn.ReLU()(tensor_x))

# Layers - SoftMax Layer
seq_layers = torch.nn.Sequential(
    flatten,
    linear,
    torch.nn.ReLU(),
    torch.nn.Linear(in_features=512,out_features=20),
    torch.nn.ReLU(),
    torch.nn.Linear(in_features=20,out_features=10)
)
imgx, labelx = train_dataset[0]
logits = seq_layers(imgx)
softmax = torch.nn.Softmax(dim=1)
probs = softmax(logits)
print(probs)