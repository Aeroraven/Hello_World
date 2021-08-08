import torch
import torch.nn as nn
import numpy as np


# Torch function - Registering Buffer
class HelloWorld(nn.Module):
    def __init__(self):
        super(HelloWorld, self).__init__()
        self.register_buffer("queue", torch.randn(1, 10))
        print(self.queue)

    def forward(self, x):
        return x


model = HelloWorld().to("cpu")

# Torch function - Lp Normalization
tensorx = torch.tensor([[1., 2, 3, 4], [5, 6, 7, 8]])
print("BEFORE:")
print(tensorx)
tensorx = nn.functional.normalize(tensorx, p=1, dim=0)
print("AFTER:")
print(tensorx)
