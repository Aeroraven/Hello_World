import torch
import torch.nn as nn
import numpy as np


class HelloWorld(nn.Module):
    def __init__(self):
        super(HelloWorld, self).__init__()
        self.register_buffer("queue", torch.randn(1, 10))
        print(self.queue)

    def forward(self, x):
        return x


model = HelloWorld().to("cpu")
