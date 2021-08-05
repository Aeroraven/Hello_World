import torch
import torchvision
import numpy as np
from torch import Tensor

x = torch.linspace(0, np.pi, 100)
x = np.array(x)
y = np.sin(x)
x = x.tolist()
y = y.tolist()
data = list(zip(x, y))
# Tensor from Python List
tensor_a = torch.tensor(data)

# Tensor from Numpy Array
data_np = np.array(data)
tensor_b = torch.tensor(data_np)

# Tensor from other tensors
tensor_ones = torch.ones_like(tensor_a)  # Retain the shape and replace all elements with 1

# Set the dimensions of a tensor
tensor_shape = (4, 3,)

# Generate tensors
tensor_random = torch.rand(tensor_shape)
tensor_ones = torch.ones(tensor_shape)
tensor_zeros = torch.zeros(tensor_shape)

# Tensor Attributes
print("Shape:" + str(tensor_random.shape))
print("Data Type:" + str(tensor_random.dtype))
print("Device:" + str(tensor_random.device))

# Tensor Operations - Tensor Slicing
print(tensor_random[0][1])  # Element (0,1)
print(tensor_random[0, 1])  # Element (0,1)
print(tensor_random[0])  # Row 0
print(tensor_random[:, 0])  # Column 0
print(tensor_random[:, -1])  # Column C-1
print(tensor_random[::2, :])  # Odd rows
print(tensor_random[1::2,:]) # Even rows

# Tensor Operations - Tensor Joining
tensor_joined = torch.cat([tensor_random,tensor_random],dim=0) # Append rows
print(tensor_joined)
tensor_joined = torch.cat([tensor_random,tensor_random],dim=1) # Append columns
print(tensor_joined)


tensor_p: Tensor = torch.tensor(np.array([[(i+1)*1 for j in range(5)] for i in range(5)]))
# Tensor Operations - Transpose
print("Transpose")
print(tensor_p.T)
# Tensor Arithmetic Operations - Add
print(tensor_p+tensor_p)
# Tensor Arithmetic Operations - Matmul
print(tensor_p.matmul(tensor_p.T)) # Multiply by its transposed result
print(tensor_p @ tensor_p.T)
# Tensor Arithmetic Operations - Element-wise Product
print(tensor_p * tensor_p)
print(tensor_p.mul(tensor_p))
# Tensor Arithmetic Operations - Scalar Operations
print(tensor_p*10)
print(tensor_p+10)
print(tensor_p**2)
print(torch.sqrt(tensor_p))

# Tensor Operations - Sum
print(tensor_p.sum()) # Calculate the sum of all tensor elements
print(tensor_p.sum().item()) # Convert the sum to Python number

# Tensor In-place Operations - Duplication
tensor_x = torch.empty(tensor_p.shape)
print("Tensor Copy: Before")
print(tensor_x)
tensor_x.copy_(tensor_p)
print("Tensor Copy: After")
print(tensor_x)

# Tensor In-place Operations - Transposition
tensor_x = torch.empty(tensor_p.shape)
tensor_x.copy_(tensor_p)
print("Tensor Transposition: Before")
print(tensor_x)
tensor_x.t_()
print("Tensor Transposition: After")
print(tensor_x)

# Tensor In-place Operations - Add
tensor_x = torch.ones(tensor_p.shape)
print("Tensor In-place Add: Before")
print(tensor_x)
tensor_x.add_(2)
print("Tensor In-place Add: After")
print(tensor_x)

# Tensor Operation - Convert to Numpy Array
tensor_x = torch.ones(tensor_p[:,0].shape)
print(tensor_x.numpy())
print(type(tensor_x.numpy()))

# Tensor Operation - Convert from Numpy Array
ndarray_x = np.linspace(1,10,10)
tensor_x = torch.from_numpy(ndarray_x)
print(tensor_x)
