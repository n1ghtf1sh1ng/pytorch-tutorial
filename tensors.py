import torch
import numpy as np

# Tensors can be created directly from data. The data type is automatically inferred.
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
print(f'x_data: {x_data}\n')

# Tensors can be created from NumPy arrays (and vice versa - see Bridge with NumPy).
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(f'x_np: {x_np}\n')

# The new tensor retains the properties (shape, datatype) of the argument tensor, unless explicitly overridden.
x_ones = torch.ones_like(x_data)  # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float)  # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")

# shape is a tuple of tensor dimensions. In the functions below, it determines the dimensionality of the output tensor.
shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")
