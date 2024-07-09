import torch

# Initializing Tensor
device = "cuda" if torch.cuda.is_available() else "cpu"
my_tensor = torch.tensor([[1,2,3],[4,5,6]], dtype=torch.float32, device = device, requires_grad=True)


# Other common initialization methods
x = torch.empty(size=(3,3))
x = torch.zeros((3,3))
x = torch.rand((3,3))
x = torch.ones((3,3))
x = torch.eye(5,5)
x = torch.arange(start=0, end=5, step=1)
x = torch.linspace(start=0.9, end=1, steps=10)
x = torch.empty(size=(1,5)).normal_(mean=0, std=1)
x = torch.empty(size=(1,5)).uniform_(0,1)
x = torch.diag(torch.ones(3))

# How to initialize and convert tensors to other types (int, float, double)
tensor = torch.arange(4)
print(tensor.bool()) # boolean
print(tensor.short()) # int16
print(tensor.long()) # int64
print(tensor.half()) # float16
print(tensor.float()) # float32
print(tensor.double()) # float64

# Array to Tensor conversion and vice-versa
import numpy as np
np_array = np.zeros((5,5))
tensor = torch.from_numpy(np_array)
np_array_back = tensor.numpy()

# Tensor Math & Comparison Operations

x = torch.tensor([1,2,3])
y = torch.tensor([9,8,7])

# Addition
z1 = torch.empty(3)
torch.add(x,y, out=z1)
z2 = x + y
z3 = torch.add(x,y)

# Subtraction
z = x - y

# Division
z = torch.true_divide(x,y) # element wise division
z = x / y

# Inplace Operations
t = torch.zeros(3)
t.add_(x) # t = t + x
t += x

# Exponentiation
z = x.pow(2)
z = x ** 2

# Simple Comparison
z = x > 0
z = x < 0

# Matrix Multiplication
x1 = torch.rand((2,5))
x2 = torch.rand((5,3))
x3 = torch.mm(x1,x2) # 2x3

# Matrix Exponentiation
matrix_exp = torch.rand(5,5)
matrix_exp.matrix_power(3)

# Element wise multiplication
z = x * y
z = x.mul(y)

# Dot product
z = torch.dot(x,y)

# Batch Matrix Multiplication
batch = 32
n = 10
m = 20
p = 30

tensor1 = torch.rand((batch,n,m))
tensor2 = torch.rand((batch,m,p))
out_bmm = torch.bmm(tensor1, tensor2) # (batch,n,p)

# Example of Broadcasting =  expand the smaller tensor to the larger tensor, then perform the operation
x1 = torch.rand((5,5))
x2 = torch.rand((1,5))

z = x1 - x2
z = x1 ** x2

# Other useful tensor operations
sum_x = torch.sum(x, dim=0)
values, indices = torch.max(x, dim=0)
values, indices = torch.min(x, dim=0)
abs_x = torch.abs(x)
z = torch.argmax(x, dim=0)
mean_x = torch.mean(x.float(), dim=0)
z = torch.eq(x,y)
sorted_y, indices = torch.sort(y, dim=0, descending=False)
z = torch.clamp(x, min=0, max=10) # All values <0 set to 0 and values >10 set to 10

x = torch.tensor([1,0,1,1,1], dtype=torch.bool)
z = torch.any(x) # True
z = torch.all(x) # False

# Tensor Indexing
batch_size = 10
features = 25
x = torch.rand((batch_size, features))

# Get first examples features
print(x[0].shape) # or x[0,:]

# Get the first feature for all examples
print(x[:,0].shape)

# Fancy Indexing
x = torch.arange(10)
indices = [2,5,8]
print(x[indices])

x = torch.rand((3,5))
rows = torch.tensor([1,0])
cols = torch.tensor([4,0])
print(x[rows,cols])

# More Advanced Indexing
x = torch.arange(10)
print(x[(x < 2) | (x > 8)])
print(x[x.remainder(2) == 0])

# Useful Operations
print(torch.where(x > 5, x, x*2))
print(torch.tensor([0,0,1,2,2,3,4]).unique())
print(x.ndimension())

# Tensor Reshaping
x = torch.arange(9)
x_3x3 = x.view(3,3)
x_3x3 = x.reshape(3,3)

# Add a dimension
x = torch.arange(10)
print(x.unsqueeze(0).shape)
print(x.unsqueeze(1).shape)

x = torch.arange(10).unsqueeze(0).unsqueeze(1)
z = x.squeeze(0)

# Flatten the tensor
x = torch.arange(10).unsqueeze(0).unsqueeze(1)
z = x.view(-1)

# Concatenation
x1 = torch.rand((2,5))
x2 = torch.rand((2,5))
print(torch.cat((x1,x2), dim=0).shape)
print(torch.cat((x1,x2), dim=1).shape)

# Flattening
batch = 64
x = torch.rand((batch,2,5))
z = x.view(batch, -1)
z = x.reshape(batch, -1)

# Swapping Axes
x = torch.rand((2,3))
print(x.t())

# Linear Algebra
x1 = torch.rand((5,2,5))
x2 = torch.rand((1,5,3))
out  = x1 @ x2
print(out.shape)

# Einstein Summation => batch, n, m, p, q
x = torch.rand((10,3,4,5))
y = torch.rand((10,3,5,6))
z = torch.einsum('bnmp,bnql->bnpl', x, y)

# Gathering
dim = 0
src = torch.rand(4,4,6)
idx = torch.randint(low=0, high=4, size=(1,4,6))
out = src.zeros_like(size = (1,10,6))
for i in range(1):
    for j in range(10):
        for k in range(6):
            out[i,j,k] = src[i, idx[i,j,k], k]

out = torch.gather(src, dim, idx)

