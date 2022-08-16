#print(!nvidia-smi)

"""
resource PyTorch Deep Learning project (github)
"""


import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print(torch.__version__)

#Introduction to Tensors: represent numberic data in
#Scalar >>> pytorch tensors are created using torch.tensor()
scalar = torch.tensor(6)
print(scalar)
print(scalar.ndim)
print(scalar.item())



#Vector : created
vector = torch.tensor([7,7])
print(vector)
print(vector.ndim, ' dimensions')
print(vector.shape, ' shape')

#Matrix :
MATRIX = torch.tensor([[7, 8],
                       [5, 9]])

print(MATRIX)
print(MATRIX.shape, ' shape and: ', MATRIX.ndim, '  dimensions')
print(MATRIX[1] , '  second raw since matrix starts wiht index 0 ')


#TENSOR
TENSOR = torch.tensor([[[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]]])

print(TENSOR.shape, ' shape and: ', TENSOR.ndim, '  dimensions')
print(TENSOR[0][1][1] , '  second raw since matrix starts wiht index 0 ')
print(TENSOR[0], ' zero dimension')
print(TENSOR[0][:][:], ' all with : : in the second and third dimensions')


TENSOR1 = torch.tensor([[[[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]]]])

print(TENSOR1.shape, ' shape and: ', TENSOR1.ndim, '  dimensions')
print(TENSOR1[0][0][1] , '  second raw since matrix starts wiht index 0 ')
print(TENSOR1[0], ' zero dimension')
print(TENSOR1[0][:][:], ' all with : : : in the second and third and forth dimensions')




#Random TENSORs  >>>  Why?? is a big part on PyTorch because the way many Neural Network learned
#They start with random numbers and adjust those random numbers to learn better patterns of data.

#create a random TENSOR wiht size/shape

random_tensor = torch.rand(3, 4)
print(random_tensor)
print(random_tensor.ndim, ' dimensions and ', random_tensor.shape, ' shape')

random_tensor10 = torch.rand(10, 10)
print(random_tensor10)
print(random_tensor10.ndim, ' dimensions and ', random_tensor10.shape, ' shape')

random_tensor10_3D = torch.rand(1,10, 10)
print(random_tensor10_3D)
print(random_tensor10_3D.ndim, ' dimensions and ', random_tensor10_3D.shape, ' shape')


random_tensor10_3D = torch.rand(10,10, 10)
print(random_tensor10_3D)
print(random_tensor10_3D.ndim, ' dimensions and ', random_tensor10_3D.shape, ' shape')

#creat a random tensor with similar shape to an image random_tensor10_3D
random_image_size_tensor = torch.rand(size=(3, 224, 224)) #color channels, height, width,
print(random_image_size_tensor)
print(random_image_size_tensor.shape, ' shape and ', random_image_size_tensor.ndim, ' dimensions')


random_image_size_tensor_bw = torch.rand(size=(1, 255, 255))
print(random_image_size_tensor_bw)
print(random_image_size_tensor_bw.shape, ' shape and: ', random_image_size_tensor_bw.ndim, '  dimensions')


#exercises #################################
random_tensor_mine = torch.rand(5, 6)
print(random_tensor_mine.ndim, ' dimensions and ', random_tensor_mine.shape, ' shape')


#create a tenosr of zeros or ones
zero = torch.zeros(size=(5, 6))
print(zero.ndim, ' dimensions ')

masked_tensor = random_tensor_mine[0:2][1:3] * zero
print(masked_tensor)

random_tensor_mine = random_tensor_mine * zero
print(random_tensor_mine)


#ones TENSOR
ones = torch.ones(size=(5, 6))
print(ones.dtype)

#How to create a range of tensors and tensors-like

print('TOrch.range')
range_tensor = torch.arange(0, 10)
print(range_tensor)
range_tensor = torch.arange(0, 11)
print(range_tensor)


range_tensor_arange = torch.arange(start=0, end= 1000, step= 99)
print(range_tensor_arange)

print('Tenosrs like'
      )

ten_zeros = torch.zeros_like(input=range_tensor)
print(ten_zeros)

a = torch.ones_like(range_tensor)
print(a)


print('Tensors datatype ...')
print("Tenosrs not right datatype,  right shape , right device")
float32_tensor = torch.tensor([3.0, 6.0, 9.0],
                              dtype=None,#datatype float, 16bitfloating precession single precision is 32bit
                              device=None, #CPU
                              requires_grad=False) # Packprogation parameters
print(float32_tensor, '\n its type ',
      float32_tensor.dtype)


float_16_tensors = float32_tensor.type(torch.float16)
print(float_16_tensors ,float_16_tensors.dtype)

float_16_32_tensors = float32_tensor * float32_tensor
print(float_16_32_tensors, float_16_32_tensors.dtype)


int_32_tensor = torch.tensor([1,2,3], dtype=torch.int32)
print(int_32_tensor)

print(float32_tensor * int_32_tensor)


print("check the shape, datatype, device. tenosr.dtype \
tensor.shape as an attribute or tensor.size() as a function \
tenosr.device \
")


#create a float32_tensor
some_tensor = torch.rand(3, 5)
print(some_tensor)
print(f"datatype tensor: {some_tensor.dtype}")
print(f"shape/shape tensor: {some_tensor.size()}")
print(f"device tensor: {some_tensor.device}")



print("Manipulating tensors  ... ")

print(" Tensor operations : "
      "Addition, Subtraction"
      "Multiplication Elementwise multiplication"
      "Division"
      "MATRIX Multiplication")

#create a tenosr.device
tensor = torch.tensor([1, 3, 3])
print(tensor)
tensor = tensor + 10
print(tensor)
tensor = tensor * 10
print(tensor)
tensor = tensor - 10
print(tensor)
tensor = tensor / 10
print(tensor)

#Try out PyTorch in built function

tensor1 = torch.mul(tensor, 10)
print(tensor1)

tensor1 = torch.add(tensor, 10)
print(tensor1)


#Matrix Multiplication

print('Two main ways to perform Multiplication in neural Network'
      'Elementwise Multiplication'
      'MATRIX Multiplication (most common in neural Network (dotproduct)')

tensor_mul = tensor * tensor

print(f"{tensor} '*' {tensor} = {tensor_mul}")

tensor_mul_dot = torch.dot( tensor, tensor)

print(f"{tensor} '*' {tensor} = {tensor_mul_dot} , dotproduct ")

print("MATRIX Multiplication")
tensor_matrix_mul = torch.matmul(tensor, tensor)

print(f"{tensor} '*' {tensor} = {tensor_matrix_mul}")

import time
value = 0
start = time.time()
for i in range(len(tensor)):
    value += tensor[i] * tensor[i]
end = time.time()
print(f"Time consueming is {end- start}  with value {value}")

start = time.time()
tensor_matrix_mul = torch.matmul(tensor, tensor)
end = time.time()

print(f" {tensor} '*' {tensor} = {tensor_matrix_mul}")
print(f"Time consueming is {end- start}")


print("One of the most common errors in pytorch is shape error:"
      "nner dimensions must match @ >>> matrix multiplication. >>> torch.matmul "
      "(3 ,2) @ (3, 2) wont work "
      "(2,3) @(3,2) works because the inner dimension matches the"
      "................................................................"
      "The resulting matrix has the shape is the outter dimensions of both matris"
      "(2, 3) @ (3, 6) >>>> (2, 6)")

t1 = torch.rand(size=(5, 3))
print(t1.shape)
t2 = torch.rand(size=(3, 3))
print(t2.shape)
print(torch.matmul(t1, t2))

t1 = torch.rand(size=(10, 3))
print(t1.shape)
t2 = torch.rand(size=(3, 10))
print(t2.shape)
print(torch.matmul(t1, t2))

print("Shape of matrix multiplication "
      )
tensor_A = torch.tensor([[1 ,2],
                         [4, 6],
                         [6, 7],
                         [4, 4]])
print(tensor_A.shape)
tensor_B = torch.tensor([[1 ,2],
                         [4, 6],
                         [6, 7],
                         [4, 5]])
print(tensor_B.shape)

result = torch.mm(tensor_A.T, tensor_B)
print(result.shape)

print("To fix tenosr shpae issues, we can Manipulating the shape of one of our tenosr sh"
      "using transposed"
      "a transpose swich the axes or dimensions of a given tensor")


#The matrix multiplication operations works when tenosr_A is transposed.

print(tensor_B.T)


print("Tensor aggregation :: min, max, mean, median:")
x = torch.arange(0, 100, 10)
print(x.min(), torch.min(x), ' min')
print(x.max(), torch.max(x), ' max')
print(x.sum(), torch.sum(x), ' sum')
print(x.median(), torch.median(x), ' median')
print(x.dtype)
x = x.type(torch.float16)
print(x.dtype)
print("torch.mean function require a a tensor of float16 or 32")
print(x.mean(), torch.mean(x), ' mean')

print("........... torch.argmin ........ torch.argmax ..... for indexing min/max   ..... finding the posisitons")


print(x)
print(x.argmin(), '  minmum value\'s index')
print(x.argmax(), '  maximum value\'s index ')



print("reshaping, stacking, squeezing, unsqueesing tenosors")
print("torch.reshape : reshapes an input tensor into a defined shape")
print("view - return a view of an input tensor of certain shape but keep the same memory as the original tensor share the same memory")
print("stacking: combine multiple tensors on top of each other/ concatinate a sequence of tensors along a new dimension :")
print("squeeze - remove all or one dimensions from a tensor /// unsqueeze add all or one dimensions to a tensor ")
print("permute - return a view of the input with dimensions permuted (swapped) in a certain way")

#create a tenosr and reshape it
print("#create a tenosr and reshape it")
x = torch.arange(1., 10.)
print(x, x.shape)

x_reshaped = x.reshape(1, 9)
print(x_reshaped, x_reshaped.shape)

x_reshaped = x.reshape(9, 1)
print(x_reshaped, x_reshaped.shape)



#create a tenosr
x = torch.arange(1., 11.)
print(x, x.shape)

x_reshaped = x.reshape(2, 5)
print(x_reshaped, x_reshaped.shape)

x_reshaped = x.reshape(5, 2)
print(x_reshaped, x_reshaped.shape)


print("#create a tenosr and review it")

z = x.view(1, 10)
print(z, z.shape, ' z shared the same memory of x')


print("#changing z changes x, since a view of the tensor shares the same memory of the original tensor")

z[:, 0] = 5.55
print(z, '\n', x, ' z shared the same memory of x')


print("\n\n Stack tensors on top of each others .... ")

x_stacked = torch.stack([x,x,x,x], dim=0)
print(f" The original tensor \n {x.shape}, ' as \n', {x}, '\n staced vertically 4x and got {x_stacked.shape} as \n  {x_stacked}")



x_stacked = torch.stack([x,x,x,x], dim=1)
print(f" The original tensor \n {x.shape}, ' as \n', {x}, '\n staced horisontally 4x and got {x_stacked.shape} as \n  {x_stacked}")



x_stacked = torch.vstack((x,x))
print(f" The original tensor \n {x.shape}, ' as \n', {x}, '\n staced vertically 4x and got {x_stacked.shape} as \n  {x_stacked}")



x_stacked = torch.hstack((x,x))
print(f" The original tensor \n {x.shape}, ' as \n', {x}, '\n staced horisontally 4x and got {x_stacked.shape} as \n  {x_stacked}")





print("squeeze - remove all or one dimensions from a tensor /// unsqueeze add all or one dimensions to a tensor \ Returns a tensor with all the dimensions of input of size 1 removed.")
x = torch.zeros(2, 1, 2, 1, 2)
print(x)
print(x.size(),'   this ithe origial dimension ..')
y = torch.squeeze(x)
print(y.size(),'   the squeezed with no given dimension ..')

y = torch.squeeze(x, 0)
print(y.shape, '   it will squeeze on the zero dimension ..')

y = torch.squeeze(x, 1)
print(y.size(), '   it will squeeze on the one dimension ..')

print(f"x_reshaped {x_reshaped} has shape of \n {x_reshaped.shape} ")
y_reshaped = x_reshaped.reshape(1, 10)
print(f"y_reshaped {y_reshaped} has shape of \n {y_reshaped.shape} ")

xs = x_reshaped.squeeze()
ys = y_reshaped.squeeze()
print(f"x_reshaped.squeezed {xs} has shape of \n {xs.shape} and dim {xs.ndim} ")
print(f"y_reshaped.squeezed {ys} has shape of \n {ys.shape} and dim {ys.ndim}")


print("torch.unsqueeze() adds a single dimension to a tensor at a specific dim")

print(f"Previous target :{ys}")
print(f"Previous shape :{ys.shape}")

print("add an extra zero/horizontal dimension with unsqueeze ... ")
yuns = ys.unsqueeze(dim=0)
print(f"\nNew tensor with zeros/ horizontal dimension : {yuns}")
print(f"New tenosr shape :{yuns.shape}")

print("add an extra one/vertical dimension with unsqueeze ... ")
yuns = ys.unsqueeze(dim=1)
print(f"\nNew tensor with one/ vertical dimension : {yuns}")
print(f"New tenosr shape :{yuns.shape}")


print(f"\n torch.permute - rearranges the dimensions of a tensor to a specific order\n Returns a view of the original tensor input with its dimensions reorderred/ permuted.")

x = torch.randn(2, 3, 5)
print(x)
print(x.size())
print(f"torch.permute(Tensor, (x,y,z)).size() return the desired ordering of dimensions ... ")

x_permuted = torch.permute(x, (1,0,2))
print(f" The permuted tensor is \n {x_permuted} has the shape of {x_permuted.size()} :)")

print(f"permuted normally used on images ...")
x_original = torch.rand(size=(224,224,3)) # [height, widht, color_channels]
print(x_original.shape, '  channel last')

#permpute the original tensor to rearange teh axis (dim) order so to be [color_channel, height, width]

x_permuted = x_original.permute(2, 0, 1) # shifts axis 0>1, 1>2, 2>0
print(x_permuted.shape, '  channel first')



x_original[0, 0, 0] = 88888
print(x_original[0, 0, 0] )
print(x_permuted[0, 0, 0] )

print(f"Indexing selected data from tensors >>> indexing with PyTorch is similar to indexing with NumPy ...")

x = torch.arange(1, 10).reshape(1, 3, 3)
print(f'x tenosr is \n {x} and \n its shape is {x.shape}')

#Let's index in our new tensor
print(f'x tenosr is   its zero index is {x[0]} of the first dimension')


print(f'x tenosr is  its zero index is {x[0][0]} of the middle dimension')

print(f'x tenosr is   its zero index is {x[0, 0]} of the middle dimension')

print(f'x tenosr is  its zero index is {x[0, 0, 0]} of the last dimension')

print(f'x tenosr is   its zero index is {x[0][ 0][ 0]} of the last dimension')

print(f'x tenosr is \n {x} and \n its 9 value  is {x[0, 2, 2]} of the last dimension')



print("use a : to select all the dimensions")

print(x[:, 0], '  to select all of the 0th dimension of the zeros elemnt')
print(x[:, :, 1], '  get all values of 0th and 1st dimension but only index 1 of 2nd dimension')
print(x[:, :, 2], '  get all values of 0th and 1st dimension but only index 2 of 2nd dimension')
print(x[:, 1, :], '  get all values of 0th and index 1th of the 1st dimension all 2nd dimension')


print("Get all the values of the zero dimension but only the 1 index value of the 1st and 2nd dimension ")
print(x[:, 1, 1])
print("Get index 0 of 0th and 1st dimension and all values of 2nd dimension ..")
print(x[0,0, : ])

print(x)
print(f"Index on x to return {9}: >> {x[0,2,2]} \2n "
      f"Index on x to return {3, 6, 9}:  >> {x[:,:,2]}")

print(f"Index on x to return {7}: >> {x[0,2,0]} \2n "
      f"Index on x to return {2, 5, 8}:  >> {x[:,:,1]}")
print(f"Index on x to return {6}: >> {x[0,1,1]} \2n "
      f"Index on x to return {7, 8, 9}:  >> {x[:,2,:]}")



print(f"\n \n PyTorch tensors Numpy \n Numpy is a popular scientific Python computing library. "
      f"\n and because of this PyTorch has functionality to interact with it < data"
      f"\n Numpy to a PyTorch tenosr>>> torch.from_numpy(ndarray)"
      f"\n PyTorch tensor >> Numpy >>> torch.tensor.numpy() ")

array = np.arange(1.0, 8.0)
print(array.dtype)
tensor = torch.from_numpy(array)
print(array, 'Numpy array \n '
             'Tenosr is ', tensor)
tensor = torch.from_numpy(array).type(torch.float16)
print(array, 'Numpy array \n '
             'Tenosr is ', tensor)

print("When converting from Numpy to PyTorch >>> pytorch reflect the datatype of Numpy"
      "so you can use .type(torch.float16)")

print('Change the value of the array .... ')

array = array + 10
print(array, 'Numpy array \n '
             'Tenosr is ', tensor)

print("Tensor to Numpy array .... ")
tensor = torch.ones(7)
nump_tensor = tensor.numpy()
print(tensor, tensor.dtype, nump_tensor, type(nump_tensor), nump_tensor.dtype)


print("Change the tensor, what happens to numpy_tensor .... \n"
      "They aren't share memory ...")

tensor = tensor + -10
print(tensor, 'tensor  \n '
             'array is ', array)


print(f"PyTorch Reproducibilty >>> random out of random .... \n "
      f"In short a neural network learn by ::: \n "
      f"start with a random number >>> tenosr operations >>> update random numbers \n "
      f"to try and make them better representations of the data >> again ... again ..\n "
      f"\n ")

x1 = torch.rand(3, 3)
print(x1)
x3 = torch.rand(3, 3)
print(x3)
x2 = torch.rand(3, 3)
print(x2)

print(x1 == x2)

print(f"To reduce a randomness in neural network and PyTorch .. comes using a random seed \n "
      f"A random is flavour of randomenss   ")

# create random tensors ... make a random but reproducibilty tensor
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)


x1 = torch.rand(3, 3)
print(x1)
torch.manual_seed(RANDOM_SEED)
x3 = torch.rand(3, 3)
print(x3)
torch.manual_seed(RANDOM_SEED)
x2 = torch.rand(3, 3)
print(x2)

print(x1 == x2)

print("torch.manual_seed(RANDOM_SEED)" * 3)


print("Running tensors/ PyTorch on GPUs, faster computation \n "
      "Nvidia, CUDA, hody dor (good) ... ")

print(f"Check  the avialibility of PyToch has access to GPUs Yes or No >>>>>  {torch.cuda.is_available()} :)")

print(f"Setup device agnostic code .... ")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"if we have a GPU then device should be 'cuda' \n ")
if device == 'cuda':
    print("yes we have a GPU ")
else:
    print("no we do not have a GPU ")


print(f"Count how many devices 'GPUs'  >>> {torch.cuda.device_count()} devices :) ")


import argparse
import torch

parser = argparse.ArgumentParser(description='PyTorch Example')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
args = parser.parse_args()
args.device = None
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
    print(args.device, '   we have a GPUs')
else:
    args.device = torch.device('cpu')
    print(args.device, ' unforturnately we do not have a device ')


print(f"Putting tenosrs and models on the GPU \n "
      f"To accelerate our computation ... ")

tensor = torch.tensor([1, 2, 3], device='cpu')
print(tensor, tensor.device, '   where our tensor raised by defualt ...')
tensor = torch.tensor([1, 2, 3], device=args.device)
print(tensor, tensor.device, '   where our tensor shifted to a GPUs ...')

tensor = torch.tensor([1, 2, 3]).to(device)
print(tensor, tensor.device, '   where our tensor shifted to a GPUs ...')


print("Moving Tenosrs back to a GPU >>> if tenosr in a GPU, can't transfrom it to a numpy ..")
print(f"tensor.numpy() \n "
      f"TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.")

print("To fix the GPU tensor with NumPy issue, we can first set it to a cpu  ... ")
tensor_back_on_cpu = tensor.cpu()
print(tensor, '      Tensor on GPU >>>>> ----  ==== ')
print(tensor_back_on_cpu , '      Tensor on cpu >>>>> ----  ==== ')
print(tensor_back_on_cpu.numpy() , '      Tensor on CPU >>>>> ----  ==== ')

print(f"Count how many devices 'GPUs'  >>> {torch.cuda.device_count()} devices :) \n "
      f"The devices names are: {torch.cuda.get_device_name()} ")

print(torch.distributed.is_available())
print(torch.distributed.is_mpi_available())
print(torch.distributed.is_nccl_available())
print(torch.distributed.is_torchelastic_launched())


device_0 = torch.cuda.get_device_name(0)
print(device_0)
device_0 = torch.cuda.device(0)
print(device_0)

device_1 = torch.cuda.get_device_name(1)
print(device_1)
device_1 = torch.cuda.device(1)
print(device_1)

device_2 = torch.cuda.get_device_name(2)
print(device_2)
device_2 = torch.cuda.device(2)
print(device_2)

device_3 = torch.cuda.get_device_name(3)
print(device_3)
device_3 = torch.cuda.device(3)
print(device_3)


manual_seed = torch.manual_seed(RANDOM_SEED)
tensor_A = torch.rand(2, 2)
manual_seed = torch.manual_seed(RANDOM_SEED)
tensor_B = torch.rand(2, 2)
print(tensor_A == tensor_B)

manual_seed = torch.manual_seed(RANDOM_SEED)
tensor_C = torch.rand(2, 2)
manual_seed = torch.manual_seed(RANDOM_SEED)
tensor_D = torch.rand(2, 2)
print(tensor_C == tensor_D)

device_0 = 'cuda:0'
device_1 = 'cuda:1'
device_2 = 'cuda:2'
device_3 = 'cuda:3'



print(f"{tensor_A} as tensor on  CPU .... \n "
      f"{tensor_A.to(device_0)}, in GPU \n "
      f"{tensor_A.cpu()},  back on CPU \n "
      f"{tensor_A.numpy()}, as numpy array on CPU  ...  ")

print(f"{tensor_B} as tensor on  CPU .... \n "
      f"{tensor_B.to(device_1)}, in GPU \n "
      f"{tensor_A == tensor_B} \n"
      f"{tensor_B.cpu()},  back on CPU \n "
      f"{tensor_B.numpy()}, as numpy array on CPU  ...  ")

print(f"{tensor_C} as tensor on  CPU .... \n "
      f"{tensor_C.to(device_2)}, in GPU \n "
      f"{tensor_C == tensor_B} \n"
      f"{tensor_C.cpu()},  back on CPU \n "
      f"{tensor_C.numpy()}, as numpy array on CPU  ...  ")

print(f"{tensor_D} as tensor on  CPU .... \n "
      f"{tensor_D.to(device_3)}, in GPU \n "
      f"{tensor_D == tensor_A} \n"
      f"{tensor_D.cpu()},  back on CPU \n "
      f"{tensor_D.numpy()}, as numpy array on CPU  ...  ")