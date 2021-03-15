import torch
import torch.nn as nn
import torch.nn.functional as F


# Autograd:
#   torch.Tensor
#       .requires_grad = True  ::   track all operations
#       .backward() ::  at the end, can have all the gradients computed automatically
#       into a .grad attribute
#       .detach()  ::  stop a tensor from tracking history
#       wrap code in `with torch.no_grad()  ::  for no tracking( less memory )

#   Functions can create Tensors, Tensor then has .grad_fn to point to Function
#   .backward()  ::  returns?? the derivative


# Example of vector-Jacobian product
x = torch.randn(3, requires_grad=True)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2

print(y)

# Now in this case `y` is no longer a scalar. `torch.autograd`
#  could not compute the full Jacobian directly, but if we
#  just want the vector-Jacobian product, simply pass the
#  vector to `backward` as argument
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)

print(x.grad)

# Turn of tracking history `.requires_grad=True` with this:
print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)

# Or by using .detach() to get a new Tensor with the same content but that does not require gradients:
print(x.requires_grad)
y = x.detach()
print(y.requires_grad)
print(x.eq(y).all())


# NEURAL NETWORKS
#   nn depends on autograd to define models and differentiate them
#   An nn.Module contains layers, and a method forward(input)that returns the output
"""
# A typical training procedure for a neural network is as follows:
#
#     Define the neural network that has some LEARNABLE PARAMETERS (or weights)
#     Iterate over a dataset of inputs
#     Process input through the network
#     Compute the loss (how far is the output from being correct)
#     Propagate gradients back into the network’s parameters
#     Update the weights of the network, typically using a simple update rule: weight= weight - learning_rate * gradient
"""


from Net_Tutorial import Net

net = Net()
print(net)

# The learnable parameters of a model
params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight

# try a 32x32 input
input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out, '\n')

# Zero the gradient buffers of all parameters and backprops with random gradients
net.zero_grad()
out.backward(torch.randn(1, 10))

"""
# Recap:
#     torch.Tensor
#           - A multi-dimensional array with support for autograd
#               operations like backward(). Also holds the gradient w.r.t. the tensor.
#     nn.Module
#           - Neural network module. Convenient way of encapsulating parameters,
#               with helpers for moving them to GPU, exporting, loading, etc.
#     nn.Parameter
#           - A kind of Tensor, that is automatically registered as a
#               parameter when assigned as an attribute to a Module.
#     autograd.Function
#           - Implements forward and backward definitions of an autograd operation.
#               Every Tensor operation creates at least a single Function node that
#               connects to functions that created a Tensor and encodes its history.
"""

# Loss Function
#   A loss function takes the (output, target) pair of inputs,
#   and computes a value that  estimates how far away the output is from the target.
#   There are several different loss functions under the nn package.
output = net(input)
target = torch.randn(10)  # a dummy target, for example
print(f'output:{output}\ntarget:{target}')
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()

loss = criterion(output, target)
print("Loss:", loss, '\n')


# Backprop
#   To backpropagate the error all we have to do is to loss.backward().
#       You need to clear the existing gradients though, else gradients will be accumulated to existing gradients.
#   Now we shall call loss.backward(), and have a look at conv1’s bias gradients before and after the backward.
net.zero_grad()     # zeroes the gradient buffers of all parameters

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)


# Update the weights
#    The simplest update rule used in practice is the Stochastic Gradient Descent (SGD):
#            weight = weight - learning_rate * gradient
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)

# However, as you use neural networks, you want to use various different update rules such as
#   SGD, Nesterov-SGD, Adam, RMSProp, etc. To enable this, we built a small package:
#       torch.optim that implements all these methods. Using it is very simple:
import torch.optim as optim

# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # Does the update