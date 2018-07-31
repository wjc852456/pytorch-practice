import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import copy

#net = nn.Conv2d(3,5,3)
#base_in = torch.randn(2,3,5,5)
net = nn.Linear(5,10)
base_in = torch.randn(2,5)

quant_module = copy.deepcopy(net)
m1 = copy.deepcopy(net)
m2 = copy.deepcopy(net)
m3 = copy.deepcopy(net)
w = net.weight

Z1 = 0.3
Z2 = -0.8
weight = net.weight-Z1
input = base_in - Z2

net.weight = Parameter(weight)
output_baseline = net(input)
print('output_baseline:\n{}'.format(output_baseline))
m1.weight = w
m1.bias = None
m2.weight = Parameter(Z1*torch.ones_like(net.weight))
m2.bias = None
m3.weight = Parameter(Z1*Z2*torch.ones_like(net.weight))
m3.bias = None

quant_module.bias = net.bias
q1q2 = quant_module(base_in)
q1Z2 = m1( Z2*torch.ones_like(base_in) )
q2Z1 = m2(base_in)
Z1Z2 = m3( torch.ones_like(base_in) )
output = q1q2 - (q1Z2 + q2Z1) + Z1Z2
print('output:\n{}'.format(output))