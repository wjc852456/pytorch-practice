import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import sys
import os
sys.path.append(os.path.expanduser('~/pytorch-quant/utee'))
sys.path.append(os.path.expanduser('~/pytorch-mobilenet-v2'))
import quant
import MobileNetV2


def bn2conv(model):
    r""" conv layer must be arranged before bn layer !!!"""
    if isinstance(model,nn.Sequential):
        ikv = enumerate(model._modules.items())
        for i,(k,v) in ikv:
            if isinstance(v,nn.Conv2d):
                key,bn = next(ikv)[1]
                if isinstance(bn, nn.BatchNorm2d):
                    if bn.affine:
                        a = bn.weight / torch.sqrt(bn.running_var+bn.eps)
                        b = - bn.weight * bn.running_mean / torch.sqrt(bn.running_var+bn.eps) + bn.bias
                    else:
                        a = 1.0 / torch.sqrt(bn.running_var+bn.eps)
                        b = - bn.running_mean / torch.sqrt(bn.running_var+bn.eps)
                    v.weight = Parameter( v.weight * a.reshape(v.out_channels,1,1,1) )
                    v.bias   = Parameter(b)
                    model._modules[key] = nn.Sequential()
            else:
                bn2conv(v)
    else:
        for k,v in model._modules.items():
            bn2conv(v)
    
#net = MobileNetV2.MobileNetV2()
#bn2conv(net)
#print(net)


ifmap = 3
ofmap = 4
conv1 = nn.Conv2d(ifmap, ofmap, 3, padding=1, bias=False)
bn1   = nn.BatchNorm2d(ofmap)
seq1  = nn.Sequential(conv1, bn1).eval()
x = torch.randn(2,3,5,5)
y = seq1(x)
print(seq1)
print('\n')
bn2conv(seq1)
y1 = seq1(x)
print(seq1)

'''
conv2 = conv1
conv2.load_state_dict(conv1.state_dict(),strict=False)
bn2   = quant.BatchNorm2d_new(ofmap)
bn2.load_state_dict(bn1.state_dict())
bn2.set_newparam()
conv2.weight = Parameter(conv2.weight*bn2.a.reshape(ofmap,1,1,1))
conv2.bias = Parameter(bn2.b)
y2 = conv2(x)
#print(y2)
#print(conv2.bias)
'''