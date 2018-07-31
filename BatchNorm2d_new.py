import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

def replace_bn(model):
    if isinstance(model, nn.Sequential):
        for k,v in model._modules.items():
            if(isinstance(v, nn.BatchNorm2d)):
                C = v.num_features
                bn_new = nn.Conv2d(C, C, 1, groups=C)
                if v.affine:
                    a = v.weight / torch.sqrt(v.running_var+v.eps)
                    b = - v.weight * v.running_mean / torch.sqrt(v.running_var+v.eps) + v.bias
                else:
                    a = 1.0 / torch.sqrt(v.running_var+v.eps)
                    b = - v.running_mean / torch.sqrt(v.running_var+v.eps)
                bn_new.weight = Parameter(a.reshape(C,1,1,1))
                bn_new.bias   = Parameter(b)
                model._modules[k] = bn_new
            else:
                replace_bn(v)
    else:
        for k,v in model._modules.items():
            replace_bn(v)

''' 
# optimized by depth-wise conv
class BatchNorm2d_new(nn.BatchNorm2d):
    def set_newparam(self):
        if self.affine:
            self.a = Parameter(self.weight / torch.sqrt(self.running_var+self.eps))
            self.b = Parameter(- self.weight * self.running_mean / torch.sqrt(self.running_var+self.eps) \
                        + self.bias)
        else:
            self.a = Parameter(1.0 / torch.sqrt(self.running_var+self.eps))
            self.b = Parameter(- self.running_mean / torch.sqrt(self.running_var+self.eps) )
    def forward(self, input):
        N,C,H,W = input.size()
        for c in range(C):
            input[:,c,:,:] = input[:,c,:,:]*self.a[c]+self.b[c]
        return input
'''
#x = torch.randn(4, 3, 2, 2)
x = torch.range(0,4*3*2*2-1).reshape(4,3,2,2)

m = nn.BatchNorm2d(3)  # bn设置的参数实际上是channel的参数
m = m.eval()
#m.weight = Parameter(torch.ones(3)+1)
#m.bias   = Parameter(torch.ones(3))
#m.running_mean = torch.zeros(3)+10
#m.running_var  = torch.ones(3)*4
y = m(x)
print("BatchNorm:\n{}".format(y))
#print(m.state_dict())

'''
m_1 = BatchNorm2d_new(m.num_features)
m_1.load_state_dict(m.state_dict(),strict=False)
m_1.set_newparam()
m_1 = m_1.eval()
y1 = m_1(x)
#print("BatchNorm_1:\n{}".format(y1))
#state_dict = m_1.state_dict()
#for k,v in state_dict.items():
#    print(k)
#print( m_1.b )
'''

m_2 = nn.Sequential(nn.BatchNorm2d(3))
m_2 = m_2.eval()
for k,v in m_2._modules.items():
    if isinstance(v,nn.BatchNorm2d):
        v.load_state_dict(m.state_dict())
replace_bn(m_2)
y2 = m_2(x)
print("BatchNorm_2:\n{}".format(y2))
#print(m_2.state_dict())