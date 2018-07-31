import torch
import torch.nn as nn
import torch.jit as jit

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(3,10,kernel_size=3)
    def forward(self,x):
        x = self.conv1(x)
        return x

net = SimpleNet()
var = torch.rand(1,3,224,224)
trace, out = jit.get_trace_graph(net,var)
print("trace:\n{}".format(trace))
print(out.size())