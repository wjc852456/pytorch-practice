from torch import nn
import torch
 
m = nn.BatchNorm2d(3)  # bn设置的参数实际上是channel的参数
input = torch.randn(4, 3, 2, 2)
output = m(input)
state_dict = m.state_dict()
for k,v in state_dict.items():
    print("{}".format(k))
    print("{}".format(v))
# print(output)
a = (input[0, 0, :, :]+input[1, 0, :, :]+input[2, 0, :, :]+input[3, 0, :, :]).sum()/16
b = (input[0, 1, :, :]+input[1, 1, :, :]+input[2, 1, :, :]+input[3, 1, :, :]).sum()/16
c = (input[0, 2, :, :]+input[1, 2, :, :]+input[2, 2, :, :]+input[3, 2, :, :]).sum()/16
print("The actual mean value of the first channel is %f" % a.data)
print("The actual mean value of the first channel is %f" % b.data)
print("The actual mean value of the first channel is %f" % c.data)
print("The running_mean value of the BN layer is %f, %f, %f" % (m.running_mean.data[0],m.running_mean.data[1],m.running_mean.data[2]))
print(m)