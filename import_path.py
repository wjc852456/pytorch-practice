import sys
import os
root_path = '~/pytorch-mobilenet-v2/MobileNetV2.py'
root_list = root_path.split('/')
#print(root_list)
path = '/'.join(root_list[0:-1])
#print(path)
root = os.path.expanduser(path)
#print(root)
sys.path.append(root)
model_name = "MobileNetV2"
exec('from {} import {}'.format(root_list[-1][:-3], model_name))
net = eval('{}()'.format(model_name))
print(net)