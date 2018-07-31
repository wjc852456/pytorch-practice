from torch.autograd import Variable
import torch.onnx
import torchvision

dummy_input = torch.randn(10, 3, 224, 224).cuda()
model = torchvision.models.alexnet(pretrained=True).cuda()

# providing these is optional, but makes working with the
# converted model nicer.
input_names = [ "learned_%d" % i for i in range(16) ] + [ "actual_input_1" ]
output_names = [ "output1" ]

torch.onnx.export(model, dummy_input, "alexnet.proto", verbose=True, input_names=input_names, output_names=output_names)
