import timm
import torch.nn as nn
import torchvision

class Net(nn.Module):
    def __init__(self, opt):
        super(Net, self).__init__()
        pretrained = opt.pretrained
        model_name = opt.model_name
        num_classes = opt.num_classes
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)