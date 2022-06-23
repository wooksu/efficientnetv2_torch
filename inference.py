import os
import torch
import torch.nn as nn
from model import Net
from data import generate_loader
from option import get_option

@torch.no_grad()
def main(opt):
    dev = torch.device("cuda:{}".format(opt.gpu) if torch.cuda.is_available() else "cpu")
    net = torch.load(opt.model_name).to(dev)
    if opt.multigpu: # if you want to use only some gpus, nn.DataParallel(, device_ids = [0, 1])
        net = nn.DataParallel(net).to(dev)
    
    test_loader = generate_loader('test', opt)

    num_correct, num_total = 0, 0
    net.eval()
    for inputs in test_loader:
        images = inputs[0].to(dev)
        labels = inputs[1].to(dev)

        outputs = net(images)
        _, preds = torch.max(outputs.detach(), 1)
        num_correct += (preds == labels).sum().item()
        num_total += labels.size(0)

    print("Test Acc: {:.4f}".format(num_correct / num_total * 100))

if __name__ == '__main__':
    opt = get_option()
    main(opt)