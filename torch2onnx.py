import os
import torch
import torch.nn as nn
import numpy as np
import timm
import onnx
import argparse
from collections import OrderedDict
import onnxruntime as ort


def multigpu_load(net, path):
        state_dict = torch.load(
            path, map_location=lambda storage, loc: storage)
        l = list(state_dict.keys())
        
        l = list(map(lambda x: x[13:], l))
        state_dict = OrderedDict((l[i], v) for i,(k,v) in enumerate(state_dict.items()))

        own_state = net.state_dict()
        for name, param in state_dict.items():
            if name in own_state:         
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    # head and tail modules can be different
                    if name.find("head") == -1 and name.find("tail") == -1:
                        raise RuntimeError(
                            "While copying the parameter named {}, "
                            "whose dimensions in the model are {} and "
                            "whose dimensions in the checkpoint are {}."
                            .format(name, own_state[name].size(), param.size())
                        )
            else:
                raise RuntimeError(
                    "Missing key {} in model's state_dict".format(name)
                )


def main():
    '''
    model_list = [
        'tf_efficientnetv2_s' : img_size = 384,
        'tf_efficientnetv2_m' : img_size = 480,
        'tf_efficientnetv2_l' : img_size = 480
    ]
    '''
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model", type=str, default='tf_efficientnetv2_s')
    argparser.add_argument("--multigpu", type=bool, default=True)
    argparser.add_argument("--model_path", type=str, default='/pt/torch/14.pt')
    argparser.add_argument("--save_path", type=str, default='./pt/onnx/14.onnx')
    argparser.add_argument("--num_classes", type=int, default=37)
    argparser.add_argument("--img_size", type=int, default=384)
    args = argparser.parse_args()


    net = timm.create_model(args.model, pretrained=False, num_classes=args.num_classes)
    if args.multigpu:
        multigpu_load(net, args.model_path)
    else:
        net.load_state_dict(torch.load(args.model_path))
    dummy_input = torch.randn(1, 3, args.img_size, args.img_size)
    torch.onnx.export(
        model=net,
        args=dummy_input,
        f=args.save_path, # where should it be saved
        input_names=['input'],
        output_names=['output'],
        # opset_version=12,
        )

    # convert test
    dummy = dummy_input.numpy().astype(np.float32)
    net.eval()
    ort_session = ort.InferenceSession(args.save_path)
    onnx_output = ort_session.run(None, {"input": dummy})
    torch_output = net(dummy_input)
    print("After convert, diff: ", np.abs(onnx_output[0] - torch_output.detach().numpy()).mean())


if __name__ == '__main__':
    main()