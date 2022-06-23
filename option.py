import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--multigpu", type=bool, default=True)

    # models
    parser.add_argument("--pretrained", type=bool, default=True)
    parser.add_argument("--model_name", type=str, default='tf_efficientnetv2_s_in21k')

    # dataset
    parser.add_argument("--data_dir", type=str, default="/workspace/")
    parser.add_argument("--data_name", type=str, default='fashion-dataset')

    # training setups
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--weight_decay", type=float, default=1.0e-05) 
    parser.add_argument("--n_epoch", type=int, default=15)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--eval_epoch", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=1)

    # misc
    parser.add_argument("--ckpt_root", type=str, default="/workspace/FT_model")

    # tflite
    parser.add_argument("--tflite_path", type=str, default='./tflite/a.tfilte')
    return parser.parse_args()


def make_template(opt):

    # model
    if "efficientnetv2_s" in opt.model_name:
        opt.img_size = 384
    elif "efficientnetv2_m" in opt.model_name:
        opt.img_size = 480
    elif "efficientnetv2_l" in opt.model_name:
        opt.img_size = 480   
    elif "_xl" in opt.model_name:
        opt.img_size = 512
    elif "_b0" in opt.model_name:
        opt.img_size = 224   
    elif "_b1" in opt.model_name:
        opt.img_size = 240   
    elif "_b2" in opt.model_name:
        opt.img_size = 260   
    elif "_b3" in opt.model_name:
        opt.img_size = 300   

    # dataset
    with open(os.path.join(opt.data_dir, opt.data_name, 'label.txt'), 'r') as f:
        lines = f.readlines()
    opt.num_classes = len(lines)

def get_option():
    opt = parse_args()
    make_template(opt)
    return opt