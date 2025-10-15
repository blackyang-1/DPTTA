import argparse

#code by black-y 2025.10.15 16:02

def parse_args():
    parser = argparse.ArgumentParser(prog='main')
    parser.add_argument('-lr', '--lr', default=1e-5,type=float,required=False, help='learning rate')
    parser.add_argument('-bt1', '--beta1', type=float, default=1,required=False, help='weight for self-supervised loss')
    parser.add_argument('-bt2', '--beta2', type=float, default=1,required=False, help='weight for output loss')
    parser.add_argument('-bs', '--batch_size', type=int, default=128,required=False, help='batch size')
    parser.add_argument('-seed', '--seed', type=int, default=42,required=False, help='random seed')
    parser.add_argument('-noise_level', '--noise_level', type=float, default=120,required=False, help='noise level for augmentation')
    parser.add_argument('-img_size', '--image_size', type=int, default=30, required=False, help='image size')  
    
    opt = parser.parse_args()

    return opt
