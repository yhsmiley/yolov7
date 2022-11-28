import argparse

import torch

from models.experimental import attempt_load
from models.yolo import Model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, help='reparameterized model filepath')
    parser.add_argument('--save_path', type=str, help='.pt path of output model')
    parser.add_argument('--cfg', type=str, default='', help='.yaml path in cfg/deploy/*.yaml')
    opt = parser.parse_args()

    model1 = attempt_load(opt.weights, map_location='cpu')
    torch.save({'state_dict': model1.state_dict(), 'class_names': model1.names}, opt.save_path)

    # test loading model for sanity
    model2 = Model(opt.cfg)
    checkpoint = torch.load(opt.save_path, map_location='cpu')
    loaded_state_dict = checkpoint['state_dict']
    model2.fuse()
    model2.load_state_dict(loaded_state_dict)
