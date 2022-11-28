import argparse
import yaml
from copy import deepcopy
from pathlib import Path

import torch

from models.yolo import Model
from utils.torch_utils import select_device, is_parallel


def get_idx(model_arch):
    if model_arch == 'yolov7-tiny':
        return 77, -1
    elif model_arch == 'yolov7':
        return 105, -1
    elif model_arch == 'yolov7x':
        return 121, -1
    elif model_arch == 'yolov7-w6':
        return 118, 122
    elif model_arch == 'yolov7-e6':
        return 140, 144
    elif model_arch == 'yolov7-d6':
        return 162, 166
    elif model_arch == 'yolov7-e6e':
        return 261, 265


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_arch', type=str, default='yolov7', choices=['yolov7-tiny', 'yolov7', 'yolov7x', 'yolov7-w6', 'yolov7-e6', 'yolov7-d6', 'yolov7-e6e'], help='which model architecture')
    parser.add_argument('--training_ckpt', type=str, help='.pt path of model trained by cfg/training/*.yaml')
    parser.add_argument('--output_ckpt', type=str, help='.pt path of reparameterized model')
    parser.add_argument('--deploy_cfg', type=str, default='', help='.yaml path in cfg/deploy/*.yaml')
    parser.add_argument('--nc', type=int, default=80, help='number of classes')
    opt = parser.parse_args()

    device = select_device('0', batch_size=1)
    ckpt = torch.load(opt.training_ckpt, map_location=device)
    model = Model(opt.deploy_cfg, ch=3, nc=opt.nc).to(device)

    with open(opt.deploy_cfg) as f:
        yml = yaml.load(f, Loader=yaml.SafeLoader)
    anchors = len(yml['anchors'][0]) // 2

    # copy intersect weights
    state_dict = ckpt['model'].float().state_dict()
    exclude = []
    intersect_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict() and not any(x in k for x in exclude) and v.shape == model.state_dict()[k].shape}
    model.load_state_dict(intersect_state_dict, strict=False)
    model.names = ckpt['model'].names
    model.nc = ckpt['model'].nc

    # reparametrized YOLOV7, 255=(80 + 5) * 3,  80 is coco number class.
    range_num = (model.nc + 5) * anchors

    idx, idx2 = get_idx(opt.model_arch)

    p6_archs = ['yolov7-w6', 'yolov7-e6', 'yolov7-d6', 'yolov7-e6e']
    if opt.model_arch not in p6_archs:
        for i in range(range_num):
            model.state_dict()[f'model.{idx}.m.0.weight'].data[i, :, :, :] *= state_dict[f'model.{idx}.im.0.implicit'].data[:, i, : :].squeeze()
            model.state_dict()[f'model.{idx}.m.1.weight'].data[i, :, :, :] *= state_dict[f'model.{idx}.im.1.implicit'].data[:, i, : :].squeeze()
            model.state_dict()[f'model.{idx}.m.2.weight'].data[i, :, :, :] *= state_dict[f'model.{idx}.im.2.implicit'].data[:, i, : :].squeeze()
        model.state_dict()[f'model.{idx}.m.0.bias'].data += state_dict[f'model.{idx}.m.0.weight'].mul(state_dict[f'model.{idx}.ia.0.implicit']).sum(1).squeeze()
        model.state_dict()[f'model.{idx}.m.1.bias'].data += state_dict[f'model.{idx}.m.1.weight'].mul(state_dict[f'model.{idx}.ia.1.implicit']).sum(1).squeeze()
        model.state_dict()[f'model.{idx}.m.2.bias'].data += state_dict[f'model.{idx}.m.2.weight'].mul(state_dict[f'model.{idx}.ia.2.implicit']).sum(1).squeeze()
        model.state_dict()[f'model.{idx}.m.0.bias'].data *= state_dict[f'model.{idx}.im.0.implicit'].data.squeeze()
        model.state_dict()[f'model.{idx}.m.1.bias'].data *= state_dict[f'model.{idx}.im.1.implicit'].data.squeeze()
        model.state_dict()[f'model.{idx}.m.2.bias'].data *= state_dict[f'model.{idx}.im.2.implicit'].data.squeeze()
    else:
        # copy weights of lead head
        model.state_dict()[f'model.{idx}.m.0.weight'].data -= model.state_dict()[f'model.{idx}.m.0.weight'].data
        model.state_dict()[f'model.{idx}.m.1.weight'].data -= model.state_dict()[f'model.{idx}.m.1.weight'].data
        model.state_dict()[f'model.{idx}.m.2.weight'].data -= model.state_dict()[f'model.{idx}.m.2.weight'].data
        model.state_dict()[f'model.{idx}.m.3.weight'].data -= model.state_dict()[f'model.{idx}.m.3.weight'].data
        model.state_dict()[f'model.{idx}.m.0.weight'].data += state_dict[f'model.{idx2}.m.0.weight'].data
        model.state_dict()[f'model.{idx}.m.1.weight'].data += state_dict[f'model.{idx2}.m.1.weight'].data
        model.state_dict()[f'model.{idx}.m.2.weight'].data += state_dict[f'model.{idx2}.m.2.weight'].data
        model.state_dict()[f'model.{idx}.m.3.weight'].data += state_dict[f'model.{idx2}.m.3.weight'].data
        model.state_dict()[f'model.{idx}.m.0.bias'].data -= model.state_dict()[f'model.{idx}.m.0.bias'].data
        model.state_dict()[f'model.{idx}.m.1.bias'].data -= model.state_dict()[f'model.{idx}.m.1.bias'].data
        model.state_dict()[f'model.{idx}.m.2.bias'].data -= model.state_dict()[f'model.{idx}.m.2.bias'].data
        model.state_dict()[f'model.{idx}.m.3.bias'].data -= model.state_dict()[f'model.{idx}.m.3.bias'].data
        model.state_dict()[f'model.{idx}.m.0.bias'].data += state_dict[f'model.{idx2}.m.0.bias'].data
        model.state_dict()[f'model.{idx}.m.1.bias'].data += state_dict[f'model.{idx2}.m.1.bias'].data
        model.state_dict()[f'model.{idx}.m.2.bias'].data += state_dict[f'model.{idx2}.m.2.bias'].data
        model.state_dict()[f'model.{idx}.m.3.bias'].data += state_dict[f'model.{idx2}.m.3.bias'].data

        for i in range(range_num):
            model.state_dict()[f'model.{idx}.m.0.weight'].data[i, :, :, :] *= state_dict[f'model.{idx2}.im.0.implicit'].data[:, i, : :].squeeze()
            model.state_dict()[f'model.{idx}.m.1.weight'].data[i, :, :, :] *= state_dict[f'model.{idx2}.im.1.implicit'].data[:, i, : :].squeeze()
            model.state_dict()[f'model.{idx}.m.2.weight'].data[i, :, :, :] *= state_dict[f'model.{idx2}.im.2.implicit'].data[:, i, : :].squeeze()
            model.state_dict()[f'model.{idx}.m.3.weight'].data[i, :, :, :] *= state_dict[f'model.{idx2}.im.3.implicit'].data[:, i, : :].squeeze()
        model.state_dict()[f'model.{idx}.m.0.bias'].data += state_dict[f'model.{idx2}.m.0.weight'].mul(state_dict[f'model.{idx2}.ia.0.implicit']).sum(1).squeeze()
        model.state_dict()[f'model.{idx}.m.1.bias'].data += state_dict[f'model.{idx2}.m.1.weight'].mul(state_dict[f'model.{idx2}.ia.1.implicit']).sum(1).squeeze()
        model.state_dict()[f'model.{idx}.m.2.bias'].data += state_dict[f'model.{idx2}.m.2.weight'].mul(state_dict[f'model.{idx2}.ia.2.implicit']).sum(1).squeeze()
        model.state_dict()[f'model.{idx}.m.3.bias'].data += state_dict[f'model.{idx2}.m.3.weight'].mul(state_dict[f'model.{idx2}.ia.3.implicit']).sum(1).squeeze()
        model.state_dict()[f'model.{idx}.m.0.bias'].data *= state_dict[f'model.{idx2}.im.0.implicit'].data.squeeze()
        model.state_dict()[f'model.{idx}.m.1.bias'].data *= state_dict[f'model.{idx2}.im.1.implicit'].data.squeeze()
        model.state_dict()[f'model.{idx}.m.2.bias'].data *= state_dict[f'model.{idx2}.im.2.implicit'].data.squeeze()
        model.state_dict()[f'model.{idx}.m.3.bias'].data *= state_dict[f'model.{idx2}.im.3.implicit'].data.squeeze()

    # model to be saved
    ckpt = {'model': deepcopy(model.module if is_parallel(model) else model).half(),
            'optimizer': None,
            'training_results': None,
            'epoch': -1}

    output_folder = Path(opt.output_ckpt).parent
    output_folder.mkdir(parents=True, exist_ok=True)

    # save reparameterized model
    torch.save(ckpt, opt.output_ckpt)
