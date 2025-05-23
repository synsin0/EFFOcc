# Copyright (c) OpenMMLab. All rights reserved.

import argparse

import torch
from mmcv import Config, DictAction

from mmdet3d.models import build_model

try:
    from mmcv.cnn import get_model_complexity_info
except ImportError:
    raise ImportError('Please upgrade mmcv to >0.6.2')


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[930000, 4],
        help='input point cloud size')
    parser.add_argument(
        '--modality',
        type=str,
        default='point',
        choices=['point', 'image', 'multi'],
        help='input data modality')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def construct_input(input_shape):
    rot = torch.eye(4).float().cuda().view(1, 1, 4, 4).expand(1,6,4,4)

    intrins = torch.eye(3).float().cuda().view(1,1, 3, 3).expand(1,6,3,3)
    input = dict(
        img_inputs=[
        torch.ones(()).new_empty((1, 6, 3, 256, 704)).cuda(), rot,
        # torch.ones(()).new_empty((1, 6, 3, 512, 1408)).cuda(), rot,
        rot, intrins, intrins,
        torch.ones((1, 6, 3)).cuda(),
        torch.eye(4).float().cuda().view(1, 4, 4)
        ],
        points = [torch.zeros([1,300000, 5])],
    )
    return input


def main():

    args = parse_args()

    if args.modality == 'point':
        assert len(args.shape) == 2, 'invalid input shape'
        input_shape = tuple(args.shape)
    elif args.modality == 'image':
        if len(args.shape) == 1:
            input_shape = (3, args.shape[0], args.shape[0])
        elif len(args.shape) == 2:
            input_shape = (3, ) + tuple(args.shape)
        else:
            raise ValueError('invalid input shape')
    elif args.modality == 'multi':
        raise NotImplementedError(
            'FLOPs counter is currently not supported for models with '
            'multi-modality input')

    cfg = Config.fromfile(args.config)
    if 'stereo' in args.config or 'longterm' in args.config:
        assert False,'Config has not supported: %s ' % args.config
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not supported for {}'.format(
                model.__class__.__name__))

    flops, params = get_model_complexity_info(
        model, input_shape,
        input_constructor=construct_input)
    split_line = '=' * 30
    print(f'{split_line}\nInput shape: {input_shape}\n'
          f'Flops: {flops}\nParams: {params}\n{split_line}')
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')

    # from thop import profile
    # flops, params = profile(model, (construct_input(input_shape),))
    # print('flops: ', flops, 'params: ', params)
    # print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f'Params: {pytorch_total_params}')

if __name__ == '__main__':
    main()
