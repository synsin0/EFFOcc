# Copyright (c) OpenMMLab. All rights reserved.
import random
import warnings
import os
import numpy as np
import torch
import mmcv
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (HOOKS, DistSamplerSeedHook, EpochBasedRunner,
                         Fp16OptimizerHook, OptimizerHook, build_optimizer,
                         build_runner, get_dist_info)
from mmcv.utils import build_from_cfg
from torch import distributed as dist

from mmdet3d.datasets import build_dataset
from mmdet3d.utils import find_latest_checkpoint
from mmdet.core import DistEvalHook as MMDET_DistEvalHook
from mmdet.core import EvalHook as MMDET_EvalHook
from mmdet.datasets import build_dataloader as build_mmdet_dataloader
from mmdet.datasets import replace_ImageToTensor
from mmdet.utils import get_root_logger as get_mmdet_root_logger
from mmseg.core import DistEvalHook as MMSEG_DistEvalHook
from mmseg.core import EvalHook as MMSEG_EvalHook
from mmseg.datasets import build_dataloader as build_mmseg_dataloader
from mmseg.utils import get_root_logger as get_mmseg_root_logger
from .active_training_utils import select_active_labels
from mmcv.runner import load_checkpoint
import copy
import time
try:
    # If mmdet version > 2.20.0, setup_multi_processes would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import setup_multi_processes
except ImportError:
    from mmdet3d.utils import setup_multi_processes

def init_random_seed(seed=None, device='cuda'):
    """Initialize random seed.

    If the seed is not set, the seed will be automatically randomized,
    and then broadcast to all processes to prevent some potential bugs.
    Args:
        seed (int, optional): The seed. Default to None.
        device (str, optional): The device where the seed will be put on.
            Default to 'cuda'.
    Returns:
        int: Seed to be used.
    """
    if seed is not None:
        return seed

    # Make sure all ranks share the same random seed to prevent
    # some potential bugs. Please refer to
    # https://github.com/open-mmlab/mmdetection/issues/6339
    rank, world_size = get_dist_info()
    seed = np.random.randint(2**31)
    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# def active_train_segmentor(model,
#                     dataset,
#                     cfg,
#                     distributed=False,
#                     validate=False,
#                     timestamp=None,
#                     meta=None):
#     """Launch segmentor active_training."""
#     logger = get_mmseg_root_logger(cfg.log_level)

#     # prepare data loaders
#     dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
#     data_loaders = [
#         build_mmseg_dataloader(
#             ds,
#             cfg.data.samples_per_gpu,
#             cfg.data.workers_per_gpu,
#             # cfg.gpus will be ignored if distributed
#             len(cfg.gpu_ids),
#             dist=distributed,
#             seed=cfg.seed,
#             drop_last=True) for ds in dataset
#     ]

#     # put model on gpus
#     if distributed:
#         find_unused_parameters = cfg.get('find_unused_parameters', False)
#         # Sets the `find_unused_parameters` parameter in
#         # torch.nn.parallel.DistributedDataParallel
#         model = MMDistributedDataParallel(
#             model.cuda(),
#             device_ids=[torch.cuda.current_device()],
#             broadcast_buffers=False,
#             find_unused_parameters=find_unused_parameters)
#     else:
#         model = MMDataParallel(
#             model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

#     # build runner
#     optimizer = build_optimizer(model, cfg.optimizer)

#     if cfg.get('runner') is None:
#         cfg.runner = {'type': 'IterBasedRunner', 'max_iters': cfg.total_iters}
#         warnings.warn(
#             'config is now expected to have a `runner` section, '
#             'please set `runner` in your config.', UserWarning)

#     runner = build_runner(
#         cfg.runner,
#         default_args=dict(
#             model=model,
#             batch_processor=None,
#             optimizer=optimizer,
#             work_dir=cfg.work_dir,
#             logger=logger,
#             meta=meta))

#     # register hooks
#     runner.register_training_hooks(cfg.lr_config, cfg.optimizer_config,
#                                    cfg.checkpoint_config, cfg.log_config,
#                                    cfg.get('momentum_config', None))

#     # an ugly walkaround to make the .log and .log.json filenames the same
#     runner.timestamp = timestamp

#     # register eval hooks
#     if validate:
#         val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
#         val_dataloader = build_mmseg_dataloader(
#             val_dataset,
#             samples_per_gpu=1,
#             workers_per_gpu=cfg.data.workers_per_gpu,
#             dist=distributed,
#             shuffle=False)
#         eval_cfg = cfg.get('evaluation', {})
#         eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
#         eval_hook = MMSEG_DistEvalHook if distributed else MMSEG_EvalHook
#         # In this PR (https://github.com/open-mmlab/mmcv/pull/1193), the
#         # priority of IterTimerHook has been modified from 'NORMAL' to 'LOW'.
#         runner.register_hook(
#             eval_hook(val_dataloader, **eval_cfg), priority='LOW')

#     # user-defined hooks
#     if cfg.get('custom_hooks', None):
#         custom_hooks = cfg.custom_hooks
#         assert isinstance(custom_hooks, list), \
#             f'custom_hooks expect list type, but got {type(custom_hooks)}'
#         for hook_cfg in cfg.custom_hooks:
#             assert isinstance(hook_cfg, dict), \
#                 'Each item in custom_hooks expects dict type, but got ' \
#                 f'{type(hook_cfg)}'
#             hook_cfg = hook_cfg.copy()
#             priority = hook_cfg.pop('priority', 'NORMAL')
#             hook = build_from_cfg(hook_cfg, HOOKS)
#             runner.register_hook(hook, priority=priority)

#     if cfg.resume_from:
#         runner.resume(cfg.resume_from)
#     elif cfg.load_from:
#         runner.load_checkpoint(cfg.load_from)
#     runner.run(data_loaders, cfg.workflow)


def active_train_detector(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   timestamp=None,
                   meta=None):
    logger = get_mmdet_root_logger(log_level=cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    if 'imgs_per_gpu' in cfg.data:
        logger.warning('"imgs_per_gpu" is deprecated in MMDet V2.0. '
                       'Please use "samples_per_gpu" instead')
        if 'samples_per_gpu' in cfg.data:
            logger.warning(
                f'Got "imgs_per_gpu"={cfg.data.imgs_per_gpu} and '
                f'"samples_per_gpu"={cfg.data.samples_per_gpu}, "imgs_per_gpu"'
                f'={cfg.data.imgs_per_gpu} is used in this experiments')
        else:
            logger.warning(
                'Automatically set "samples_per_gpu"="imgs_per_gpu"='
                f'{cfg.data.imgs_per_gpu} in this experiments')
        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu

    runner_type = 'EpochBasedRunner' if 'runner' not in cfg else cfg.runner[
        'type']
    data_loaders = [
        build_mmdet_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # `num_gpus` will be ignored if distributed
            num_gpus=len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            runner_type=runner_type,
            persistent_workers=cfg.data.get('persistent_workers', False))
        for ds in dataset
    ]

    labelled_dataloader, unlabelled_dataloader = data_loaders[0], data_loaders[1]

    load_checkpoint(model, cfg.model.init_model, map_location='cpu')
    
    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)


    # register eval hooks
    if validate:
        # Support batch_size > 1 in validation
        val_samples_per_gpu = cfg.data.val.pop('samples_per_gpu', 1)
        if val_samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(
                cfg.data.val.pipeline)
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_mmdet_dataloader(
            val_dataset,
            samples_per_gpu=val_samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)
    
    for active_cycle in range(cfg.active_train.active_selection):
        logger.info(
            'Starting the  {}th active learning cycle in this experiments'.format(active_cycle + 1))
        
        rank, world_size = get_dist_info()

        mmcv.mkdir_or_exist(os.path.join(cfg.work_dir, 'active_labels'))

        select_active_labels(
                    model,
                    labelled_dataloader,
                    unlabelled_dataloader,
                    get_dist_info()[0],
                    logger,
                    method = cfg.active_train.method,
                    leave_pbar=True,
                    cur_epoch = active_cycle,
                    dist_train=True,
                    active_label_dir=os.path.join(cfg.work_dir, 'active_labels'),
                    accumulated_iter=None,
                    cfg = cfg,
                )
        
        time.sleep(2)
        selected_frames = mmcv.load(os.path.join(os.path.join(cfg.work_dir, 'active_labels'), 'selected_frames_epoch_{}.pkl'.format(active_cycle)))  

        labelled_dataset = labelled_dataloader.dataset
        unlabelled_dataset = unlabelled_dataloader.dataset

        labeled_frames = [info['token'] for info in labelled_dataset.data_infos]
        unlabelled_frames = [info['token'] for info in unlabelled_dataset.data_infos]

        all_infos = labelled_dataset.data_infos + unlabelled_dataset.data_infos
        
        labelled_infos = [info for info in all_infos if info['token'] in selected_frames or info['token'] in labeled_frames]
        unlabelled_infos = [info for info in all_infos if info['token'] not in selected_frames and info['token'] in unlabelled_frames]

        labelled_dataset.data_infos = labelled_infos
        unlabelled_dataset.data_infos = unlabelled_infos
        
        assert len(labelled_dataset) == len(labeled_frames) + len(selected_frames)
        assert len(unlabelled_dataset) == len(unlabelled_frames) - len(selected_frames)

        labelled_dataset._set_group_flag()
        unlabelled_dataset._set_group_flag()

        data_loaders = [
            build_mmdet_dataloader(
                ds,
                cfg.data.samples_per_gpu,
                cfg.data.workers_per_gpu,
                # `num_gpus` will be ignored if distributed
                num_gpus=len(cfg.gpu_ids),
                dist=distributed,
                seed=cfg.seed,
                runner_type=runner_type,
                persistent_workers=cfg.data.get('persistent_workers', False))
            for ds in [labelled_dataset, unlabelled_dataset]
        ]
        
        print('reloaded dataloaders according to the latest data selection')
        labelled_dataloader, unlabelled_dataloader = data_loaders[0], data_loaders[1]

        # trainig process from labelled datasets    
        if 'runner' not in cfg:
            cfg.runner = {
                'type': 'EpochBasedRunner',
                'max_epochs': cfg.total_epochs
            }
            warnings.warn(
                'config is now expected to have a `runner` section, '
                'please set `runner` in your config.', UserWarning)
        else:
            if 'total_epochs' in cfg:
                assert cfg.total_epochs == cfg.runner.max_epochs

        runner = build_runner(
            cfg.runner,
            default_args=dict(
                model=model,
                optimizer=optimizer,
                work_dir=cfg.work_dir,
                logger=logger,
                meta=meta))

        # an ugly workaround to make .log and .log.json filenames the same
        runner.timestamp = timestamp

        # fp16 setting
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            optimizer_config = Fp16OptimizerHook(
                **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
        elif distributed and 'type' not in cfg.optimizer_config:
            optimizer_config = OptimizerHook(**cfg.optimizer_config)
        else:
            optimizer_config = cfg.optimizer_config
        
        # register hooks
        runner.register_training_hooks(
            copy.deepcopy(cfg.lr_config),  # {'policy': 'step', 'warmup': 'linear', 'warmup_iters': 200, 'warmup_ratio': 0.001, 'step': [24]}
            copy.deepcopy(optimizer_config),
            copy.deepcopy(cfg.checkpoint_config),
            copy.deepcopy(cfg.log_config),
            cfg.get('momentum_config', None),
            custom_hooks_config=cfg.get('custom_hooks', None))

        if distributed:
            if isinstance(runner, EpochBasedRunner):
                runner.register_hook(DistSamplerSeedHook())


        if validate:
            eval_cfg = cfg.get('evaluation', {})
            eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
            eval_hook = MMDET_DistEvalHook if distributed else MMDET_EvalHook
            # In this PR (https://github.com/open-mmlab/mmcv/pull/1193), the
            # priority of IterTimerHook has been modified from 'NORMAL' to 'LOW'.
            runner.register_hook(
                eval_hook(val_dataloader, **eval_cfg), priority='LOW')

        resume_from = None
        
        # reload init models
        # runner.load_checkpoint(cfg.model.init_model)
        # if cfg.resume_from is None and cfg.get('auto_resume'):
        #     resume_from = find_latest_checkpoint(cfg.work_dir)

        # if resume_from is not None:
        #     cfg.resume_from = resume_from

        # if cfg.resume_from:
        #     runner.resume(cfg.resume_from)
        # elif cfg.load_from:
        #     runner.load_checkpoint(cfg.load_from)
  
        runner.work_dir = os.path.join(runner.work_dir, 'active_cycle_{}'.format(active_cycle))
        mmcv.mkdir_or_exist(runner.work_dir)

        cfg.active_train.active_voxel_mask_dir = os.path.join(runner.work_dir, 'active_voxel_mask')

        runner.run([labelled_dataloader], cfg.workflow, cfg = cfg)


def active_train_model(model,
                dataset,
                cfg,
                distributed=False,
                validate=False,
                timestamp=None,
                meta=None):
    """A function wrapper for launching model training according to cfg.

    Because we need different eval_hook in runner. Should be deprecated in the
    future.
    """
    if cfg.model.type in ['EncoderDecoder3D']:
        active_train_segmentor(
            model,
            dataset,
            cfg,
            distributed=distributed,
            validate=validate,
            timestamp=timestamp,
            meta=meta)
    else:
        active_train_detector(
            model,
            dataset,
            cfg,
            distributed=distributed,
            validate=validate,
            timestamp=timestamp,
            meta=meta)
