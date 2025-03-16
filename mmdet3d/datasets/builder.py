# Copyright (c) OpenMMLab. All rights reserved.
import platform

from mmcv.utils import Registry, build_from_cfg

from mmdet.datasets import DATASETS as MMDET_DATASETS
from mmdet.datasets.builder import _concat_dataset

import copy
import platform
import random
from functools import partial

import numpy as np
from mmcv.parallel import collate
from mmcv.runner import get_dist_info
from mmcv.utils import Registry, build_from_cfg
from torch.utils.data import DataLoader
import random
random.seed(42)
np.random.seed(42)

# from torch.utils.data import Subset
import copy
from torch.utils.data import DataLoader, Dataset, Sampler
if platform.system() != 'Windows':
    # https://github.com/pytorch/pytorch/issues/973
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    base_soft_limit = rlimit[0]
    hard_limit = rlimit[1]
    soft_limit = min(max(4096, base_soft_limit), hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

OBJECTSAMPLERS = Registry('Object sampler')
DATASETS = Registry('dataset')
PIPELINES = Registry('pipeline')


def build_dataset(cfg, default_args=None):
    from mmdet3d.datasets.dataset_wrappers import CBGSDataset
    from mmdet.datasets.dataset_wrappers import (ClassBalancedDataset,
                                                 ConcatDataset, RepeatDataset)
    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
    elif cfg['type'] == 'ConcatDataset':
        dataset = ConcatDataset(
            [build_dataset(c, default_args) for c in cfg['datasets']],
            cfg.get('separate_eval', True))
    elif cfg['type'] == 'RepeatDataset':
        dataset = RepeatDataset(
            build_dataset(cfg['dataset'], default_args), cfg['times'])
    elif cfg['type'] == 'ClassBalancedDataset':
        dataset = ClassBalancedDataset(
            build_dataset(cfg['dataset'], default_args), cfg['oversample_thr'])
    elif cfg['type'] == 'CBGSDataset':
        dataset = CBGSDataset(build_dataset(cfg['dataset'], default_args))
    elif isinstance(cfg.get('ann_file'), (list, tuple)):
        dataset = _concat_dataset(cfg, default_args)
    elif cfg['type'] in DATASETS._module_dict.keys():
        dataset = build_from_cfg(cfg, DATASETS, default_args)
    else:
        dataset = build_from_cfg(cfg, MMDET_DATASETS, default_args)
    return dataset


def build_dataset_label_unlabelled(cfg, default_args=None):
    from mmdet3d.datasets.dataset_wrappers import CBGSDataset
    from mmdet.datasets.dataset_wrappers import (ClassBalancedDataset,
                                                 ConcatDataset, RepeatDataset)
    

    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
    elif cfg['type'] == 'ConcatDataset':
        dataset = ConcatDataset(
            [build_dataset(c, default_args) for c in cfg['datasets']],
            cfg.get('separate_eval', True))
    elif cfg['type'] == 'RepeatDataset':
        dataset = RepeatDataset(
            build_dataset(cfg['dataset'], default_args), cfg['times'])
    elif cfg['type'] == 'ClassBalancedDataset':
        dataset = ClassBalancedDataset(
            build_dataset(cfg['dataset'], default_args), cfg['oversample_thr'])
    elif cfg['type'] == 'CBGSDataset':
        dataset = CBGSDataset(build_dataset(cfg['dataset'], default_args))
    elif isinstance(cfg.get('ann_file'), (list, tuple)):
        dataset = _concat_dataset(cfg, default_args)
    elif cfg['type'] in DATASETS._module_dict.keys():
        labelled_dataset = build_from_cfg(cfg, DATASETS, default_args)
        unlabelled_dataset = build_from_cfg(cfg, DATASETS, default_args)

    else:
        dataset = build_from_cfg(cfg, MMDET_DATASETS, default_args)

    # all_samples = len(dataset)
    dataset = labelled_dataset
    labeled_indices = list(range(0, len(dataset), cfg.labeled_interval))
    unlabeled_indices = [x for x in range(len(dataset)) if x not in labeled_indices]


    infos = dataset.data_infos
    labeled_infos = [x for idx, x in enumerate(infos) if idx in labeled_indices]
    unlabeled_infos = [x for idx, x in enumerate(infos) if idx in unlabeled_indices]


    # labelled_dataset = copy.deepcopy(dataset)
    # unlabelled_dataset = copy.deepcopy(dataset)
    labelled_dataset.data_infos = labeled_infos
    unlabelled_dataset.data_infos = unlabeled_infos

    # labelled_dataset = Subset(dataset, labeled_indices)
    # unlabelled_dataset = Subset(dataset, unlabeled_indices)

    labelled_dataset.CLASSES = dataset.CLASSES
    unlabelled_dataset.CLASSES = dataset.CLASSES
    # labelled_dataset.PALETTE = dataset.PALETTE
    # unlabelled_dataset.PALETTE = dataset.PALETTE
    labelled_dataset._set_group_flag()
    unlabelled_dataset._set_group_flag()

  
    return [labelled_dataset, unlabelled_dataset]



from mmdet.datasets.samplers import GroupSampler
from .samplers.group_sampler import DistributedGroupSampler
from .samplers.distributed_sampler import DistributedSampler
from .samplers.sampler import build_sampler

def build_dataloader(dataset,
                     samples_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     shuffle=True,
                     seed=None,
                     runner_type=None,
                     shuffler_sampler=None,
                     nonshuffler_sampler=None,
                     **kwargs):
    """Build PyTorch DataLoader.
    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.
    Args:
        dataset (Dataset): A PyTorch dataset.
        samples_per_gpu (int): Number of training samples on each GPU, i.e.,
            batch size of each GPU.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU.
        num_gpus (int): Number of GPUs. Only used in non-distributed training.
        dist (bool): Distributed training/test or not. Default: True.
        shuffle (bool): Whether to shuffle the data at every epoch.
            Default: True.
        kwargs: any keyword argument to be used to initialize DataLoader
    Returns:
        DataLoader: A PyTorch dataloader.
    """
    rank, world_size = get_dist_info()
    if dist:
        # DistributedGroupSampler will definitely shuffle the data to satisfy
        # that images on each GPU are in the same group
        if shuffle:
            sampler = build_sampler(shuffler_sampler if shuffler_sampler is not None else dict(type='DistributedGroupSampler'),
                                     dict(
                                         dataset=dataset,
                                         samples_per_gpu=samples_per_gpu,
                                         num_replicas=world_size,
                                         rank=rank,
                                         seed=seed)
                                     )

        else:
            sampler = build_sampler(nonshuffler_sampler if nonshuffler_sampler is not None else dict(type='DistributedSampler'),
                                     dict(
                                         dataset=dataset,
                                         num_replicas=world_size,
                                         rank=rank,
                                         shuffle=shuffle,
                                         seed=seed)
                                     )

        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        # assert False, 'not support in bevformer'
        print('WARNING!!!!, Only can be used for obtain inference speed!!!!')
        sampler = GroupSampler(dataset, samples_per_gpu) if shuffle else None
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
        pin_memory=False,
        worker_init_fn=init_fn,
        **kwargs)

    return data_loader


def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)




def build_dataset_label_unlabelled_sequences(cfg, default_args=None):
    from mmdet3d.datasets.dataset_wrappers import CBGSDataset
    from mmdet.datasets.dataset_wrappers import (ClassBalancedDataset,
                                                 ConcatDataset, RepeatDataset)
    

    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
    elif cfg['type'] == 'ConcatDataset':
        dataset = ConcatDataset(
            [build_dataset(c, default_args) for c in cfg['datasets']],
            cfg.get('separate_eval', True))
    elif cfg['type'] == 'RepeatDataset':
        dataset = RepeatDataset(
            build_dataset(cfg['dataset'], default_args), cfg['times'])
    elif cfg['type'] == 'ClassBalancedDataset':
        dataset = ClassBalancedDataset(
            build_dataset(cfg['dataset'], default_args), cfg['oversample_thr'])
    elif cfg['type'] == 'CBGSDataset':
        dataset = CBGSDataset(build_dataset(cfg['dataset'], default_args))
    elif isinstance(cfg.get('ann_file'), (list, tuple)):
        dataset = _concat_dataset(cfg, default_args)
    elif cfg['type'] in DATASETS._module_dict.keys():
        labelled_dataset = build_from_cfg(cfg, DATASETS, default_args)
        unlabelled_dataset = build_from_cfg(cfg, DATASETS, default_args)

    else:
        dataset = build_from_cfg(cfg, MMDET_DATASETS, default_args)

    # all_samples = len(dataset)
    dataset = labelled_dataset
    infos = dataset.data_infos
    infos = list(sorted(infos, key=lambda e: e['timestamp']))

    all_scene_token = list(sorted(set([info['scene_token'] for info in infos])))
    
    random.seed(42)
    selected_token_list = sorted(random.sample(all_scene_token, cfg.active_original_sequences_as_labeled))


    labeled_infos = []
    unlabeled_infos = []
    for info in infos:
        if info['scene_token'] in selected_token_list:
            labeled_infos.append(info)
        else:
            unlabeled_infos.append(info)


    # unlabeled_infos = unlabeled_infos[:10]  # for debug case
    # labelled_dataset = copy.deepcopy(dataset)
    # unlabelled_dataset = copy.deepcopy(dataset)
    labelled_dataset.data_infos = labeled_infos
    unlabelled_dataset.data_infos = unlabeled_infos

    # labelled_dataset = Subset(dataset, labeled_indices)
    # unlabelled_dataset = Subset(dataset, unlabeled_indices)

    labelled_dataset.CLASSES = dataset.CLASSES
    unlabelled_dataset.CLASSES = dataset.CLASSES
    # labelled_dataset.PALETTE = dataset.PALETTE
    # unlabelled_dataset.PALETTE = dataset.PALETTE
    labelled_dataset._set_group_flag()
    unlabelled_dataset._set_group_flag()

  
    return [labelled_dataset, unlabelled_dataset]