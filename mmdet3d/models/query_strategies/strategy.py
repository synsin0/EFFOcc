import os
import pickle

import torch

import os.path as osp
import pickle
import shutil
import tempfile
import time
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader

import mmcv
from mmcv.runner import get_dist_info

def collect_results_cpu(result_part: list,
                        size: int,
                        tmpdir: Optional[str] = None) -> Optional[list]:
    """Collect results under cpu mode.

    On cpu mode, this function will save the results on different gpus to
    ``tmpdir`` and collect them by the rank 0 worker.

    Args:
        result_part (list): Result list containing result parts
            to be collected.
        size (int): Size of the results, commonly equal to length of
            the results.
        tmpdir (str | None): temporal directory for collected results to
            store. If set to None, it will create a random temporal directory
            for it.

    Returns:
        list: The collected results.
    """
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    part_file = osp.join(tmpdir, f'part_{rank}.pkl')  # type: ignore
    mmcv.dump(result_part, part_file)
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')  # type: ignore
            part_result = mmcv.load(part_file)
            # When data is severely insufficient, an empty part_result
            # on a certain gpu could makes the overall outputs empty.
            if part_result:
                part_list.append(part_result)
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)  # type: ignore
        return ordered_results


def collect_results_gpu(result_part: list, size: int) -> Optional[list]:
    """Collect results under gpu mode.

    On gpu mode, this function will encode results to gpu tensors and use gpu
    communication for results collection.

    Args:
        result_part (list): Result list containing result parts
            to be collected.
        size (int): Size of the results, commonly equal to length of
            the results.

    Returns:
        list: The collected results.
    """
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_result = pickle.loads(recv[:shape[0]].cpu().numpy().tobytes())
            # When data is severely insufficient, an empty part_result
            # on a certain gpu could makes the overall outputs empty.
            if part_result:
                part_list.append(part_result)
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results
    else:
        return None



class Strategy:
    def __init__(self, model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg):
        
        self.cfg = cfg
        self.active_label_dir = active_label_dir
        self.rank = rank
        self.model = model
        self.labelled_loader = labelled_loader
        self.unlabelled_loader = unlabelled_loader
        self.labelled_set = labelled_loader.dataset
        self.unlabelled_set = unlabelled_loader.dataset
        self.bbox_records = {}
        self.point_measures = ['mean', 'median', 'variance']
        for met in self.point_measures:
            setattr(self, '{}_point_records'.format(met), {})
   
        # self.pairs = list(zip(self.unlabelled_set.frame_ids, self.unlabelled_set.infos))

    def save_points(self, frame_id, batch_dict):
        # 'num_bbox': num_bbox,
        # 'mean_points': mean_points,
        # 'median_points': median_points,
        # 'variance_points': variance_points,

        self.bbox_records[frame_id] = batch_dict['num_bbox']
        
        self.mean_point_records[frame_id] = batch_dict['mean_points']
        self.median_point_records[frame_id] = batch_dict['median_points']
        self.variance_point_records[frame_id] = batch_dict['variance_points']
           
    

    def update_dashboard(self, cur_epoch=None, accumulated_iter=None):

        classes = list(self.selected_bbox[0].keys())
    
    
        total_bbox = 0
        for cls_idx in classes:
            
            num_cls_bbox = sum([i[cls_idx] for i in self.selected_bbox])
            wandb.log({'active_selection/num_bbox_{}'.format(cls_idx): num_cls_bbox}, step=accumulated_iter)
            total_bbox += num_cls_bbox
        
            if num_cls_bbox:
                for met in self.point_measures:
                    stats_point = sum([i[cls_idx] for i in getattr(self, 'selected_{}_points'.format(met))]) / len(getattr(self, 'selected_{}_points'.format(met)))
                    wandb.log({'active_selection/{}_points_{}'.format(met, cls_idx): stats_point}, step=accumulated_iter)
            else:
                for met in self.point_measures:
                    wandb.log({'active_selection/{}_points_{}'.format(met, cls_idx): 0}, step=accumulated_iter)

        
        wandb.log({'active_selection/total_bbox_selected': total_bbox}, step=accumulated_iter)

    
    def save_active_labels(self, selected_frames=None, grad_embeddings=None, cur_epoch=None):
      
        if selected_frames is not None:
            # self.selected_bbox = [self.bbox_records[i] for i in selected_frames]
            for met in self.point_measures:
                setattr(self, 'selected_{}_points'.format(met), [getattr(self, '{}_point_records'.format(met))[i] for i in selected_frames])

            with open(os.path.join(self.active_label_dir, 'selected_frames_epoch_{}_rank_{}.pkl'.format(cur_epoch, self.rank)), 'wb') as f:
                pickle.dump({'frame_id': selected_frames, 'selected_mean_points': self.selected_mean_points, 'selected_bbox': self.selected_bbox, \
                'selected_median_points': self.selected_median_points, 'selected_variance_points': self.selected_variance_points}, f)
            print('successfully saved selected frames for epoch {} for rank {}'.format(cur_epoch, self.rank))

        if grad_embeddings is not None:
            with open(os.path.join(self.active_label_dir, 'grad_embeddings_epoch_{}.pkl'.format(cur_epoch)), 'wb') as f:
                pickle.dump(grad_embeddings, f)
            print('successfully saved grad embeddings for epoch {}'.format(cur_epoch))

    def query(self, leave_pbar=True, cur_epoch=None):
        pass