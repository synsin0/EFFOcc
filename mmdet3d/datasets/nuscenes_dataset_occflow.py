# Copyright (c) OpenMMLab. All rights reserved.
import os
import mmcv
import torch
import cv2
import numpy as np
from tqdm import tqdm

from .builder import DATASETS
from .nuscenes_dataset import NuScenesDataset
from .occ_metrics import Metric_mIoU, Metric_FScore

from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from nuscenes.utils.geometry_utils import transform_matrix
from .ray_metrics import main as ray_based_miou
from torch.utils.data import DataLoader
from .nuscenes_ego_pose_loader import nuScenesDataset
from nuscenes.nuscenes import NuScenes


colors_map = np.array(
    [
        [0,   0,   0, 255],  # 0 undefined
        [255, 158, 0, 255],  # 1 car  orange
        [0, 0, 230, 255],    # 2 pedestrian  Blue
        [47, 79, 79, 255],   # 3 sign  Darkslategrey
        [220, 20, 60, 255],  # 4 CYCLIST  Crimson
        [255, 69, 0, 255],   # 5 traiffic_light  Orangered
        [255, 140, 0, 255],  # 6 pole  Darkorange
        [233, 150, 70, 255], # 7 construction_cone  Darksalmon
        [255, 61, 99, 255],  # 8 bycycle  Red
        [112, 128, 144, 255],# 9 motorcycle  Slategrey
        [222, 184, 135, 255],# 10 building Burlywood
        [0, 175, 0, 255],    # 11 vegetation  Green
        [165, 42, 42, 255],  # 12 trunk  nuTonomy green
        [0, 207, 191, 255],  # 13 curb, road, lane_marker, other_ground
        [75, 0, 75, 255], # 14 walkable, sidewalk
        [255, 0, 0, 255], # 15 unobsrvd
        [0, 0, 0, 0],  # 16 undefined
        [0, 0, 0, 0],  # 16 undefined
    ])



@DATASETS.register_module()
class NuScenesDatasetOccpancyFlow(NuScenesDataset):
    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        input_dict = super(NuScenesDatasetOccpancyFlow, self).get_data_info(index)
        # standard protocol modified from SECOND.Pytorch
        input_dict['occ_path'] = self.data_infos[index]['occflow_path']
        return input_dict

    def evaluate(self, occ_results, runner=None, show_dir=None, **eval_kwargs):
        occ_gts = []
        flow_gts = []
        lidar_origins = []

        print('\nStarting Evaluation...')

        sem_results = [r['occ_results'].cpu().numpy() for r in occ_results]
        flow_results = [r['flow_results'].cpu().numpy() for r in occ_results]

        data_loader_kwargs={
            "pin_memory": False,
            "shuffle": False,
            "batch_size": 1,
            "num_workers": 8,
        }

        nusc = NuScenes('v1.0-trainval', 'data/nuscenes')

        data_loader = DataLoader(
            nuScenesDataset(nusc, 'val'),
            **data_loader_kwargs,
        )
        
        for i, batch in tqdm(enumerate(data_loader), ncols=50):
            if i >= len(self.data_infos):
                break

            output_origin = batch[1]
            lidar_origins.append(output_origin)
        
            info = self.data_infos[i]
            occ_gt = np.load(info['occflow_path'], allow_pickle=True)
            gt_semantics = occ_gt['semantics']
            gt_flow = occ_gt['flow']

            occ_gts.append(gt_semantics)
            flow_gts.append(gt_flow)
        
        return ray_based_miou(sem_results, occ_gts, flow_results, flow_gts, lidar_origins)

    def format_results(self, occ_results,submission_prefix,**kwargs):
        if submission_prefix is not None:
            mmcv.mkdir_or_exist(submission_prefix)

        for index, occ_pred in enumerate(tqdm(occ_results)):
            info = self.data_infos[index]
            sample_token = info['token']
            save_path=os.path.join(submission_prefix,'{}.npz'.format(sample_token))
            np.savez_compressed(save_path,occ_pred.astype(np.uint8))

        print('\nFinished.')


    def vis_occ(self, semantics):
        # simple visualization of result in BEV
        semantics_valid = np.logical_not(semantics == 17)
        d = np.arange(16).reshape(1, 1, 16)
        d = np.repeat(d, 200, axis=0)
        d = np.repeat(d, 200, axis=1).astype(np.float32)
        d = d * semantics_valid
        selected = np.argmax(d, axis=2)

        selected_torch = torch.from_numpy(selected)
        semantics_torch = torch.from_numpy(semantics)

        occ_bev_torch = torch.gather(semantics_torch, dim=2,
                                     index=selected_torch.unsqueeze(-1))
        occ_bev = occ_bev_torch.numpy()

        occ_bev = occ_bev.flatten().astype(np.int32)
        occ_bev_vis = colors_map[occ_bev].astype(np.uint8)
        occ_bev_vis = occ_bev_vis.reshape(200, 200, 4)[::-1, ::-1, :3]
        occ_bev_vis = cv2.resize(occ_bev_vis,(400,400))
        return occ_bev_vis