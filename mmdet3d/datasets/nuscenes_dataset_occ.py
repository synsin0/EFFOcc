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
from pyquaternion import Quaternion
import random
random.seed(42)
np.random.seed(42)

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
class NuScenesDatasetOccpancy(NuScenesDataset):
    def __init__(self, lidar_corruption=None, img_corruption=None, load_sequences=0, use_sequences_as_labeled=0, active_original_sequences_as_labeled = 0, load_samples_from_files=None, **kwargs):
        
        self.load_samples_from_files = load_samples_from_files
        self.load_sequences = load_sequences     
        self.use_sequences_as_labeled = use_sequences_as_labeled           
        super().__init__(**kwargs)
        self.lidar_corruption = lidar_corruption
        self.img_corruption = img_corruption
        
        self.robo3d_root = "data/Robo3D/nuScenes-C"
        if self.lidar_corruption is not None:
            self.lidar_corrupt_root = os.path.join(self.robo3d_root,'pointcloud',lidar_corruption)
        if self.img_corruption is not None:
            self.img_corrupt_root = os.path.join(self.robo3d_root,'image',img_corruption)
        

        # self.data_infos = self.data_infos[:100]

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        data = mmcv.load(ann_file, file_format='pkl')
        data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
        
        if self.load_sequences > 0:
            all_scene_token = list(sorted(set([info['scene_token'] for info in data_infos])))
            random.seed(42)
            selected_token_list = sorted(random.sample(all_scene_token, self.load_sequences))
            data_infos = [info for info in data_infos if info['scene_token'] in selected_token_list]

        if self.use_sequences_as_labeled > 0: # for distillation
            all_scene_token = list(sorted(set([info['scene_token'] for info in data_infos])))
            random.seed(42)

            selected_token_list = sorted(random.sample(all_scene_token, self.use_sequences_as_labeled))
            # import ipdb; ipdb.set_trace()
            for info in data_infos:
                if info['scene_token'] in selected_token_list:
                    info['use_with_label'] = True
                else:
                    info['use_with_label'] = False
                
        if self.load_samples_from_files is not None:
            use_frames = mmcv.load(self.load_samples_from_files)
            data_infos = [info for info in data_infos if info['token'] in use_frames]
        data_infos = data_infos[::self.load_interval]
        self.metadata = data['metadata']
        self.version = self.metadata['version']
        return data_infos

    def get_sensor_transforms(self, info):
        # sweep ego to global
        w, x, y, z = info['ego2global_rotation']
        ego2global_rot = torch.Tensor(
                Quaternion(w, x, y, z).rotation_matrix)
        ego2global_rot = torch.Tensor(ego2global_rot)

        ego2global_tran = torch.Tensor(
            info['ego2global_translation'])
        ego2global = ego2global_rot.new_zeros((4, 4))
        ego2global[3, 3] = 1
        ego2global[:3, :3] = ego2global_rot
        ego2global[:3, -1] = ego2global_tran
        return ego2global

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
        input_dict = super(NuScenesDatasetOccpancy, self).get_data_info(index)
        # standard protocol modified from SECOND.Pytorch
        if not self.test_mode:
            input_dict['occ_gt_path'] = self.data_infos[index]['occ_path']
        
        if 'occ_path' in self.data_infos[index].keys():
            input_dict['occ_gt_path'] = self.data_infos[index]['occ_path']
        
        input_dict['scene_token'] = self.data_infos[index]['scene_token']
        if self.lidar_corruption is not None:
            input_dict['pts_filename'] = input_dict['pts_filename'].replace("./data/nuscenes", self.lidar_corrupt_root)
            input_dict['curr']['lidar_path'] = input_dict['pts_filename']
        if self.img_corruption is not None:
            for cam_name, cam in input_dict['curr']['cams'].items():
                cam['data_path'] = cam['data_path'].replace("./data/nuscenes", self.img_corrupt_root).replace("/samples","")


        input_dict['ego2global'] = self.get_sensor_transforms(self.data_infos[index])
        
        if 'use_with_label' in self.data_infos[index].keys():
            input_dict['use_with_label'] = self.data_infos[index]['use_with_label']
        return input_dict

    def evaluate(self, occ_results, runner=None, show_dir=None, **eval_kwargs):

        # self.format_results(occ_results,submission_prefix='./fusionocc_test_results')
        # self.format_results(occ_results, submission_prefix=None)
        
        
        self.occ_eval_metrics = Metric_mIoU(
            num_classes=18,
            use_lidar_mask=False,
            use_image_mask=True)
        
    
        # if show_dir:
        #     save_results = []

        print('\nStarting Evaluation...')
        for index, occ_pred in enumerate(tqdm(occ_results)):
            info = self.data_infos[index]

            occ_gt = np.load(os.path.join(info['occ_path'],'labels.npz'))
            gt_semantics = occ_gt['semantics']
            mask_lidar = occ_gt['mask_lidar'].astype(bool)
            mask_camera = occ_gt['mask_camera'].astype(bool)
            # occ_pred = occ_pred
            self.occ_eval_metrics.add_batch(occ_pred, gt_semantics, mask_lidar, mask_camera)

            # if show_dir:
            #     save_result = {}
            #     save_result['voxel_semantics'] = gt_semantics
            #     save_result['voxel_semantics_preds'] = occ_pred
            #     save_result['mask_lidar'] = mask_lidar
            #     save_result['mask_camera'] = mask_camera
            #     save_result['mask_infov'] = None
            #     save_results.append(save_result)


            # if index%100==0 and show_dir is not None:
            #     gt_vis = self.vis_occ(gt_semantics)
            #     pred_vis = self.vis_occ(occ_pred)
            #     mmcv.imwrite(np.concatenate([gt_vis, pred_vis], axis=1),
            #                  os.path.join(show_dir + "%d.jpg"%index))
            
            
            # show_dir = 'occpred_output/flashocc_100%_with_pretrain/results/'
            if show_dir is not None:
                mmcv.mkdir_or_exist(show_dir)
                scene_name = info['scene_token']
                sample_token = info['token']
                mmcv.mkdir_or_exist(os.path.join(show_dir, scene_name, sample_token))
                save_path = os.path.join(show_dir, scene_name, sample_token, 'pred.npz')
                np.savez_compressed(save_path, pred=occ_pred, gt=occ_gt, sample_token=sample_token)


        # if show_dir:
        #     mmcv.dump(save_results, os.path.join(show_dir , "occ3d_nusc.pkl"))
        return self.occ_eval_metrics.count_miou()

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

    def format_results(self, occ_results,submission_prefix=None,**kwargs):
        if submission_prefix is not None:
            mmcv.mkdir_or_exist(submission_prefix)

        for index, occ_pred in enumerate(tqdm(occ_results)):
            info = self.data_infos[index]
            sample_token = info['token']
            save_path=os.path.join(submission_prefix,'{}.npz'.format(sample_token))
            np.savez_compressed(save_path,occ_pred.astype(np.uint8))
        print('\nFinished.')

