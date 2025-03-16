# Copyright (c) Phigent Robotics. All rights reserved.
from .bevdet import BEVDet, BEVStereo4D
from mmdet3d.models import DETECTORS
from mmdet3d.models.builder import build_head
import torch.nn.functional as F
import torch.nn as nn
from mmdet3d.models import builder
import torch
from mmdet3d.ops.bev_pool_v2.bev_pool import TRTBEVPoolv2
import copy
from .distill_occ2d import calculate_iou
import numpy as np
from mmcv.runner import auto_fp16
import mmcv
import os
import copy
import time
@DETECTORS.register_module(force=True)
class ActiveFlashFusionOCC(BEVDet):
    def __init__(self,
                 init_model = None,
                 occ_head=None,
                 upsample=False,
                 occ_fuser=None,
                 use_infov_mask=False,
                 use_lidar_mask=False,
                 use_camera_mask=True,
                 mode = 'occ_challenge',
                 learning_mode = 'normal',
                 active_voxel_selection = 64,
                 unify_all_category = False,
                 save_active_selection_to_disk = False,
                 **kwargs):
        super(ActiveFlashFusionOCC, self).__init__(**kwargs)
        self.occ_head = build_head(occ_head)
        self.pts_bbox_head = None
        self.upsample = upsample
        self.occ_fuser = builder.build_fusion_layer(occ_fuser) if occ_fuser is not None else None
        self.use_infov_mask=use_infov_mask
        self.use_lidar_mask=use_lidar_mask
        self.use_camera_mask=use_camera_mask
        self.mode = mode
        self.learning_mode = learning_mode
        self.active_voxel_selection = active_voxel_selection
        self.use_active_from_epoch = 0
        self.use_active_last_epoch = 0

        self.active_voxel_mask_dir = ''
        self.unify_all_category = unify_all_category

        self.save_active_selection_to_disk = save_active_selection_to_disk
        if not self.save_active_selection_to_disk:
            self.active_selection_dict = dict()
        # self.init_model = init_model

    def extract_pts_feat(self, pts):
        """Extract features of points."""

        voxels, num_points, coors = self.voxelize(pts)

        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)

        return [x]
        
    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        img = self.prepare_inputs(img)
        x, _  = self.image_encoder(img[0])
        return [x] + img[1:]

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        
        img_feats = self.extract_img_feat(img_inputs, img_metas)
        img_feats_bev = \
            self.img_view_transformer(img_feats + img_inputs[1:7],
                                      depth_from_lidar=kwargs['gt_depth'])

        pts_voxel_feats = self.extract_pts_feat(points)
        
        occ_bev_feature = self.occ_fuser(img_feats_bev[0], pts_voxel_feats[0])
        
        occ_bev_feature = self.pts_backbone(occ_bev_feature)
        if self.with_pts_neck:
            occ_bev_feature = self.pts_neck(occ_bev_feature)
        
        occ_bev_feature = occ_bev_feature[0]

        losses = dict()
        voxel_semantics = kwargs['voxel_semantics']     # (B, Dx, Dy, Dz)
        mask_camera = kwargs.get('mask_camera')     # (B, Dx, Dy, Dz)
        mask_lidar = kwargs.get('mask_lidar')
        mask_infov = kwargs.get('mask_infov')


        final_mask = torch.ones_like(voxel_semantics)
        if self.use_infov_mask:
            final_mask = torch.logical_and(mask_infov, final_mask)
        if self.use_lidar_mask:
            final_mask = torch.logical_and(mask_lidar, final_mask)
        if self.use_camera_mask:
            final_mask = torch.logical_and(mask_camera, final_mask)
        final_mask = final_mask.bool()
        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)
        
        if self.learning_mode == 'normal':
            loss_occ = self.forward_occ_train(occ_bev_feature, voxel_semantics, final_mask)
            losses.update(loss_occ)
        elif self.learning_mode == 'activate_al':
            final_full_cls_logits = self.occ_head(occ_bev_feature)
            outs = final_full_cls_logits
            num_classes = final_full_cls_logits.shape[-1]
            active_mask_multi_batch = []
            for bs in range(final_full_cls_logits.shape[0]):
                logits = final_full_cls_logits[bs]
                # logits = logits.view(-1, num_classes)
                
                occ_score = logits.softmax(-1)    # (B, Dx, Dy, Dz, C)
                occ_res = occ_score.argmax(-1)      # (B, Dx, Dy, Dz)

                voxel_values = \
                    -(F.softmax(logits, dim=-1) *
                        F.log_softmax(logits, dim=-1)).sum(dim=-1)
                
                if self.use_active_from_epoch > 0:
                    if self.save_active_selection_to_disk:
                        last_active_mask = mmcv.load(os.path.join(self.active_voxel_mask_dir, 'epoch_{}'.format(self.use_active_last_epoch),'{}.pkl'.format(img_metas[bs]['sample_idx'])))
                        last_active_mask = torch.from_numpy(last_active_mask).to(final_mask.device)
                    else:
                        last_active_mask = copy.deepcopy(self.active_selection_dict[img_metas[bs]['sample_idx']])
                        last_active_mask = torch.from_numpy(last_active_mask).to(final_mask.device)
                
                    # active_mask = torch.logical_or(active_mask,last_active_mask)         
            
                voxel_values[~final_mask[bs]] = 0
                if self.use_active_from_epoch > 0:
                    voxel_values[last_active_mask] = 0
                
                if self.unify_all_category:
                    flattened_voxel_values = voxel_values.view(-1)
                    top_values, indices = torch.topk(flattened_voxel_values, self.active_voxel_selection, largest=True)
                

                    voxel_indices = torch.stack((indices // (voxel_values.shape[1] * voxel_values.shape[2]) , (indices // voxel_values.shape[2]) % voxel_values.shape[1] ,indices % voxel_values.shape[2]),dim=1)
                   
                    active_mask = torch.zeros_like(final_mask[bs])            
                    active_mask[voxel_indices[:,0], voxel_indices[:,1], voxel_indices[:,2]] = True
                else:
                    active_mask = torch.zeros_like(final_mask[bs])    
                    self.active_voxel_selection_per_class = int(self.active_voxel_selection / len(occ_res.unique()))    
                    for i in occ_res.unique():
                        class_mask = occ_res==i
                        class_voxel_values = voxel_values.clone()
                        class_voxel_values[~class_mask] = 0
                        flattened_voxel_values = class_voxel_values.view(-1)
                        top_values, indices = torch.topk(flattened_voxel_values, self.active_voxel_selection_per_class, largest=True)
                        voxel_indices = torch.stack((indices // (class_voxel_values.shape[1] * class_voxel_values.shape[2]) , (indices // class_voxel_values.shape[2]) % class_voxel_values.shape[1] ,indices % class_voxel_values.shape[2]),dim=1)
                        active_mask[voxel_indices[:,0], voxel_indices[:,1], voxel_indices[:,2]] = True

                if self.use_active_from_epoch > 0:                 
                    active_mask = torch.logical_or(active_mask,last_active_mask)


                active_mask_multi_batch.append(active_mask)
                
                # start = time.time()
                if self.save_active_selection_to_disk:
                    mmcv.mkdir_or_exist(os.path.join(self.active_voxel_mask_dir, 'epoch_{}'.format(self.use_active_from_epoch)))
                    mmcv.dump(active_mask.cpu().numpy(), os.path.join(self.active_voxel_mask_dir, 'epoch_{}'.format(self.use_active_from_epoch),'{}.pkl'.format(img_metas[bs]['sample_idx'])))
                else:
                    self.active_selection_dict[img_metas[bs]['sample_idx']] = active_mask.cpu().numpy()
                # end = time.time()
                # print(end-start)

            # import ipdb
            # ipdb.set_trace()
            final_mask = torch.stack(active_mask_multi_batch,dim=0)
                  # assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
            loss_occ = self.occ_head.loss(
                outs,  # (B, Dx, Dy, Dz, n_cls)
                voxel_semantics,  # (B, Dx, Dy, Dz)
                final_mask,  # (B, Dx, Dy, Dz)
            )
            losses.update(loss_occ)

        elif self.learning_mode == 'deactivate_al':
            final_full_cls_logits = self.occ_head(occ_bev_feature)
            outs = final_full_cls_logits
            num_classes = final_full_cls_logits.shape[-1]


            active_mask_multi_batch = []
            for bs in range(final_full_cls_logits.shape[0]):
                if self.save_active_selection_to_disk:
                    active_mask = mmcv.load(os.path.join(self.active_voxel_mask_dir, 'epoch_{}'.format(self.use_active_from_epoch),'{}.pkl'.format(img_metas[bs]['sample_idx'])))
                    active_mask = torch.from_numpy(active_mask).to(final_mask.device)
                else:
                    active_mask = copy.deepcopy(self.active_selection_dict[img_metas[bs]['sample_idx']])
                    active_mask = torch.from_numpy(active_mask).to(final_mask.device)

                active_mask_multi_batch.append(active_mask)
            
            final_mask = torch.stack(active_mask_multi_batch,dim=0)
            loss_occ = self.occ_head.loss(
                outs,  # (B, Dx, Dy, Dz, n_cls)
                voxel_semantics,  # (B, Dx, Dy, Dz)
                final_mask,  # (B, Dx, Dy, Dz)
            )
            losses.update(loss_occ)

        return losses

    def forward_occ_train(self, img_feats, voxel_semantics, mask_camera):
        """
        Args:
            img_feats: (B, C, Dz, Dy, Dx) / (B, C, Dy, Dx)
            voxel_semantics: (B, Dx, Dy, Dz)
            mask_camera: (B, Dx, Dy, Dz)
        Returns:
        """
        outs = self.occ_head(img_feats)


        # occ_preds = self.occ_head.get_occ(outs, img_metas=None)      # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        # ious, mIoU = calculate_iou(np.stack(occ_preds), voxel_semantics.cpu().numpy())
        # import ipdb
        # ipdb.set_trace()

        # assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
        loss_occ = self.occ_head.loss(
            outs,  # (B, Dx, Dy, Dz, n_cls)
            voxel_semantics,  # (B, Dx, Dy, Dz)
            mask_camera,  # (B, Dx, Dy, Dz)
        )
        return loss_occ
    def extract_bev_feat(self,
                    points,
                    img_metas,
                    img_inputs=None,
                    rescale=False,
                    **kwargs):
        img_feats = self.extract_img_feat(img_inputs, img_metas)
        img_feats_bev = \
            self.img_view_transformer(img_feats + img_inputs[1:7],
                                      depth_from_lidar=kwargs['gt_depth'])

        pts_voxel_feats = self.extract_pts_feat(points)
        
        occ_bev_feature = self.occ_fuser(img_feats_bev[0], pts_voxel_feats[0])
        
        occ_bev_feature = self.pts_backbone(occ_bev_feature)
        if self.with_pts_neck:
            occ_bev_feature = self.pts_neck(occ_bev_feature)
        
        occ_bev_feature = occ_bev_feature[0]
        return occ_bev_feature

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)

        img_feats = self.extract_img_feat(img, img_metas)
        img_feats_bev = \
            self.img_view_transformer(img_feats + img[1:7],
                                      depth_from_lidar=kwargs['gt_depth'])

        pts_voxel_feats = self.extract_pts_feat(points)
        
        occ_bev_feature = self.occ_fuser(img_feats_bev[0], pts_voxel_feats[0])
        
        occ_bev_feature = self.pts_backbone(occ_bev_feature)
        if self.with_pts_neck:
            occ_bev_feature = self.pts_neck(occ_bev_feature)
        
        occ_bev_feature = occ_bev_feature[0]

        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)
        
        # if self.learning_mode == 'normal':
        if self.mode == 'occ_challenge':
            occ_list = self.simple_test_occ(occ_bev_feature, img_metas)    # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
            return occ_list
        elif self.mode == 'openoccupancy':
            results = self.simple_test_openoccupancy(occ_bev_feature, img_metas, **kwargs)    # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
            return results      
        # elif self.learning_mode == 'active':
        #     outs = self.occ_head(occ_bev_feature)
        #     outs = outs.softmax(-1)
        #     return outs


    def forward_dummy(self,
                     points=None,
                     img_metas=None,
                     img_inputs=None,
                     **kwargs):
        """
        Args:
            points (list[torch.Tensor]): the outer list indicates test-time
                augmentations and inner torch.Tensor should have a shape NxC,
                which contains all points in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            img (list[torch.Tensor], optional): the outer
                list indicates test-time augmentations and inner
                torch.Tensor should have a shape NxCxHxW, which contains
                all images in the batch. Defaults to None.
        """

        H, W = img_inputs[0].shape[-2:]
        kwargs['gt_depth'] = torch.zeros([1,6,H,W]).cuda()
        points = [points] if points is None else points
        points[0] = points[0].cuda()
        return self.simple_test(points[0], None, img_inputs,
                                **kwargs)      


    def simple_test_occ(self, img_feats, img_metas=None):
        """
        Args:
            img_feats: (B, C, Dz, Dy, Dx) / (B, C, Dy, Dx)
            img_metas:

        Returns:
            occ_preds: List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        """
        outs = self.occ_head(img_feats)

        occ_preds = self.occ_head.get_occ(outs, img_metas)      # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        return occ_preds

    def simple_test_openoccupancy(self, img_feats, img_metas=None, voxel_semantics=None, visible_mask=None,**kwargs):
        """
        Args:
            img_feats: (B, C, Dz, Dy, Dx) / (B, C, Dy, Dx)
            img_metas:

        Returns:
            occ_preds: List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        """
        outs = self.occ_head(img_feats)

        pred_c = self.occ_head.get_occ(outs, img_metas)      # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        gt_occ = voxel_semantics[0].cpu()
        
        pred_c = torch.Tensor(np.stack(pred_c,axis=0))

        # gt_occ = torch.stack(gt_occ,axis=0).cpu()
        # kwargs.get('voxel_semantics')
        visible_mask = visible_mask
        # kwargs.get('visible_mask')
        
        SC_metric, _ = self.evaluation_semantic(pred_c, gt_occ, eval_type='SC', visible_mask=visible_mask)
        SSC_metric, SSC_occ_metric = self.evaluation_semantic(pred_c, gt_occ, eval_type='SSC', visible_mask=visible_mask)
        pred_f = None

        test_output = {
            'SC_metric': SC_metric,
            'SSC_metric': SSC_metric,
            # 'pred_c': pred_c,
            # 'pred_f': pred_f,
        }

        return [test_output]

    @auto_fp16(apply_to=('img', 'points'))
    def forward(self, return_loss=True, active = False, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.

        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if active:
            return self.forward_occ_results(**kwargs)
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def forward_test(self,
                     points=None,
                     img_metas=None,
                     img_inputs=None,
                     **kwargs):
        """
        Args:
            points (list[torch.Tensor]): the outer list indicates test-time
                augmentations and inner torch.Tensor should have a shape NxC,
                which contains all points in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            img (list[torch.Tensor], optional): the outer
                list indicates test-time augmentations and inner
                torch.Tensor should have a shape NxCxHxW, which contains
                all images in the batch. Defaults to None.
        """
        for var, name in [(img_inputs, 'img_inputs'),
                          (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        

        num_augs = len(img_inputs)
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(img_inputs), len(img_metas)))

        if not isinstance(img_inputs[0][0], list):
            img_inputs = [img_inputs] if img_inputs is None else img_inputs
            points = [points] if points is None else points
            return self.simple_test(points[0], img_metas[0], img_inputs[0],
                                    **kwargs)
        else:
            return self.aug_test(None, img_metas[0], img_inputs[0], **kwargs)


    def forward_occ_results(self,
                    points=None,
                    img_metas=None,
                    img_inputs=None,
                    rescale=False,
                    **kwargs):
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)

        img_feats = self.extract_img_feat(img_inputs, img_metas)
        img_feats_bev = \
            self.img_view_transformer(img_feats + img_inputs[1:7],
                                      depth_from_lidar=kwargs['gt_depth'])

        pts_voxel_feats = self.extract_pts_feat(points)
        
        occ_bev_feature = self.occ_fuser(img_feats_bev[0], pts_voxel_feats[0])
        
        occ_bev_feature = self.pts_backbone(occ_bev_feature)
        if self.with_pts_neck:
            occ_bev_feature = self.pts_neck(occ_bev_feature)
        
        occ_bev_feature = occ_bev_feature[0]

        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)
        
        outs = self.occ_head(occ_bev_feature)
        
        return outs 

    def evaluation_semantic(self, pred, gt, eval_type, visible_mask=None):
        self.empty_idx = 0
        _, H, W, D = gt.shape
        # pred = F.interpolate(pred, size=[H, W, D], mode='trilinear', align_corners=False).contiguous()

        # pred = torch.argmax(pred[0], dim=0).numpy()
        pred = pred[0].numpy()
        gt = gt[0].numpy()
        pred = pred.astype(np.int)
        gt = gt.astype(np.int)

        # ignore noise
        noise_mask = gt != 255

        if eval_type == 'SC':
            # 0 1 split
            gt[gt != self.empty_idx] = 1
            pred[pred != self.empty_idx] = 1
            return fast_hist(pred[noise_mask], gt[noise_mask], max_label=2), None


        if eval_type == 'SSC':
            hist_occ = None
            if visible_mask is not None:
                visible_mask = visible_mask[0].numpy()
                mask = noise_mask & (visible_mask!=0)
                hist_occ = fast_hist(pred[mask], gt[mask], max_label=17)

            hist = fast_hist(pred[noise_mask], gt[noise_mask], max_label=17)
            return hist, hist_occ

def fast_hist(pred, label, max_label=18):
    pred = copy.deepcopy(pred.flatten())
    label = copy.deepcopy(label.flatten())
    bin_count = np.bincount(max_label * label.astype(int) + pred, minlength=max_label ** 2)
    return bin_count[:max_label ** 2].reshape(max_label, max_label)
