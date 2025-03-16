# Copyright (c) Phigent Robotics. All rights reserved.
from .bevdet import BEVDet, BEVStereo4D
from mmdet3d.models import DETECTORS
from mmdet3d.models.builder import build_head
import torch.nn.functional as F
import torch.nn as nn
from mmdet3d.models import builder
import torch
from mmdet3d.ops.bev_pool_v2.bev_pool import TRTBEVPoolv2

from .distill_occ2d import calculate_iou
import numpy as np



@DETECTORS.register_module(force=True)
class FlashFusionOCCFlow(BEVDet):
    def __init__(self,
                 occ_head=None,
                 upsample=False,
                 occ_fuser=None,
                 use_infov_mask=False,
                 use_lidar_mask=False,
                 use_camera_mask=False,
                 
                 **kwargs):
        super(FlashFusionOCCFlow, self).__init__(**kwargs)
        self.occ_head = build_head(occ_head)
        self.pts_bbox_head = None
        self.upsample = upsample
        self.occ_fuser = builder.build_fusion_layer(occ_fuser) if occ_fuser is not None else None
        self.use_infov_mask=use_infov_mask
        self.use_lidar_mask=use_lidar_mask
        self.use_camera_mask=use_camera_mask

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
        voxel_flow = kwargs['voxel_flow']

        mask_camera = kwargs.get('mask_camera')
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

        loss_occ = self.forward_occ_train(occ_bev_feature, voxel_semantics, final_mask, voxel_flow)
        losses.update(loss_occ)
        return losses

    def forward_occ_train(self, img_feats, voxel_semantics, mask_camera, voxel_flow):
        """
        Args:
            img_feats: (B, C, Dz, Dy, Dx) / (B, C, Dy, Dx)
            voxel_semantics: (B, Dx, Dy, Dz)
            mask_camera: (B, Dx, Dy, Dz)
        Returns:
        """
        occ_pred, flow_pred = self.occ_head(img_feats)


        # occ_preds = self.occ_head.get_occ(outs, img_metas=None)      # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        # ious, mIoU = calculate_iou(np.stack(occ_preds), voxel_semantics.cpu().numpy())
        # import ipdb
        # ipdb.set_trace()

        # assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
        loss_occ = self.occ_head.loss(voxel_semantics, voxel_flow, mask_camera, occ_pred, flow_pred)
        
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

        occ_dict = self.simple_test_occ(occ_bev_feature, img_metas)    # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        return occ_dict

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
        occ_pred, flow_out = self.occ_head(img_feats)

        occ_score=occ_pred.softmax(-1)
        occ_res=occ_score.argmax(-1)
        # occ_res = occ_res.squeeze(dim=0).cpu().numpy().astype(np.uint8)
        return [{
            'occ_results': occ_res.cpu(),
            'flow_results': flow_out.cpu(),
        }]

