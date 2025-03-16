# Copyright (c) Phigent Robotics. All rights reserved.
from .bevdet import BEVDet, BEVStereo4D
from mmdet3d.models import DETECTORS
from mmdet3d.models.builder import build_head
import torch.nn.functional as F
import torch.nn as nn
import torch

@DETECTORS.register_module(force=True)
class FlashBEVDetOCC(BEVDet):
    def __init__(self,
                 occ_head=None,
                 upsample=False,
                 use_infov_mask=False,
                 use_lidar_mask=False,
                 use_camera_mask=True,
                 **kwargs):
        super(FlashBEVDetOCC, self).__init__(**kwargs)
        self.occ_head = build_head(occ_head)
        self.pts_bbox_head = None
        self.upsample = upsample
        self.use_infov_mask=use_infov_mask
        self.use_lidar_mask=use_lidar_mask
        self.use_camera_mask=use_camera_mask

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
        if len(img_inputs) == 1:
            points = points[0]
            img_inputs = img_inputs[0]
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        img_feats, pts_feats, depth = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)

        if not kwargs.get('return_result', False):
            losses = dict()
            voxel_semantics = kwargs['voxel_semantics']     # (B, Dx, Dy, Dz)
            mask_camera = kwargs.get('mask_camera', None)     # (B, Dx, Dy, Dz)
            mask_lidar = kwargs.get('mask_lidar', None)
            mask_infov = kwargs.get('mask_infov', None)

            final_mask = torch.ones_like(voxel_semantics)
            if self.use_infov_mask and mask_infov is not None:
                final_mask = torch.logical_and(mask_infov, final_mask)
            if self.use_lidar_mask and mask_lidar is not None:
                final_mask = torch.logical_and(mask_lidar, final_mask)
            if self.use_camera_mask and mask_camera is not None:
                final_mask = torch.logical_and(mask_camera, final_mask)
            final_mask = final_mask.bool()

        occ_bev_feature = img_feats[0]
        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)

        if kwargs.get('return_result', False):
            outs = self.forward_occ_train(occ_bev_feature, **kwargs)
            return outs
        
        else:
            loss_occ = self.forward_occ_train(occ_bev_feature, final_mask, **kwargs)
            losses.update(loss_occ)
            return losses      

    def forward_occ_train(self, img_feats, final_mask=None, **kwargs):
        """
        Args:
            img_feats: (B, C, Dz, Dy, Dx) / (B, C, Dy, Dx)
            voxel_semantics: (B, Dx, Dy, Dz)
            mask_camera: (B, Dx, Dy, Dz)
        Returns:
        """
        outs = self.occ_head(img_feats)

        if kwargs.get('return_result', False):
            return outs

        # occ_preds = self.occ_head.get_occ(outs, img_metas=None)      # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        # ious, mIoU = calculate_iou(np.stack(occ_preds), voxel_semantics.cpu().numpy())

        # assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
        voxel_semantics = kwargs['voxel_semantics']

        loss_occ = self.occ_head.loss(
            outs,  # (B, Dx, Dy, Dz, n_cls)
            voxel_semantics,  # (B, Dx, Dy, Dz)
            final_mask,  # (B, Dx, Dy, Dz)
        )
        return loss_occ

    # def forward_occ_train(self, img_feats, voxel_semantics, mask_camera):
    #     """
    #     Args:
    #         img_feats: (B, C, Dz, Dy, Dx) / (B, C, Dy, Dx)
    #         voxel_semantics: (B, Dx, Dy, Dz)
    #         mask_camera: (B, Dx, Dy, Dz)
    #     Returns:
    #     """
    #     outs = self.occ_head(img_feats)
    #     # assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
    #     loss_occ = self.occ_head.loss(
    #         outs,  # (B, Dx, Dy, Dz, n_cls)
    #         voxel_semantics,  # (B, Dx, Dy, Dz)
    #         mask_camera,  # (B, Dx, Dy, Dz)
    #     )
    #     return loss_occ

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        img_feats, _, _ = self.extract_feat(
            points, img=img, img_metas=img_metas, **kwargs)

        occ_bev_feature = img_feats[0]
        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)
        
        if 'return_probs' in kwargs.keys() and kwargs['return_probs']:
            seg_logit, preds = self.simple_test_occ(occ_bev_feature, img_metas)
            prob, pred = seg_logit.max(dim=-1)
            return seg_logit, prob.cpu().numpy(), preds
        else:
            outs, occ_list = self.simple_test_occ(occ_bev_feature, img_metas)    # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
            return occ_list

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
        return outs, occ_preds


@DETECTORS.register_module(force=True)
class FlashBEVStereo4DOCC(BEVStereo4D):
    def __init__(self,
                 occ_head=None,
                 use_infov_mask=False,
                 use_lidar_mask=False,
                 use_camera_mask=True,
                 **kwargs):
        super(FlashBEVStereo4DOCC, self).__init__(**kwargs)
        self.occ_head = build_head(occ_head)
        self.pts_bbox_head = None
        self.use_infov_mask=use_infov_mask
        self.use_lidar_mask=use_lidar_mask
        self.use_camera_mask=use_camera_mask
        self.align_after_view_transfromation = False

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
        img_feats, pts_feats, depth = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        

        gt_depth = kwargs['gt_depth']   # (B, N_views, img_H, img_W)
        losses = dict()
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        losses['loss_depth'] = loss_depth

        voxel_semantics = kwargs['voxel_semantics']     # (B, Dx, Dy, Dz)
        mask_camera = kwargs['mask_camera']     # (B, Dx, Dy, Dz)
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

        loss_occ = self.forward_occ_train(img_feats[0], voxel_semantics, final_mask)
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
        assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
        loss_occ = self.occ_head.loss(
            outs,  # (B, Dx, Dy, Dz, n_cls)
            voxel_semantics,  # (B, Dx, Dy, Dz)
            mask_camera,  # (B, Dx, Dy, Dz)
        )
        return loss_occ

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        img_feats, _, _ = self.extract_feat(
            points, img=img, img_metas=img_metas, **kwargs)

        occ_list = self.simple_test_occ(img_feats[0], img_metas)    # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        return occ_list

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