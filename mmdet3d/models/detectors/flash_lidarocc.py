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

from mmdet3d.models.utils import FFN
from mmdet3d.models.utils.spconv_voxelize import SPConvVoxelization

@DETECTORS.register_module(force=True)
class FlashLiDAROCC(BEVDet):
    def __init__(self,
                 occ_head=None,
                 upsample=False,
                 use_infov_mask=False,
                 use_lidar_mask=False,
                 use_camera_mask=True,
                 **kwargs):
        super(FlashLiDAROCC, self).__init__(**kwargs)
        self.occ_head = build_head(occ_head)
        self.pts_bbox_head = None
        self.upsample = upsample
        self.use_infov_mask=use_infov_mask
        self.use_lidar_mask=use_lidar_mask
        self.use_camera_mask=use_camera_mask

    def extract_pts_feat(self, pts):
        """Extract features of points."""

        voxels, num_points, coors = self.voxelize(pts)

        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x

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


    
        points = [points] if points is None else points
        return self.simple_test(points[0], img_metas[0], None,
                                **kwargs)

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

        points = [points] if points is None else points
        points[0] = points[0].cuda()
        return self.simple_test(points[0], None, None,
                                **kwargs)      

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

        # img_feats, pts_feats, depth = self.extract_feat(
        #     points, img=img, img_metas=img_metas, **kwargs)
        pts_voxel_feats = self.extract_pts_feat(points)


        losses = dict()
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

        occ_bev_feature = pts_voxel_feats[0]
        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)

        loss_occ = self.forward_occ_train(occ_bev_feature, voxel_semantics, final_mask)
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


        # assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
        loss_occ = self.occ_head.loss(
            outs,  # (B, Dx, Dy, Dz, n_cls)
            voxel_semantics,  # (B, Dx, Dy, Dz)
            mask_camera,  # (B, Dx, Dy, Dz)
        )
        return loss_occ

    def simple_test(self,
                    points,
                    img_metas=None,
                    img=None,
                    rescale=False,
                    **kwargs):
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        # img_feats, _, _ = self.extract_feat(
        #     points, img=img, img_metas=img_metas, **kwargs)
        pts_voxel_feats = self.extract_pts_feat(points)

        occ_bev_feature = pts_voxel_feats[0]
        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)

        occ_list = self.simple_test_occ(occ_bev_feature, img_metas)    # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
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


@DETECTORS.register_module(force=True)
class FlashFusionOCC(BEVDet):
    def __init__(self,
                 occ_head=None,
                 upsample=False,
                 occ_fuser=None,
                 use_infov_mask=False,
                 use_lidar_mask=False,
                 use_camera_mask=True,
                 mode = 'occ_challenge',
                 **kwargs):
        super(FlashFusionOCC, self).__init__(**kwargs)
        self.occ_head = build_head(occ_head)
        self.pts_bbox_head = None
        self.upsample = upsample
        self.occ_fuser = builder.build_fusion_layer(occ_fuser) if occ_fuser is not None else None
        self.use_infov_mask=use_infov_mask
        self.use_lidar_mask=use_lidar_mask
        self.use_camera_mask=use_camera_mask
        self.mode = mode

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

    def extract_bev_feat_occ(self,
                    points,
                    img_metas,
                    img_inputs=None,
                    rescale=False,
                    **kwargs):
        img_feats = self.extract_img_feat(img_inputs, img_metas)
        img_feats_bev = \
            self.img_view_transformer(img_feats + img_inputs[1:7],
                                      depth_from_lidar=kwargs['gt_depth'],
                                      with_bevdepth_supervision=False)
        # img_feats_bev = \
        #     self.img_view_transformer(img_feats + img_inputs[1:7])

        pts_voxel_feats = self.extract_pts_feat(points)
        
        occ_bev_feature = self.occ_fuser(img_feats_bev[0], pts_voxel_feats[0])
        
        occ_bev_feature = self.pts_backbone(occ_bev_feature)
        if self.with_pts_neck:
            occ_bev_feature = self.pts_neck(occ_bev_feature)
        
        occ_bev_feature = occ_bev_feature[0]

        out_occ_logits = self.occ_head(occ_bev_feature)

        return occ_bev_feature, out_occ_logits, img_feats_bev[0], pts_voxel_feats[0]

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
        if len(img_inputs) == 1:
            points = points[0]
            img_inputs = img_inputs[0]
    

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
        
        if self.mode == 'occ_challenge':
            occ_list = self.simple_test_occ(occ_bev_feature, img_metas)    # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
            return occ_list
        elif self.mode == 'openoccupancy':
            results = self.simple_test_openoccupancy(occ_bev_feature, img_metas, **kwargs)    # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
            return results      

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
        
        if self.mode == 'occ_challenge':
            occ_list = self.simple_test_occ(occ_bev_feature, img_metas)    # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
            return occ_list
        elif self.mode == 'openoccupancy':
            results = self.simple_test_openoccupancy(occ_bev_feature, img_metas, **kwargs)    # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
            return results 



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

@DETECTORS.register_module(force=True)
class FlashFusionOCCv2(BEVDet):
    def __init__(self,
                 occ_head=None,
                 upsample=False,
                 occ_fuser=None,
                 use_infov_mask=False,
                 use_lidar_mask=False,
                 use_camera_mask=True,
                 **kwargs):
        super(FlashFusionOCCv2, self).__init__(**kwargs)
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
        
        # occ_bev_feature = self.pts_backbone(occ_bev_feature)
        # if self.with_pts_neck:
        #     occ_bev_feature = self.pts_neck(occ_bev_feature)
        
        # occ_bev_feature = occ_bev_feature[0]

        losses = dict()
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
        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)

        loss_occ = self.forward_occ_train(occ_bev_feature, voxel_semantics, final_mask)
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
        # assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
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
        img_feats = self.extract_img_feat(img, img_metas)
        img_feats_bev = \
            self.img_view_transformer(img_feats + img[1:7],
                                      depth_from_lidar=kwargs['gt_depth'])

        pts_voxel_feats = self.extract_pts_feat(points)
        
        occ_bev_feature = self.occ_fuser(img_feats_bev[0], pts_voxel_feats[0])
        
        # occ_bev_feature = self.pts_backbone(occ_bev_feature)
        # if self.with_pts_neck:
        #     occ_bev_feature = self.pts_neck(occ_bev_feature)
        
        # occ_bev_feature = occ_bev_feature[0]

        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)

        occ_list = self.simple_test_occ(occ_bev_feature, img_metas)    # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        return occ_list

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
        kwargs['gt_depth'] = torch.zeros([1,6,256,704]).cuda()
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

# @DETECTORS.register_module(force=True)
# class FlashFusionOCCv2(BEVDet):
#     def __init__(self,
#                  occ_head=None,
#                  upsample=False,
#                  occ_fuser=None,
#                  occ_encoder_backbone=None,
#                  occ_encoder_neck = None,
#                  **kwargs):
#         super(FlashFusionOCCv2, self).__init__(**kwargs)
#         self.occ_head = build_head(occ_head)
#         self.pts_bbox_head = None
#         self.upsample = upsample
#         self.occ_fuser = builder.build_fusion_layer(occ_fuser) if occ_fuser is not None else None
#         self.occ_encoder_backbone = builder.build_backbone(occ_encoder_backbone)
#         self.occ_encoder_neck = builder.build_neck(occ_encoder_neck)
#     def extract_pts_feat(self, pts):
#         """Extract features of points."""

#         voxels, num_points, coors = self.voxelize(pts)

#         voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
#         batch_size = coors[-1, 0] + 1
#         x = self.pts_middle_encoder(voxel_features, coors, batch_size)
       
#         x = self.pts_backbone(x)
#         if self.with_pts_neck:
#             x = self.pts_neck(x)
#         return x
        
#     def extract_img_feat(self, img, img_metas):
#         """Extract features of images."""
#         img = self.prepare_inputs(img)
#         x, _  = self.image_encoder(img[0])
#         return [x] + img[1:]

#     def occ_encoder(self, x):
#         x = self.occ_encoder_backbone(x)
#         x = self.occ_encoder_neck(x)
#         return x

#     def forward_train(self,
#                       points=None,
#                       img_metas=None,
#                       gt_bboxes_3d=None,
#                       gt_labels_3d=None,
#                       gt_labels=None,
#                       gt_bboxes=None,
#                       img_inputs=None,
#                       proposals=None,
#                       gt_bboxes_ignore=None,
#                       **kwargs):
#         """Forward training function.

#         Args:
#             points (list[torch.Tensor], optional): Points of each sample.
#                 Defaults to None.
#             img_metas (list[dict], optional): Meta information of each sample.
#                 Defaults to None.
#             gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
#                 Ground truth 3D boxes. Defaults to None.
#             gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
#                 of 3D boxes. Defaults to None.
#             gt_labels (list[torch.Tensor], optional): Ground truth labels
#                 of 2D boxes in images. Defaults to None.
#             gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
#                 images. Defaults to None.
#             img (torch.Tensor optional): Images of each sample with shape
#                 (N, C, H, W). Defaults to None.
#             proposals ([list[torch.Tensor], optional): Predicted proposals
#                 used for training Fast RCNN. Defaults to None.
#             gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
#                 2D boxes in images to be ignored. Defaults to None.

#         Returns:
#             dict: Losses of different branches.
#         """
#         # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
#         # pts_feats: None
#         # depth: (B*N_views, D, fH, fW)
        

#         img_feats = self.extract_img_feat(img_inputs, img_metas)
#         img_feats_bev, depth = \
#             self.img_view_transformer(img_feats + img_inputs[1:7],
#                                       depth_from_lidar=kwargs['gt_depth'])

                                     
#         img_feats_bev = self.bev_encoder(img_feats_bev) # torch.Size([2, 256, 200, 200])
#         img_feats_bev = [img_feats_bev]

#         pts_voxel_feats = self.extract_pts_feat(points)
 
#         occ_bev_feature = self.occ_fuser(img_feats_bev[0], pts_voxel_feats[0])
        
#         occ_bev_feature = self.occ_encoder(occ_bev_feature)
         
#         # occ_bev_feature = occ_bev_feature[0]

#         losses = dict()
#         voxel_semantics = kwargs['voxel_semantics']     # (B, Dx, Dy, Dz)
#         mask_camera = kwargs['mask_camera']     # (B, Dx, Dy, Dz)

#         if self.upsample:
#             occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
#                                             mode='bilinear', align_corners=True)

#         loss_occ = self.forward_occ_train(occ_bev_feature, voxel_semantics, mask_camera)
#         losses.update(loss_occ)
#         return losses

#     def forward_occ_train(self, img_feats, voxel_semantics, mask_camera):
#         """
#         Args:
#             img_feats: (B, C, Dz, Dy, Dx) / (B, C, Dy, Dx)
#             voxel_semantics: (B, Dx, Dy, Dz)
#             mask_camera: (B, Dx, Dy, Dz)
#         Returns:
#         """
#         outs = self.occ_head(img_feats)
#         # assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
#         loss_occ = self.occ_head.loss(
#             outs,  # (B, Dx, Dy, Dz, n_cls)
#             voxel_semantics,  # (B, Dx, Dy, Dz)
#             mask_camera,  # (B, Dx, Dy, Dz)
#         )
#         return loss_occ

#     def simple_test(self,
#                     points,
#                     img_metas,
#                     img=None,
#                     rescale=False,
#                     **kwargs):
#         # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
#         # pts_feats: None
#         # depth: (B*N_views, D, fH, fW)
#         img_feats = self.extract_img_feat(img, img_metas)
#         img_feats_bev, depth = \
#             self.img_view_transformer(img_feats + img[1:7],
#                                       depth_from_lidar=kwargs['gt_depth'])

                                     
#         img_feats_bev = self.bev_encoder(img_feats_bev) # torch.Size([2, 256, 200, 200])
#         img_feats_bev = [img_feats_bev]

#         pts_voxel_feats = self.extract_pts_feat(points)
 
#         occ_bev_feature = self.occ_fuser(img_feats_bev[0], pts_voxel_feats[0])
        
#         occ_bev_feature = self.occ_encoder(occ_bev_feature)
         

#         if self.upsample:
#             occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
#                                             mode='bilinear', align_corners=True)

#         occ_list = self.simple_test_occ(occ_bev_feature, img_metas)    # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
#         return occ_list

#     def simple_test_occ(self, img_feats, img_metas=None):
#         """
#         Args:
#             img_feats: (B, C, Dz, Dy, Dx) / (B, C, Dy, Dx)
#             img_metas:

#         Returns:
#             occ_preds: List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
#         """
#         outs = self.occ_head(img_feats)
#         occ_preds = self.occ_head.get_occ(outs, img_metas)      # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
#         return occ_preds

# @DETECTORS.register_module(force=True)
# class FlashBEVStereo4DOCC(BEVStereo4D):
#     def __init__(self,
#                  occ_head=None,
#                  **kwargs):
#         super(FlashBEVStereo4DOCC, self).__init__(**kwargs)
#         self.occ_head = build_head(occ_head)
#         self.pts_bbox_head = None

#     def forward_train(self,
#                       points=None,
#                       img_metas=None,
#                       gt_bboxes_3d=None,
#                       gt_labels_3d=None,
#                       gt_labels=None,
#                       gt_bboxes=None,
#                       img_inputs=None,
#                       proposals=None,
#                       gt_bboxes_ignore=None,
#                       **kwargs):
#         """Forward training function.

#         Args:
#             points (list[torch.Tensor], optional): Points of each sample.
#                 Defaults to None.
#             img_metas (list[dict], optional): Meta information of each sample.
#                 Defaults to None.
#             gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
#                 Ground truth 3D boxes. Defaults to None.
#             gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
#                 of 3D boxes. Defaults to None.
#             gt_labels (list[torch.Tensor], optional): Ground truth labels
#                 of 2D boxes in images. Defaults to None.
#             gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
#                 images. Defaults to None.
#             img (torch.Tensor optional): Images of each sample with shape
#                 (N, C, H, W). Defaults to None.
#             proposals ([list[torch.Tensor], optional): Predicted proposals
#                 used for training Fast RCNN. Defaults to None.
#             gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
#                 2D boxes in images to be ignored. Defaults to None.

#         Returns:
#             dict: Losses of different branches.
#         """
#         # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
#         # pts_feats: None
#         # depth: (B*N_views, D, fH, fW)
#         img_feats, pts_feats, depth = self.extract_feat(
#             points, img=img, img_metas=img_metas, **kwargs)

#         gt_depth = kwargs['gt_depth']   # (B, N_views, img_H, img_W)
#         losses = dict()
#         loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
#         losses['loss_depth'] = loss_depth

#         voxel_semantics = kwargs['voxel_semantics']     # (B, Dx, Dy, Dz)
#         mask_camera = kwargs['mask_camera']     # (B, Dx, Dy, Dz)
#         loss_occ = self.forward_occ_train(img_feats[0], voxel_semantics, mask_camera)
#         losses.update(loss_occ)
#         return losses

#     def forward_occ_train(self, img_feats, voxel_semantics, mask_camera):
#         """
#         Args:
#             img_feats: (B, C, Dz, Dy, Dx) / (B, C, Dy, Dx)
#             voxel_semantics: (B, Dx, Dy, Dz)
#             mask_camera: (B, Dx, Dy, Dz)
#         Returns:
#         """
#         outs = self.occ_head(img_feats)
#         assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
#         loss_occ = self.occ_head.loss(
#             outs,  # (B, Dx, Dy, Dz, n_cls)
#             voxel_semantics,  # (B, Dx, Dy, Dz)
#             mask_camera,  # (B, Dx, Dy, Dz)
#         )
#         return loss_occ

#     def simple_test(self,
#                     points,
#                     img_metas,
#                     img=None,
#                     rescale=False,
#                     **kwargs):
#         # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
#         # pts_feats: None
#         # depth: (B*N_views, D, fH, fW)
#         img_feats, _, _ = self.extract_feat(
#             points, img=img, img_metas=img_metas, **kwargs)

#         occ_list = self.simple_test_occ(img_feats[0], img_metas)    # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
#         return occ_list

#     def simple_test_occ(self, img_feats, img_metas=None):
#         """
#         Args:
#             img_feats: (B, C, Dz, Dy, Dx) / (B, C, Dy, Dx)
#             img_metas:

#         Returns:
#             occ_preds: List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
#         """
#         outs = self.occ_head(img_feats)
#         occ_preds = self.occ_head.get_occ(outs, img_metas)      # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
#         return occ_preds



@DETECTORS.register_module(force=True)
class FlashFusionOCCv3(BEVDet):
    def __init__(self,
                 occ_head=None,
                 upsample=False,
                 occ_fuser=None,
                 use_infov_mask=False,
                 use_lidar_mask=False,
                 use_camera_mask=True,
                 **kwargs):
        super(FlashFusionOCCv3, self).__init__(**kwargs)
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

        pts_voxel_feats = self.extract_pts_feat(points) # torch.Size([2, 256, 200, 200])
        img_feats_bev = img_feats_bev[0]
        pts_voxel_feats = pts_voxel_feats[0]
        occ_bev_feature = self.occ_fuser(img_feats_bev, pts_voxel_feats) # torch.Size([2, 256, 200, 200])
        
        # occ_bev_feature = self.pts_backbone(occ_bev_feature)
        # if self.with_pts_neck:
        #     occ_bev_feature = self.pts_neck(occ_bev_feature)
        
        # occ_bev_feature = occ_bev_feature[0]  # torch.Size([2, 512, 200, 200])

        losses = dict()
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
        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)
            pts_voxel_feats = F.interpolate(pts_voxel_feats, scale_factor=2,
                                            mode='bilinear', align_corners=True)
            img_feats_bev = F.interpolate(img_feats_bev, scale_factor=2,
                                            mode='bilinear', align_corners=True)

        loss_occ = self.forward_occ_train([occ_bev_feature, pts_voxel_feats, img_feats_bev], voxel_semantics, final_mask)
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
        # assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
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
        img_feats = self.extract_img_feat(img, img_metas)
        img_feats_bev = \
            self.img_view_transformer(img_feats + img[1:7],
                                      depth_from_lidar=kwargs['gt_depth'])

        pts_voxel_feats = self.extract_pts_feat(points)
        
        img_feats_bev = img_feats_bev[0]
        pts_voxel_feats = pts_voxel_feats[0]
        occ_bev_feature = self.occ_fuser(img_feats_bev, pts_voxel_feats) # torch.Size([2, 256, 200, 200])
        

        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)
            pts_voxel_feats = F.interpolate(pts_voxel_feats, scale_factor=2,
                                            mode='bilinear', align_corners=True)
            img_feats_bev = F.interpolate(img_feats_bev, scale_factor=2,
                                            mode='bilinear', align_corners=True)

        occ_list = self.simple_test_occ([occ_bev_feature, pts_voxel_feats, img_feats_bev], img_metas)    # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        return occ_list

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
        kwargs['gt_depth'] = torch.zeros([1,6,256,704]).cuda()
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



@DETECTORS.register_module()
class FlashFusionOCCTRT(FlashFusionOCC):

    def result_serialize(self, outs_det3d=None, outs_occ=None):
        outs_ = []
        if outs_det3d is not None:
            for out in outs_det3d:
                for key in ['reg', 'height', 'dim', 'rot', 'vel', 'heatmap']:
                    outs_.append(out[0][key])
        if outs_occ is not None:
            outs_.append(outs_occ)
        return outs_


    def result_deserialize(self, outs):
        outs_ = []
        keys = ['reg', 'height', 'dim', 'rot', 'vel', 'heatmap']
        for head_id in range(len(outs) // 6):
            outs_head = [dict()]
            for kid, key in enumerate(keys):
                outs_head[0][key] = outs[head_id * 6 + kid]
            outs_.append(outs_head)
        return outs_

    def forward(
        self,
        img, 
        points,
        gt_depth,
        ranks_depth,
        ranks_feat,
        ranks_bev,
        interval_starts,
        interval_lengths,
    ):
        depth_from_lidar = gt_depth

        B = 1
        N, C, H, W = img.shape
        x = self.img_backbone(img)
        x = self.img_neck(x)

        if isinstance(depth_from_lidar, list):
            assert len(depth_from_lidar) == 1
            depth_from_lidar = depth_from_lidar[0]
        h_img, w_img = depth_from_lidar.shape[2:]
        depth_from_lidar = depth_from_lidar.view(B * N, 1, h_img, w_img)
        depth_from_lidar = self.img_view_transformer.lidar_input_net(depth_from_lidar)
        
        x = torch.cat([x, depth_from_lidar], dim=1)
        x = self.img_view_transformer.depth_net(x)
        depth = x[:, :self.img_view_transformer.D].softmax(dim=1)
        tran_feat = x[:, self.img_view_transformer.D:(
            self.img_view_transformer.D +
            self.img_view_transformer.out_channels)]
        tran_feat = tran_feat.permute(0, 2, 3, 1)
        x = TRTBEVPoolv2.apply(depth.contiguous(), tran_feat.contiguous(),
                               ranks_depth, ranks_feat, ranks_bev,
                               interval_starts, interval_lengths, 200, 200)
        x = x.permute(0, 3, 1, 2).contiguous()

        pts_voxel_feats = self.extract_pts_feat([points])
        pts_voxel_feats = pts_voxel_feats[0]


        occ_bev_feature = self.occ_fuser(x, pts_voxel_feats) # torch.Size([2, 256, 200, 200])
        occ_bev_feature = self.pts_backbone(occ_bev_feature)
        if self.with_pts_neck:
            occ_bev_feature = self.pts_neck(occ_bev_feature)
        
        occ_bev_feature = occ_bev_feature[0]

        # bev_feat = self.bev_encoder(x)
        # outs = self.pts_bbox_head([bev_feat])
        occ_pred = self.occ_head(occ_bev_feature)

        # occ_list = self.simple_test_occ(occ_bev_feature, None)    # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        # outs = self.result_serialize(occ_list)
        outs = self.result_serialize(outs_occ = occ_pred)
        return outs


    def simple_test_occ(self, img_feats, img_metas=None):
        """
        Args:
            img_feats: (B, C, Dz, Dy, Dx) / (B, C, Dy, Dx)
            img_metas:

        Returns:
            occ_preds: List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        """
        occ_pred = self.occ_head(img_feats)

        occ_score = occ_pred.softmax(-1)    # (B, Dx, Dy, Dz, C)
        occ_res = occ_score.argmax(-1)      # (B, Dx, Dy, Dz)
        # occ_res = occ_res.to(torch.uint8)     # (B, Dx, Dy, Dz)

        return occ_res

    def get_bev_pool_input(self, input):
        input = self.prepare_inputs(input)
        coor = self.img_view_transformer.get_lidar_coor(*input[1:7])
        return self.img_view_transformer.voxel_pooling_prepare_v2(coor)




@DETECTORS.register_module(force=True)
class FlashFusionDetOCC(BEVDet):
    def __init__(self,
                 occ_head=None,
                 upsample=False,
                 occ_fuser=None,
                 use_infov_mask=False,
                 use_lidar_mask=False,
                 use_camera_mask=True,
                 mode = 'occ_challenge',
                 **kwargs):
        super(FlashFusionDetOCC, self).__init__(**kwargs)
        self.occ_head = build_head(occ_head)
        self.upsample = upsample
        self.occ_fuser = builder.build_fusion_layer(occ_fuser) if occ_fuser is not None else None
        self.use_infov_mask=use_infov_mask
        self.use_lidar_mask=use_lidar_mask
        self.use_camera_mask=use_camera_mask
        self.mode = mode

        # image view auxiliary task heads
        self.num_cls = self.pts_bbox_head.num_classes
        heads = dict(heatmap=(self.num_cls, 2))
        input_feat_dim = kwargs['pts_bbox_head']['hidden_channel']
        self.auxiliary_heads = FFN(
                input_feat_dim,
                heads,
                conv_cfg=dict(type="Conv1d"),
                norm_cfg=dict(type="BN1d"),
                bias=True)
        self.auxiliary_heads.init_weights()

        pts_voxel_cfg = kwargs.get('pts_voxel_layer', None)
        if pts_voxel_cfg:
            pts_voxel_cfg['num_point_features'] = 5
            self.pts_voxel_layer = SPConvVoxelization(**pts_voxel_cfg)



    def extract_pts_feat(self, pts):
        """Extract features of points."""

        voxels, num_points, coors = self.voxelize(pts)

        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x
        
    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        img = self.prepare_inputs(img)
        x, _  = self.image_encoder(img[0])
        return [x] + img[1:]

    def forward_img_auxiliary_train(self,
                          x,
                          img_metas,
                          gt_bboxes,
                          gt_labels,
                          gt_bboxes_ignore=None,
                          proposals=None,
                          **kwargs):
        max_instance = 150
        num_pos = 0
        centers_augego = x[0].new_zeros((len(gt_bboxes), max_instance, 3))
        box_targets_all = x[0].new_zeros((len(gt_bboxes), max_instance, 10))
        valid_mask = x[0].new_zeros((len(gt_bboxes), max_instance, 1))
        label = x[0].new_zeros((len(gt_bboxes), max_instance, 1)).to(torch.long)
        for sid in range(len(gt_bboxes)):
            centers_augego_tmp = gt_bboxes[sid].gravity_center.to(x[0])
            box_targets_tmp = self.pts_bbox_head.bbox_coder.encode(gt_bboxes[sid].tensor)
            if gt_bboxes_ignore is not None:
                centers_augego_tmp = centers_augego_tmp[gt_bboxes_ignore[sid], :]
                box_targets_tmp = box_targets_tmp[gt_bboxes_ignore[sid], :]
            num_valid_samples = centers_augego_tmp.shape[0]
            num_pos += num_valid_samples
            valid_mask[sid, :num_valid_samples, :] = 1.0
            centers_augego[sid,:num_valid_samples,:] = centers_augego_tmp
            box_targets_all[sid,:num_valid_samples,:] = box_targets_tmp
            label_tmp = gt_labels[sid].unsqueeze(-1)
            if gt_bboxes_ignore is not None:
                label_tmp = label_tmp[gt_bboxes_ignore[sid], :]
            label[sid,:num_valid_samples,:] = label_tmp
        img_feats = self.pts_bbox_head.extract_img_feat_from_3dpoints(
            centers_augego, x, fuse=False)

        heatmap = self.auxiliary_heads.heatmap(img_feats)
        loss_cls_img = self.pts_bbox_head.loss_cls(
            heatmap.permute(0, 2, 1).reshape(-1, self.num_cls),
            label.flatten(),
            valid_mask.flatten(),
            avg_factor=max(num_pos, 1))
        return dict(loss_cls_img=loss_cls_img)


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

        pts_feats = self.extract_pts_feat(points)
        
        occ_bev_feature = self.occ_fuser(img_feats_bev[0], pts_feats[0])
        

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

        loss_occ = self.forward_occ_train(occ_bev_feature, voxel_semantics, final_mask)
        losses.update(loss_occ)


        losses_pts = \
            self.forward_pts_train([img_feats, pts_feats, img_feats_bev],
                                   gt_bboxes_3d, gt_labels_3d, img_metas,
                                   gt_bboxes_ignore)
        losses.update(losses_pts)
        losses_img_auxiliary = \
            self.forward_img_auxiliary_train(img_feats,img_metas,
                                             gt_bboxes_3d, gt_labels_3d,
                                             gt_bboxes_ignore,
                                             **kwargs)
        losses.update(losses_img_auxiliary)

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

        pts_feats = self.extract_pts_feat(points)
        
        occ_bev_feature = self.occ_fuser(img_feats_bev[0], pts_feats[0])
        
        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)
        
        if self.mode == 'occ_challenge':
            occ_list = self.simple_test_occ(occ_bev_feature, img_metas)    # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
            return occ_list
        elif self.mode == 'openoccupancy':
            results = self.simple_test_openoccupancy(occ_bev_feature, img_metas, **kwargs)    # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
            return results      


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
