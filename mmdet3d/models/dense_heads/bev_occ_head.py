# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from torch import nn
import numpy as np
from mmdet3d.models.builder import HEADS, build_loss


nusc_class_frequencies = np.array([
    944004,
    1897170,
    152386,
    2391677,
    16957802,
    724139,
    189027,
    2074468,
    413451,
    2384460,
    5916653,
    175883646,
    4275424,
    51393615,
    61411620,
    105975596,
    116424404,
    1892500630
])

nusc_class_frequencies_open_occupancy = np.array([2242961742295, 25985376, 1561108, 28862014, 196106643, 15920504,
                2158753, 26539491, 4004729, 34838681, 75173306, 2255027978, 50959399, 646022466, 869055679,
                1446141335, 1724391378])

from .openocc_utils.semkitti import geo_scal_loss, sem_scal_loss, CE_ssc_loss
from .openocc_utils.lovasz_softmax import lovasz_softmax


@HEADS.register_module(force=True)
class BEVOCCHead3D(BaseModule):
    def __init__(self,
                 in_dim=32,
                 out_dim=32,
                 use_mask=True,
                 num_classes=18,
                 use_predicter=True,
                 class_wise=False,
                 loss_occ=None
                 ):
        super(BEVOCCHead3D, self).__init__()
        self.out_dim = 32
        out_channels = out_dim if use_predicter else num_classes
        self.final_conv = ConvModule(
            in_dim,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            conv_cfg=dict(type='Conv3d')
        )
        self.use_predicter = use_predicter
        if use_predicter:
            self.predicter = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim*2),
                nn.Softplus(),
                nn.Linear(self.out_dim*2, num_classes),
            )

        self.num_classes = num_classes
        self.use_mask = use_mask
        self.loss_occ = build_loss(loss_occ)
        self.class_wise = class_wise

    def forward(self, img_feats):
        """
        Args:
            img_feats: (B, C, Dz, Dy, Dx)

        Returns:

        """
        # (B, C, Dz, Dy, Dx) --> (B, C, Dz, Dy, Dx) --> (B, Dx, Dy, Dz, C)
        occ_pred = self.final_conv(img_feats).permute(0, 4, 3, 2, 1)
        if self.use_predicter:
            # (B, Dx, Dy, Dz, C) --> (B, Dx, Dy, Dz, 2*C) --> (B, Dx, Dy, Dz, n_cls)
            occ_pred = self.predicter(occ_pred)

        return occ_pred

    def loss(self, occ_pred, voxel_semantics, mask_camera):
        """
        Args:
            occ_pred: (B, Dx, Dy, Dz, n_cls)
            voxel_semantics: (B, Dx, Dy, Dz)
            mask_camera: (B, Dx, Dy, Dz)
        Returns:

        """
        loss = dict()
        voxel_semantics = voxel_semantics.long()
        if self.use_mask:
            mask_camera = mask_camera.to(torch.int32)   # (B, Dx, Dy, Dz)
            # (B, Dx, Dy, Dz) --> (B*Dx*Dy*Dz, )
            voxel_semantics = voxel_semantics.reshape(-1)
            # (B, Dx, Dy, Dz, n_cls) --> (B*Dx*Dy*Dz, n_cls)
            preds = occ_pred.reshape(-1, self.num_classes)
            # (B, Dx, Dy, Dz) --> (B*Dx*Dy*Dz, )
            mask_camera = mask_camera.reshape(-1)
            num_total_samples = mask_camera.sum()
            loss_occ = self.loss_occ(
                preds,      # (B*Dx*Dy*Dz, n_cls)
                voxel_semantics,    # (B*Dx*Dy*Dz, )
                mask_camera,        # (B*Dx*Dy*Dz, )
                avg_factor=num_total_samples
            )
            loss['loss_occ'] = loss_occ
        else:
            voxel_semantics = voxel_semantics.reshape(-1)
            preds = occ_pred.reshape(-1, self.num_classes)
            loss_occ = self.loss_occ(preds, voxel_semantics,)
            loss['loss_occ'] = loss_occ
        return loss

    def get_occ(self, occ_pred, img_metas=None):
        """
        Args:
            occ_pred: (B, Dx, Dy, Dz, C)
            img_metas:

        Returns:
            List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        """
        occ_score = occ_pred.softmax(-1)    # (B, Dx, Dy, Dz, C)
        occ_res = occ_score.argmax(-1)      # (B, Dx, Dy, Dz)
        occ_res = occ_res.cpu().numpy().astype(np.uint8)     # (B, Dx, Dy, Dz)
        return list(occ_res)


@HEADS.register_module(force=True)
class BEVOCCHead2D(BaseModule):
    def __init__(self,
                 in_dim=256,
                 out_dim=256,
                 Dz=16,
                 use_mask=True,
                 num_classes=18,
                 use_predicter=True,
                 class_wise=False,
                 loss_occ=None,
                 ):
        super(BEVOCCHead2D, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.Dz = Dz
        out_channels = out_dim if use_predicter else num_classes * Dz
        self.final_conv = ConvModule(
            self.in_dim,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            conv_cfg=dict(type='Conv2d')
        )
        self.use_predicter = use_predicter
        if use_predicter:
            self.predicter = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim * 2),
                nn.Softplus(),
                nn.Linear(self.out_dim * 2, num_classes * Dz),
            )

        self.use_mask = use_mask
        self.num_classes = num_classes
        self.loss_occ = build_loss(loss_occ)
        self.class_wise = class_wise

    def forward(self, img_feats):
        """
        Args:
            img_feats: (B, C, Dy, Dx)

        Returns:

        """
        # (B, C, Dy, Dx) --> (B, C, Dy, Dx) --> (B, Dx, Dy, C)
        occ_pred = self.final_conv(img_feats).permute(0, 3, 2, 1)
        bs, Dx, Dy = occ_pred.shape[:3]
        if self.use_predicter:
            # (B, Dx, Dy, C) --> (B, Dx, Dy, 2*C) --> (B, Dx, Dy, Dz*n_cls)
            occ_pred = self.predicter(occ_pred)
            occ_pred = occ_pred.view(bs, Dx, Dy, self.Dz, self.num_classes)

        return occ_pred

    def loss(self, occ_pred, voxel_semantics, mask_camera):
        """
        Args:
            occ_pred: (B, Dx, Dy, Dz, n_cls)
            voxel_semantics: (B, Dx, Dy, Dz)
            mask_camera: (B, Dx, Dy, Dz)
        Returns:

        """
        loss = dict()
        voxel_semantics = voxel_semantics.long()
        if self.use_mask:
            mask_camera = mask_camera.to(torch.int32)   # (B, Dx, Dy, Dz)
            # (B, Dx, Dy, Dz) --> (B*Dx*Dy*Dz, )
            voxel_semantics = voxel_semantics.reshape(-1)
            # (B, Dx, Dy, Dz, n_cls) --> (B*Dx*Dy*Dz, n_cls)
            preds = occ_pred.reshape(-1, self.num_classes)
            # (B, Dx, Dy, Dz) --> (B*Dx*Dy*Dz, )
            mask_camera = mask_camera.reshape(-1)
            num_total_samples = mask_camera.sum()

            loss_occ = self.loss_occ(
                preds,      # (B*Dx*Dy*Dz, n_cls)
                voxel_semantics,    # (B*Dx*Dy*Dz, )
                mask_camera,        # (B*Dx*Dy*Dz, )
                avg_factor=num_total_samples
            )
            loss['loss_occ'] = loss_occ
        else:
            
            voxel_semantics = voxel_semantics.reshape(-1)
            preds = occ_pred.reshape(-1, self.num_classes)

            loss_occ = self.loss_occ(preds, voxel_semantics)
            loss['loss_occ'] = loss_occ
        return loss

    def get_occ(self, occ_pred, img_metas=None):
        """
        Args:
            occ_pred: (B, Dx, Dy, Dz, C)
            img_metas:

        Returns:
            List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        """
        occ_score = occ_pred.softmax(-1)    # (B, Dx, Dy, Dz, C)
        occ_res = occ_score.argmax(-1)      # (B, Dx, Dy, Dz)
        occ_res = occ_res.cpu().numpy().astype(np.uint8)     # (B, Dx, Dy, Dz)
        return list(occ_res)




@HEADS.register_module()
class FusionOCCHead2D(BaseModule):
    def __init__(self,
                 in_dim=256,
                 out_dim=256,
                 Dz=16,
                 use_mask=True,
                 num_classes=18,
                 use_predicter=False,
                 class_wise=False,
                 loss_occ=None,
                 loss_semantic=None,
                 final_softplus=True,
                 test_threshold=0,
                 ):
        super(FusionOCCHead2D, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.Dz = Dz
        out_channels = out_dim
        self.final_softplus = final_softplus
        # out_channels = out_dim if use_predicter else num_classes * Dz
        self.final_conv = ConvModule(
            self.in_dim,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            conv_cfg=dict(type='Conv2d')
        )
        self.use_predicter = use_predicter
        if use_predicter:
            self.predicter = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim * 2),
                nn.Softplus(),
                nn.Linear(self.out_dim * 2, num_classes * Dz),
            )

        self.use_mask = use_mask
        self.num_classes = num_classes
        self.loss_occ = build_loss(loss_occ)
        self.loss_semantic = build_loss(loss_semantic)

        self.class_wise = class_wise

        if self.final_softplus:
            self.density_mlp = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim*2),
                nn.Softplus(),
                nn.Linear(self.out_dim*2, 2 * self.Dz),
                nn.Softplus(),
            )
        else:
            self.density_mlp = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim*2),
                nn.Softplus(),
                nn.Linear(self.out_dim*2, 2 * self.Dz),
            )

        self.semantic_mlp = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim*2),
            nn.Softplus(),
            nn.Linear(self.out_dim*2, (num_classes-1) * Dz),
        )
        self.test_threshold = test_threshold

    def forward(self, all_feats):
        """
        Args:
            img_feats: (B, C, Dy, Dx)

        Returns:

        """
        fused_feat, pts_feat, img_feat = all_feats

        fused_feat = torch.cat((fused_feat,pts_feat),dim=0)
        # (B, C, Dy, Dx) --> (B, C, Dy, Dx) --> (B, Dx, Dy, C)
        occ_pred = self.final_conv(fused_feat)
        B, C, H, W = occ_pred.shape
        occ_pred = occ_pred.permute(0, 3, 2, 1)
        bs, Dx, Dy = occ_pred.shape[:3]
    
        occ_pred, pts_occ_pred = torch.split(occ_pred,int(occ_pred.shape[0] / 2))
        B = int(B / 2)
        if self.use_predicter:
            # (B, Dx, Dy, C) --> (B, Dx, Dy, 2*C) --> (B, Dx, Dy, Dz*n_cls)
            occ_pred = self.predicter(occ_pred)
            occ_pred = occ_pred.view(bs, Dx, Dy, self.Dz, self.num_classes)
        else:
            # voxel_feats = occ_pred.view(B, H, W, self.Dz,self.out_dim)
            # pts_occ_pred = pts_occ_pred.view(B, H, W, self.Dz,self.out_dim)
            # predict SDF
            density_prob = self.density_mlp(pts_occ_pred).view(B, Dx, Dy, self.Dz, 2)
            semantic = self.semantic_mlp(occ_pred).view(B, Dx, Dy, self.Dz, self.num_classes - 1)
            occ_pred = [density_prob, semantic]
        return occ_pred

    def loss(self, occ_pred, voxel_semantics, mask_camera):
        density_prob, semantic = occ_pred
        voxel_semantics=voxel_semantics.long()
     
        voxel_semantics = voxel_semantics.reshape(-1)
        density_prob = density_prob.reshape(-1, 2)
        semantic = semantic.reshape(-1, self.num_classes-1)
        density_target = (voxel_semantics==17).long()
        semantic_mask = voxel_semantics!=17

        # compute loss
        loss_geo = self.loss_occ(density_prob, density_target)
        loss_sem = self.loss_semantic(semantic[semantic_mask], voxel_semantics[semantic_mask].long())

        loss_ = dict()
        loss_['loss_3d_geo'] = loss_geo
        loss_['loss_3d_sem'] = loss_sem
        return loss_


    # def loss(self, occ_pred, voxel_semantics, mask_camera):
    #     """
    #     Args:
    #         occ_pred: (B, Dx, Dy, Dz, n_cls)
    #         voxel_semantics: (B, Dx, Dy, Dz)
    #         mask_camera: (B, Dx, Dy, Dz)
    #     Returns:

    #     """
    #     loss = dict()
    #     voxel_semantics = voxel_semantics.long()
    #     if self.use_mask:
    #         mask_camera = mask_camera.to(torch.int32)   # (B, Dx, Dy, Dz)
    #         # (B, Dx, Dy, Dz) --> (B*Dx*Dy*Dz, )
    #         voxel_semantics = voxel_semantics.reshape(-1)
    #         # (B, Dx, Dy, Dz, n_cls) --> (B*Dx*Dy*Dz, n_cls)
    #         preds = occ_pred.reshape(-1, self.num_classes)
    #         # (B, Dx, Dy, Dz) --> (B*Dx*Dy*Dz, )
    #         mask_camera = mask_camera.reshape(-1)
    #         num_total_samples = mask_camera.sum()
    #         loss_occ = self.loss_occ(
    #             preds,      # (B*Dx*Dy*Dz, n_cls)
    #             voxel_semantics,    # (B*Dx*Dy*Dz, )
    #             mask_camera,        # (B*Dx*Dy*Dz, )
    #             avg_factor=num_total_samples
    #         )
    #         loss['loss_occ'] = loss_occ
    #     else:
    #         voxel_semantics = voxel_semantics.reshape(-1)
    #         preds = occ_pred.reshape(-1, self.num_classes)
    #         loss_occ = self.loss_occ(preds, voxel_semantics)
    #         loss['loss_occ'] = loss_occ
    #     return loss

    def get_occ(self, occ_pred, img_metas=None):
        """
        Args:
            occ_pred: (B, Dx, Dy, Dz, C)
            img_metas:

        Returns:
            List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        """
        density_prob, semantic = occ_pred
        density = density_prob[...,0]

        # SDF --> Occupancy
        no_empty_mask = density > self.test_threshold
        semantic_res = semantic.argmax(-1)

        B, H, W, Z, C = semantic.shape
        occ = torch.ones((B,H,W,Z), dtype=semantic_res.dtype).to(semantic_res.device)
        occ = occ * (self.num_classes-1)
        occ[no_empty_mask] = semantic_res[no_empty_mask]

        occ_res = occ.cpu().numpy().astype(np.uint8)
        return list(occ_res)




@HEADS.register_module(force=True)
class BEVOCCHead2DOpenOccupancy(BaseModule):
    def __init__(self,
                 in_dim=256,
                 out_dim=256,
                 Dz=16,
                 empty_idx=0,
                 use_mask=True,
                 num_classes=18,
                 use_predicter=True,
                 class_wise=False,
                 loss_weight_cfg=None,
                 balance_cls_weight=True,
                 ):
        super(BEVOCCHead2DOpenOccupancy, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.Dz = Dz
        out_channels = out_dim if use_predicter else num_classes * Dz
        self.final_conv = ConvModule(
            self.in_dim,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            conv_cfg=dict(type='Conv2d')
        )
        self.use_predicter = use_predicter
        if use_predicter:
            self.predicter = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim * 2),
                nn.Softplus(),
                nn.Linear(self.out_dim * 2, num_classes * Dz),
            )

        self.use_mask = use_mask
        self.num_classes = num_classes
        self.class_wise = class_wise

        self.empty_idx = empty_idx
        # loss functions
        if balance_cls_weight:
            self.class_weights = torch.from_numpy(1 / np.log(nusc_class_frequencies_open_occupancy + 0.001))
        else:
            self.class_weights = torch.ones(17)/17  # FIXME hardcode 17


        if loss_weight_cfg is None:
            self.loss_weight_cfg = {
                "loss_voxel_ce_weight": 1.0,
                "loss_voxel_sem_scal_weight": 1.0,
                "loss_voxel_geo_scal_weight": 1.0,
                "loss_voxel_lovasz_weight": 1.0,
            }
        else:
            self.loss_weight_cfg = loss_weight_cfg
        
        # voxel losses
        self.loss_voxel_ce_weight = self.loss_weight_cfg.get('loss_voxel_ce_weight', 1.0)
        self.loss_voxel_sem_scal_weight = self.loss_weight_cfg.get('loss_voxel_sem_scal_weight', 1.0)
        self.loss_voxel_geo_scal_weight = self.loss_weight_cfg.get('loss_voxel_geo_scal_weight', 1.0)
        self.loss_voxel_lovasz_weight = self.loss_weight_cfg.get('loss_voxel_lovasz_weight', 1.0)
        

    def forward(self, img_feats):
        """
        Args:
            img_feats: (B, C, Dy, Dx)

        Returns:

        """
        # (B, C, Dy, Dx) --> (B, C, Dy, Dx) --> (B, Dx, Dy, C)
        occ_pred = self.final_conv(img_feats).permute(0, 3, 2, 1)
        bs, Dx, Dy = occ_pred.shape[:3]
        if self.use_predicter:
            # (B, Dx, Dy, C) --> (B, Dx, Dy, 2*C) --> (B, Dx, Dy, Dz*n_cls)
            occ_pred = self.predicter(occ_pred)
            occ_pred = occ_pred.view(bs, Dx, Dy, self.Dz, self.num_classes)

        return occ_pred

    def loss(self, occ_pred, voxel_semantics, mask_camera=None, tag = 'openoccupancy'):
        output_voxels = occ_pred.permute(0,4,1,2,3)
        target_voxels = voxel_semantics
        # resize gt                       
        B, C, H, W, D = output_voxels.shape
        ratio = target_voxels.shape[2] // H
        if ratio != 1:
            target_voxels = target_voxels.reshape(B, H, ratio, W, ratio, D, ratio).permute(0,1,3,5,2,4,6).reshape(B, H, W, D, ratio**3)
            empty_mask = target_voxels.sum(-1) == self.empty_idx
            target_voxels = target_voxels.to(torch.int64)
            occ_space = target_voxels[~empty_mask]
            occ_space[occ_space==0] = -torch.arange(len(occ_space[occ_space==0])).to(occ_space.device) - 1
            target_voxels[~empty_mask] = occ_space
            target_voxels = torch.mode(target_voxels, dim=-1)[0]
            target_voxels[target_voxels<0] = 255
            target_voxels = target_voxels.long()

        assert torch.isnan(output_voxels).sum().item() == 0
        assert torch.isnan(target_voxels).sum().item() == 0

        loss_dict = {}
        

        # igore 255 = ignore noise. we keep the loss bascward for the label=0 (free voxels)
        loss_dict['loss_voxel_ce_{}'.format(tag)] = self.loss_voxel_ce_weight * CE_ssc_loss(output_voxels, target_voxels, self.class_weights.type_as(output_voxels), ignore_index=255)
        loss_dict['loss_voxel_sem_scal_{}'.format(tag)] = self.loss_voxel_sem_scal_weight * sem_scal_loss(output_voxels, target_voxels, ignore_index=255)
        loss_dict['loss_voxel_geo_scal_{}'.format(tag)] = self.loss_voxel_geo_scal_weight * geo_scal_loss(output_voxels, target_voxels, ignore_index=255, non_empty_idx=self.empty_idx)
        loss_dict['loss_voxel_lovasz_{}'.format(tag)] = self.loss_voxel_lovasz_weight * lovasz_softmax(torch.softmax(output_voxels, dim=1), target_voxels, ignore=255)


     
        return loss_dict

    def get_occ(self, occ_pred, img_metas=None):
        """
        Args:
            occ_pred: (B, Dx, Dy, Dz, C)
            img_metas:

        Returns:
            List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        """
        occ_score = occ_pred.softmax(-1)    # (B, Dx, Dy, Dz, C)
        occ_res = occ_score.argmax(-1)      # (B, Dx, Dy, Dz)
        occ_res = occ_res.cpu().numpy().astype(np.uint8)     # (B, Dx, Dy, Dz)
        return list(occ_res)




@HEADS.register_module(force=True)
class BEVOCCHead2DOcc3D(BaseModule):
    def __init__(self,
                 in_dim=256,
                 out_dim=256,
                 Dz=16,
                 empty_idx=0,
                 use_mask=True,
                 num_classes=18,
                 use_predicter=True,
                 class_wise=False,
                 loss_weight_cfg=None,
                 balance_cls_weight=True,
                 loss_occ=None,
                 ):
        super(BEVOCCHead2DOcc3D, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.Dz = Dz
        out_channels = out_dim if use_predicter else num_classes * Dz
        self.final_conv = ConvModule(
            self.in_dim,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            conv_cfg=dict(type='Conv2d')
        )
        self.use_predicter = use_predicter
        if use_predicter:
            self.predicter = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim * 2),
                nn.Softplus(),
                nn.Linear(self.out_dim * 2, num_classes * Dz),
            )

        self.use_mask = use_mask
        self.num_classes = num_classes
        self.class_wise = class_wise

        self.empty_idx = empty_idx
        self.loss_occ = build_loss(loss_occ)
        # loss functions
        if balance_cls_weight:
            self.class_weights = torch.from_numpy(1 / np.log(nusc_class_frequencies + 0.001))
        else:
            self.class_weights = torch.ones(18)/18  # FIXME hardcode 17


        if loss_weight_cfg is None:
            self.loss_weight_cfg = {
                "loss_voxel_ce_weight": 1.0,
                "loss_voxel_sem_scal_weight": 1.0,
                "loss_voxel_geo_scal_weight": 1.0,
                "loss_voxel_lovasz_weight": 1.0,
            }
        else:
            self.loss_weight_cfg = loss_weight_cfg
        
        # voxel losses
        self.loss_voxel_ce_weight = self.loss_weight_cfg.get('loss_voxel_ce_weight', 1.0)
        self.loss_voxel_sem_scal_weight = self.loss_weight_cfg.get('loss_voxel_sem_scal_weight', 1.0)
        self.loss_voxel_geo_scal_weight = self.loss_weight_cfg.get('loss_voxel_geo_scal_weight', 1.0)
        self.loss_voxel_lovasz_weight = self.loss_weight_cfg.get('loss_voxel_lovasz_weight', 1.0)
        

    def forward(self, img_feats):
        """
        Args:
            img_feats: (B, C, Dy, Dx)

        Returns:

        """
        # (B, C, Dy, Dx) --> (B, C, Dy, Dx) --> (B, Dx, Dy, C)
        occ_pred = self.final_conv(img_feats).permute(0, 3, 2, 1)
        bs, Dx, Dy = occ_pred.shape[:3]
        if self.use_predicter:
            # (B, Dx, Dy, C) --> (B, Dx, Dy, 2*C) --> (B, Dx, Dy, Dz*n_cls)
            occ_pred = self.predicter(occ_pred)
            occ_pred = occ_pred.view(bs, Dx, Dy, self.Dz, self.num_classes)

        return occ_pred

    def loss(self, occ_pred, voxel_semantics, mask_camera=None, tag = 'occ3d'):
        output_voxels = occ_pred.permute(0,4,1,2,3)
        target_voxels = voxel_semantics
        # resize gt                       
        B, C, H, W, D = output_voxels.shape
        ratio = target_voxels.shape[2] // H
        if ratio != 1:
            target_voxels = target_voxels.reshape(B, H, ratio, W, ratio, D, ratio).permute(0,1,3,5,2,4,6).reshape(B, H, W, D, ratio**3)
            empty_mask = target_voxels.sum(-1) == self.empty_idx
            target_voxels = target_voxels.to(torch.int64)
            occ_space = target_voxels[~empty_mask]
            occ_space[occ_space==0] = -torch.arange(len(occ_space[occ_space==0])).to(occ_space.device) - 1
            target_voxels[~empty_mask] = occ_space
            target_voxels = torch.mode(target_voxels, dim=-1)[0]
            target_voxels[target_voxels<0] = 255
            target_voxels = target_voxels.long()

        assert torch.isnan(output_voxels).sum().item() == 0
        assert torch.isnan(target_voxels).sum().item() == 0

        loss_dict = {}
        

        # igore 255 = ignore noise. we keep the loss bascward for the label=0 (free voxels)
        loss_dict['loss_voxel_ce_{}'.format(tag)] = self.loss_voxel_ce_weight * CE_ssc_loss(output_voxels, target_voxels, self.class_weights.type_as(output_voxels), ignore_index=255)
        loss_dict['loss_voxel_sem_scal_{}'.format(tag)] = self.loss_voxel_sem_scal_weight * sem_scal_loss(output_voxels, target_voxels, ignore_index=255)
        loss_dict['loss_voxel_geo_scal_{}'.format(tag)] = self.loss_voxel_geo_scal_weight * geo_scal_loss(output_voxels, target_voxels, ignore_index=255, non_empty_idx=self.empty_idx)
        loss_dict['loss_voxel_lovasz_{}'.format(tag)] = self.loss_voxel_lovasz_weight * lovasz_softmax(torch.softmax(output_voxels, dim=1), target_voxels, ignore=255)

        # voxel_semantics = voxel_semantics.long()
        # if self.use_mask:
        #     mask_camera = mask_camera.to(torch.int32)   # (B, Dx, Dy, Dz)
        #     # (B, Dx, Dy, Dz) --> (B*Dx*Dy*Dz, )
        #     voxel_semantics = voxel_semantics.reshape(-1)
        #     # (B, Dx, Dy, Dz, n_cls) --> (B*Dx*Dy*Dz, n_cls)
        #     preds = occ_pred.reshape(-1, self.num_classes)
        #     # (B, Dx, Dy, Dz) --> (B*Dx*Dy*Dz, )
        #     mask_camera = mask_camera.reshape(-1)
        #     num_total_samples = mask_camera.sum()

        #     loss_occ = self.loss_occ(
        #         preds,      # (B*Dx*Dy*Dz, n_cls)
        #         voxel_semantics,    # (B*Dx*Dy*Dz, )
        #         mask_camera,        # (B*Dx*Dy*Dz, )
        #         avg_factor=num_total_samples
        #     )

        #     loss_dict['loss_voxel_ce_{}'.format(tag)] = loss_occ


        return loss_dict

    def get_occ(self, occ_pred, img_metas=None):
        """
        Args:
            occ_pred: (B, Dx, Dy, Dz, C)
            img_metas:

        Returns:
            List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        """
        occ_score = occ_pred.softmax(-1)    # (B, Dx, Dy, Dz, C)
        occ_res = occ_score.argmax(-1)      # (B, Dx, Dy, Dz)
        occ_res = occ_res.cpu().numpy().astype(np.uint8)     # (B, Dx, Dy, Dz)
        return list(occ_res)
