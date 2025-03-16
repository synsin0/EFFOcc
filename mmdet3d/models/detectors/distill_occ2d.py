# Copyright (c) Phigent Robotics. All rights reserved.
from .bevdet import BEVDet, BEVStereo4D
from mmdet3d.models import DETECTORS
from mmdet3d.models import builder
from mmdet3d.models.builder import build_head
import torch.nn.functional as F
import torch.nn as nn
import torch
from mmcv.runner import force_fp32
from functools import partial
import mmcv
from mmdet3d.models import build_detector
from mmcv.runner import load_checkpoint, load_state_dict
from mmdet3d.utils import collect_env, get_root_logger
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.runner import _load_checkpoint
from mmdet3d.models.builder import build_loss
import numpy as np


def calculate_iou(pred, target, mask_camera):  
    # 获取唯一的类别标签  
    unique_labels = np.unique(np.concatenate((pred.flatten(), target.flatten())))  
    
    max_index = np.argmax(unique_labels)  
    unique_labels = np.delete(unique_labels, max_index)  

    # 初始化IoU和mIoU的列表  
    ious = []  
    total_iou = 0  
      
    # 遍历每个类别  
    for label in unique_labels:  
        # 获取当前类别的预测和真实掩码  
        pred_mask = (pred == label)  
        target_mask = (target == label)  
        
        # pred_mask = np.logical_and(pred_mask, mask_camera)
        # target_mask = np.logical_and(target_mask, mask_camera)
        # 计算交集和并集  
        intersection = np.logical_and(pred_mask, target_mask).sum()  
        union = np.logical_or(pred_mask, target_mask).sum()  
          
        # 避免除以零错误  
        if union == 0:  
            iou = 0  
        else:  
            iou = intersection / union  
          
        # 添加到IoU列表中  
        ious.append(iou)  
          
        # 累加IoU用于计算mIoU  
        total_iou += iou  
      
    # 计算平均IoU  
    mIoU = total_iou / len(unique_labels)  
    # 输出每个类别的IoU和平均IoU  
    print("IoU per class:", ious)  
    print("Mean IoU:", mIoU)
    return ious, mIoU  

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=partial(nn.ReLU, inplace=True), drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1, padding=0)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1, padding=0)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # FIXME to accommodate NCHW tensor
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

@DETECTORS.register_module()
class DistillFlashBEVDetOCCPureDistill(BEVDet):
    def __init__(self,
                 teacher_config, 
                 teacher_ckpt, 
                 self_ckpt = None,
                 distill_type="naive_bev",
                 distill_loss_type = "l1",
                 transformer = None,
                 inherit_head = False,
                 occ_head=None,
                 occ_fuser=None,
                 upsample=False,
                 use_infov_mask=False,
                 use_lidar_mask=False,
                 use_camera_mask=True,
                 feat_criterion=dict(type='MSELoss', reduction='none'),
                 occ_criterion=dict(type='CrossEntropyLoss', use_sigmoid=False, reduction='none', ignore_index=255, loss_weight=1.0),
                 **kwargs):
        super(DistillFlashBEVDetOCCPureDistill, self).__init__(**kwargs)
        self.occ_head = build_head(occ_head)
        self.occ_fuser = builder.build_fusion_layer(occ_fuser) if occ_fuser is not None else None
        self.pts_bbox_head = None
        self.upsample = upsample
        self.use_infov_mask=use_infov_mask
        self.use_lidar_mask=use_lidar_mask
        self.use_camera_mask=use_camera_mask
        if isinstance(teacher_config, str):
            teacher_config = mmcv.Config.fromfile(teacher_config)
        self.logger = get_root_logger()
        
        self.teacher_config = teacher_config
        self.teacher_ckpt = teacher_ckpt

        # if isinstance(self_ckpt, str) and self_ckpt.lower() != 'none':
        #     ckpt = _load_checkpoint(
        #         self_ckpt, logger=self.logger, map_location='cpu')

        #     self.load_state_dict(ckpt['state_dict'], strict = False)      

        self.inherit_head = inherit_head
      
        self.distill_loss_type = distill_loss_type
        self.distill_type = distill_type
        if self.distill_type == "mae_distill_bidirectional_transformer":
            self.transformer = build_transformer_layer_sequence(transformer)
            # self.downsample_maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.feat_criterion = build_loss(feat_criterion)
        self.occ_criterion = build_loss(occ_criterion)

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def init_weights(self):
        super(DistillFlashBEVDetOCCPureDistill, self).init_weights()


        self.teacher_model = build_detector(self.teacher_config['model'])

        if isinstance(self.teacher_ckpt, str) and self.teacher_ckpt.lower() != 'none':
            print(f'loading teacher checkpoint from {self.teacher_ckpt}')

            ckpt = _load_checkpoint(
                self.teacher_ckpt, logger=self.logger, map_location='cpu')

            self.teacher_model.load_state_dict(ckpt['state_dict'], True)

        for name, param in self.teacher_model.named_parameters():
            param.requires_grad = False

      
        # self.adaptation_layers = Mlp(in_features=self.occ_head.in_dim, out_features=self.teacher_model.occ_head.in_dim)
             
        if self.inherit_head:
            assert isinstance(self.teacher_ckpt, str) and self.teacher_ckpt.lower() != 'none'

            # self.img_view_transformer.load_state_dict(self.teacher_model.img_view_transformer.state_dict(), strict=False)

            self.occ_head.load_state_dict(self.teacher_model.occ_head.state_dict(), strict=False)

            # self.occ_fuser.load_state_dict(self.teacher_model.occ_fuser.state_dict(), strict=False)
            # self.pts_backbone.load_state_dict(self.teacher_model.pts_backbone.state_dict(), strict=False)
            # self.pts_neck.load_state_dict(self.teacher_model.pts_neck.state_dict(), strict=False)


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

        # set `fp16_enabled` flag
        # if hasattr(self, 'fp16_enabled') and self.fp16_enabled:
        #     for m in self.teacher_model.modules():
        #         if hasattr(m, 'fp16_enabled'):
        #             m.fp16_enabled = True
        with torch.no_grad():
           
            # img_feats, lss_feat, bev_backbone_feats
            # teacher_bev_feat = self.teacher_model.extract_bev_feat(
            #     img_inputs = img_inputs, img_metas = img_metas, points = points, **kwargs)
            teacher_bev_feat, teacher_bev_occ, teacher_visual_bev_feat, teacher_lidar_bev_feat = self.teacher_model.extract_bev_feat_occ(
                img_inputs = img_inputs, img_metas = img_metas, points = points, **kwargs
            )
            
            teacher_preds = self.teacher_model.simple_test_occ(teacher_bev_feat)
            
            # teacher_preds = self.teacher_model.simple_test(
            #     img = img_inputs, img_metas = img_metas, points = points, **kwargs)
           
        # print(points)
        img_feats, pts_feats, depth = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)

        # visualC = teacher_visual_bev_feat.shape[1]
        # lidarC = teacher_lidar_bev_feat.shape[1]
        # # print(teacher_visual_bev_feat.shape, teacher_lidar_bev_feat.shape)
        # student_visual_bev_feat = img_feats[0][:, :visualC, ...]
        # student_lidar_bev_feat = img_feats[0][:, visualC:visualC+lidarC, ...]
        # # print(student_visual_bev_feat.shape, student_lidar_bev_feat.shape) # visual+lidar bev feature

        losses = dict()
        voxel_semantics = kwargs['voxel_semantics']     # (B, Dx, Dy, Dz)
        mask_camera = kwargs['mask_camera']     # (B, Dx, Dy, Dz)
        mask_lidar = kwargs.get('mask_lidar')
        mask_infov = kwargs.get('mask_infov')

        # ious, mIoU = calculate_iou(np.stack(teacher_preds), voxel_semantics.cpu().numpy(), mask_camera.cpu().numpy())

        final_mask = torch.ones_like(voxel_semantics)
        if self.use_infov_mask:
            final_mask = torch.logical_and(mask_infov, final_mask)
        if self.use_lidar_mask:
            final_mask = torch.logical_and(mask_lidar, final_mask)
        if self.use_camera_mask:
            final_mask = torch.logical_and(mask_camera, final_mask)
        final_mask = final_mask.bool()

        occ_bev_feature = img_feats[0]

        # occ_bev_feature = self.occ_fuser(student_visual_bev_feat, student_lidar_bev_feat)
        
        # occ_bev_feature = self.pts_backbone(occ_bev_feature)
        # if self.with_pts_neck:
        #     occ_bev_feature = self.pts_neck(occ_bev_feature)
        
        # occ_bev_feature = occ_bev_feature[0]

        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)

        occ, loss_occ = self.forward_occ_train(occ_bev_feature, voxel_semantics, final_mask, img_metas)
        # losses.update(loss_occ)
        
        # print(occ_bev_feature.shape, teacher_bev_feat.shape)
        if self.distill_type == 'naive_bev' or self.distill_type == 'fgbg_bev':
            student_feat = occ_bev_feature
            teacher_feat = teacher_bev_feat

            student_occ = occ
            teacher_occ = teacher_bev_occ
            # print(torch.max(student_occ), torch.min(student_occ), torch.max(teacher_occ), torch.min(teacher_occ))
        
        elif self.distill_type =='mae_distill_bidirectional_transformer':
            mask_ratio = 0.75
            # occ_bev_feature = self.downsample_maxpool(self.downsample_maxpool(occ_bev_feature))
            B, C, H, W = occ_bev_feature.shape
            code = occ_bev_feature.flatten(2,3).permute(0,2,1)
            # masking: length -> length * mask_ratio
            x, mask, ids_restore = self.random_masking(code, mask_ratio)
            # append mask tokens to sequence
            visible_tokens = x.clone()
            mask_tokens = self.transformer.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
            x = torch.cat([x, mask_tokens], dim=1)  # no cls token
            x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
            predicted_tokens = self.transformer(x)
            predicted_tokens = predicted_tokens.view(B, -1, C)
            predicted_tokens[mask==0] = visible_tokens.view(-1,C)
            predicted_tokens = predicted_tokens.view(B, H, W, C).permute(0,3,1,2)
            
            # predicted_tokens = F.interpolate(predicted_tokens, scale_factor=4,
            #                                 mode='bilinear', align_corners=True)
            student_feat = predicted_tokens
            teacher_feat = teacher_bev_feat
        
        if self.distill_type == 'fgbg_bev':
            fg_feat_loss_weights = 1
            bg_feat_loss_weights = 1
            empty_feat_loss_weights = 1

            # sample_with_label = torch.from_numpy(np.array([img_meta['use_with_label'] for img_meta in img_metas])).to(student_feat.device)


            B = voxel_semantics.shape[0]
            foreground_mask, background_mask, empty_mask = (voxel_semantics <= 10).any(dim=3), torch.logical_and((voxel_semantics > 10).any(dim=3), (voxel_semantics < 17).any(dim=3)), (voxel_semantics == 17).all(dim=3)
            # foreground_mask, background_mask, empty_mask = [], [], []
            # teacher_preds = torch.from_numpy(np.stack(teacher_preds, axis=0)).to(student_feat.device)

            # for bs in range(B):
            #     if sample_with_label[bs]:
            #         for_mask, back_mask, em_mask = (voxel_semantics[bs] <= 10).any(dim=2), torch.logical_and((voxel_semantics[bs] > 10).any(dim=2), (voxel_semantics[bs] < 17).any(dim=2)), (voxel_semantics[bs] == 17).all(dim=2)
          
            #     else:
            #         for_mask, back_mask, em_mask = (teacher_preds[bs] <= 10).any(dim=2), torch.logical_and((teacher_preds[bs] > 10).any(dim=2), (teacher_preds[bs] < 17).any(dim=2)), (teacher_preds[bs] == 17).all(dim=2)
            #     foreground_mask.append(for_mask)
            #     background_mask.append(back_mask)
            #     empty_mask.append(em_mask)


            # foreground_mask = torch.stack(foreground_mask)
            # background_mask = torch.stack(background_mask)
            # empty_mask = torch.stack(empty_mask)
            
            # # # distill visual bev feature loss
            # C = student_visual_bev_feat.shape[1]
            # losses['fg_distill_visual_bev_loss'] = (self.feat_criterion(student_visual_bev_feat, teacher_visual_bev_feat) * foreground_mask.unsqueeze(1).repeat(1,C,1,1)).sum() \
            #                 / torch.sum(foreground_mask) / C * fg_feat_loss_weights
            # losses['bg_distill_visual_bev_loss'] = (self.feat_criterion(student_visual_bev_feat, teacher_visual_bev_feat) * background_mask.unsqueeze(1).repeat(1,C,1,1)).sum() \
            #                 / torch.sum(background_mask) / C * bg_feat_loss_weights
            # losses['empty_distill_visual_bev_loss'] = (self.feat_criterion(student_visual_bev_feat, teacher_visual_bev_feat) * empty_mask.unsqueeze(1).repeat(1,C,1,1)).sum() \
            #                 / torch.sum(empty_mask) / C * empty_feat_loss_weights

            # # distill lidar bev feature loss
            # C = student_lidar_bev_feat.shape[1]
            # losses['fg_distill_lidar_bev_loss'] = (self.feat_criterion(student_lidar_bev_feat, teacher_lidar_bev_feat) * foreground_mask.unsqueeze(1).repeat(1,C,1,1)).sum() \
            #                 / torch.sum(foreground_mask) / C * fg_feat_loss_weights
            # losses['bg_distill_lidar_bev_loss'] = (self.feat_criterion(student_lidar_bev_feat, teacher_lidar_bev_feat) * background_mask.unsqueeze(1).repeat(1,C,1,1)).sum() \
            #                 / torch.sum(background_mask) / C * bg_feat_loss_weights
            # losses['empty_distill_lidar_bev_loss'] = (self.feat_criterion(student_lidar_bev_feat, teacher_lidar_bev_feat) * empty_mask.unsqueeze(1).repeat(1,C,1,1)).sum() \
            #                 / torch.sum(empty_mask) / C * empty_feat_loss_weights

            # distill bev feature loss
            C = student_feat.shape[1]
            if torch.sum(foreground_mask) > 0:
                losses['fg_distill_bev_loss'] = (self.feat_criterion(student_feat, teacher_feat) * foreground_mask.unsqueeze(1).repeat(1,C,1,1)).sum() \
                                / torch.sum(foreground_mask) / C * fg_feat_loss_weights
            else:
                losses['fg_distill_bev_loss'] = torch.tensor(0., requires_grad=True).to(student_feat.device)

            if torch.sum(background_mask) > 0:
                losses['bg_distill_bev_loss'] = (self.feat_criterion(student_feat, teacher_feat) * background_mask.unsqueeze(1).repeat(1,C,1,1)).sum() \
                                / torch.sum(background_mask) / C * bg_feat_loss_weights
            else:
                losses['bg_distill_bev_loss'] = torch.tensor(0., requires_grad=True).to(student_feat.device)

            if torch.sum(empty_mask) > 0:
                losses['empty_distill_bev_loss'] = (self.feat_criterion(student_feat, teacher_feat) * empty_mask.unsqueeze(1).repeat(1,C,1,1)).sum() \
                                / torch.sum(empty_mask) / C * empty_feat_loss_weights
            else:
                losses['empty_distill_bev_loss'] = torch.tensor(0., requires_grad=True).to(student_feat.device)

            # distill bev occ loss


            foreground_mask_3d, background_mask_3d, empty_mask_3d = (voxel_semantics <= 10), torch.logical_and((voxel_semantics > 10), (voxel_semantics < 17)), (voxel_semantics == 17)


            # foreground_mask_3d, background_mask_3d, empty_mask_3d = [], [], []

            # for bs in range(B):
            #     if sample_with_label[bs]:
            #         for_mask3d, back_mask3d, em_mask3d = (voxel_semantics[bs] <= 10), torch.logical_and((voxel_semantics[bs] > 10), (voxel_semantics[bs] < 17)), (voxel_semantics[bs] == 17)
          
            #     else:
            #         for_mask3d, back_mask3d, em_mask3d = (teacher_preds[bs] <= 10), torch.logical_and((teacher_preds[bs] > 10), (teacher_preds[bs] < 17)), (teacher_preds[bs] == 17)
            #     foreground_mask_3d.append(for_mask3d)
            #     background_mask_3d.append(back_mask3d)
            #     empty_mask_3d.append(em_mask3d)


            # foreground_mask_3d = torch.stack(foreground_mask_3d)
            # background_mask_3d = torch.stack(background_mask_3d)
            # empty_mask_3d = torch.stack(empty_mask_3d)

            if torch.sum(foreground_mask_3d) > 0:
                losses['fg_distill_occ_loss'] = ((1 - F.cosine_similarity(student_occ, teacher_occ, dim=-1)) * foreground_mask_3d).sum() / torch.sum(foreground_mask_3d) * fg_feat_loss_weights
            else:
                losses['fg_distill_occ_loss'] = torch.tensor(0., requires_grad=True).to(student_feat.device)

            if torch.sum(background_mask_3d) > 0:
                losses['bg_distill_occ_loss'] = ((1 - F.cosine_similarity(student_occ, teacher_occ, dim=-1)) * background_mask_3d).sum() / torch.sum(background_mask_3d) * bg_feat_loss_weights
            else:
                losses['bg_distill_occ_loss'] = torch.tensor(0., requires_grad=True).to(student_feat.device)

            if torch.sum(empty_mask_3d) > 0:
                losses['empty_distill_occ_loss'] = ((1 - F.cosine_similarity(student_occ, teacher_occ, dim=-1)) * empty_mask_3d).sum() / torch.sum(empty_mask_3d) * empty_feat_loss_weights
            else:
                losses['empty_distill_occ_loss'] = torch.tensor(0., requires_grad=True).to(student_feat.device)

            # distill bev occ loss
            # C = student_occ.shape[-1]
            # foreground_mask_3d, background_mask_3d, empty_mask_3d = (voxel_semantics <= 10), torch.logical_and((voxel_semantics > 10), (voxel_semantics < 17)), (voxel_semantics == 17)
            # losses['fg_distill_occ_loss'] = (self.feat_criterion(student_occ, teacher_occ) * foreground_mask_3d.unsqueeze(-1).repeat(1,1,1,1,C)).sum() / torch.sum(foreground_mask_3d) / C * fg_feat_loss_weights * 0.1
            # losses['bg_distill_occ_loss'] = (self.feat_criterion(student_occ, teacher_occ) * background_mask_3d.unsqueeze(-1).repeat(1,1,1,1,C)).sum() / torch.sum(background_mask_3d) / C * bg_feat_loss_weights * 0.1
            # losses['empty_distill_occ_loss'] = (self.feat_criterion(student_occ, teacher_occ) * empty_mask_3d.unsqueeze(-1).repeat(1,1,1,1,C)).sum() / torch.sum(empty_mask_3d) / C * empty_feat_loss_weights * 0.1

            # student_occ_logits = student_occ.softmax(dim=-1)
            # teacher_occ_logits = teacher_occ.softmax(dim=-1)

            # kl_diversity = (teacher_occ_logits * torch.log(teacher_occ_logits) - teacher_occ_logits * torch.log(student_occ_logits)).sum(dim=-1)
            # cross_entropy = (- teacher_occ_logits * F.log_softmax(student_occ, dim=-1)).sum(dim=-1)

            # losses['distill_cross_entropy_loss'] = cross_entropy.mean()
            # print(cross_entropy.mean())

            # print(losses)

        else:
            p = 1
            if p == 1:
                distill_loss = F.l1_loss(student_feat, teacher_feat) 
            elif p == 2:
                distill_loss = F.mse_loss(student_feat, teacher_feat)
            
            losses['distill_loss'] = distill_loss
         
        return losses

    def forward_occ_train(self, img_feats, voxel_semantics, mask_camera, img_metas):
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
        return outs, loss_occ

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

        # visualC = 32
        # lidarC = 512

        # student_visual_bev_feat = img_feats[0][:, :visualC, ...]
        # student_lidar_bev_feat = img_feats[0][:, visualC:visualC+lidarC, ...]

        # occ_bev_feature = self.occ_fuser(student_visual_bev_feat, student_lidar_bev_feat)
        
        # occ_bev_feature = self.pts_backbone(occ_bev_feature)
        # if self.with_pts_neck:
        #     occ_bev_feature = self.pts_neck(occ_bev_feature)
        
        # occ_bev_feature = occ_bev_feature[0]

        occ_bev_feature = img_feats[0]
        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)

        occ_list = self.simple_test_occ(occ_bev_feature, img_metas)    # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        return occ_list

    # def simple_test(self,  # this function is for testing teacher
    #                 points,
    #                 img_metas,
    #                 img=None,
    #                 rescale=False,
    #                 **kwargs):

    #     return self.teacher_model.simple_test(points,
    #                 img_metas,
    #                 img=img,
    #                 rescale=rescale,
    #                 **kwargs)

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

    @force_fp32(apply_to=('teacher_feat', 'student_feat', 'teacher_preds', 'student_preds', 'heatmaps'))
    def distill_loss(self, teacher_feat, student_feat, teacher_preds, student_preds,
                     heatmaps, anno_boxes, inds, masks, gt_bboxes_3d, gt_labels_3d,
                     canvas_feat, index):
        # FIXME for now, only support tensor distillation.
        # TODO: add list distillation in the future
        # for input of size (256.704)
        # typically bevdet feature is of size (256,128,128)
        # centerpoint feature is of size (384,128,128)
        assert isinstance(teacher_feat, torch.Tensor) and isinstance(student_feat, torch.Tensor)
        # ensure each pixel on teacher feature map and student feature map have the same field-of-view
        if self.distill_type == 'all':
            losses_distill = self.all_distill_loss(teacher_feat, student_feat, index)
        elif self.distill_type == 'foreground_background':
            losses_distill = self.foreground_background_distill_loss(teacher_feat, student_feat, gt_bboxes_3d, index)
        elif self.distill_type == 's2m2_ssd_heatmap':
            tp_mask, fp_mask, fn_mask = self.s2m2_ssd_mask(heatmaps, student_preds)
            losses_distill = self.s2m2_ssd_heatmap_distill_loss(teacher_preds, student_preds, tp_mask, fp_mask, fn_mask)
        # elif self.distill_type == 's2m2_ssd_reg':
        #     tp_mask, fp_mask, fn_mask = self.s2m2_ssd_mask(heatmaps, student_preds)
        elif self.distill_type == 's2m2_ssd_feature':
            tp_mask, fp_mask, fn_mask = self.s2m2_ssd_mask(heatmaps, student_preds)
            losses_distill = self.s2m2_ssd_feature_distill_loss(teacher_feat, student_feat, tp_mask, fp_mask, fn_mask, index)
        elif self.distill_type == 'gauss_focal_heatmap':
            losses_distill = self.gauss_focal_heatmap_distill_loss(teacher_preds, student_preds, heatmaps)
        elif self.distill_type == 'non_local':
            losses_distill = self.non_local_distill_loss(teacher_feat, student_feat, index)
        elif self.distill_type == 'affinity':
            losses_distill = self.affinity_distill_loss(teacher_feat, student_feat, index)
        elif self.distill_type == 'linfengzhang':
            losses_distill = self.linfengzhang_distill_loss(teacher_feat, student_feat, index)
        elif self.distill_type == 'fgd':
            losses_distill = self.fgd_distill_loss(teacher_feat, student_feat,
                                                   gt_bboxes_3d, gt_labels_3d,
                                                   canvas_feat,
                                                   heatmaps, teacher_preds, student_preds, index)
        else:
            raise NotImplementedError

        return losses_distill

@DETECTORS.register_module(force=True)
class DistillFlashBEVStereo4DOCC(BEVStereo4D):
    def __init__(self,
                 occ_head=None,
                 use_infov_mask=False,
                 use_lidar_mask=False,
                 use_camera_mask=True,
                 **kwargs):
        super(DistillFlashBEVStereo4DOCC, self).__init__(**kwargs)
        self.occ_head = build_head(occ_head)
        self.pts_bbox_head = None
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




@DETECTORS.register_module(force=True)
class DistillFlashBEVDetOCC(BEVDet):
    def __init__(self,
                 teacher_config, 
                 teacher_ckpt, 
                 self_ckpt = None,
                 distill_type="naive_bev",
                 distill_loss_type = "l1",
                 transformer = None,
                 inherit_head = False,
                 occ_head=None,
                 occ_fuser=None,
                 upsample=False,
                 use_infov_mask=False,
                 use_lidar_mask=False,
                 use_camera_mask=True,
                 feat_criterion=dict(type='MSELoss', reduction='none'),
                 occ_criterion=dict(type='CrossEntropyLoss', use_sigmoid=False, reduction='none', ignore_index=255, loss_weight=1.0),
                 **kwargs):
        super(DistillFlashBEVDetOCC, self).__init__(**kwargs)
        self.occ_head = build_head(occ_head)
        self.occ_fuser = builder.build_fusion_layer(occ_fuser) if occ_fuser is not None else None
        self.pts_bbox_head = None
        self.upsample = upsample
        self.use_infov_mask=use_infov_mask
        self.use_lidar_mask=use_lidar_mask
        self.use_camera_mask=use_camera_mask
        if isinstance(teacher_config, str):
            teacher_config = mmcv.Config.fromfile(teacher_config)
        self.logger = get_root_logger()
        
        self.teacher_config = teacher_config
        self.teacher_ckpt = teacher_ckpt

        # if isinstance(self_ckpt, str) and self_ckpt.lower() != 'none':
        #     ckpt = _load_checkpoint(
        #         self_ckpt, logger=self.logger, map_location='cpu')

        #     self.load_state_dict(ckpt['state_dict'], strict = False)      

        self.inherit_head = inherit_head
      
        self.distill_loss_type = distill_loss_type
        self.distill_type = distill_type
        if self.distill_type == "mae_distill_bidirectional_transformer":
            self.transformer = build_transformer_layer_sequence(transformer)
            # self.downsample_maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.feat_criterion = build_loss(feat_criterion)
        self.occ_criterion = build_loss(occ_criterion)

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def init_weights(self):
        super(DistillFlashBEVDetOCC, self).init_weights()


        self.teacher_model = build_detector(self.teacher_config['model'])

        if isinstance(self.teacher_ckpt, str) and self.teacher_ckpt.lower() != 'none':
            print(f'loading teacher checkpoint from {self.teacher_ckpt}')

            ckpt = _load_checkpoint(
                self.teacher_ckpt, logger=self.logger, map_location='cpu')

            self.teacher_model.load_state_dict(ckpt['state_dict'], True)

        for name, param in self.teacher_model.named_parameters():
            param.requires_grad = False

      
        # self.adaptation_layers = Mlp(in_features=self.occ_head.in_dim, out_features=self.teacher_model.occ_head.in_dim)
             
        if self.inherit_head:
            assert isinstance(self.teacher_ckpt, str) and self.teacher_ckpt.lower() != 'none'

            # self.img_view_transformer.load_state_dict(self.teacher_model.img_view_transformer.state_dict(), strict=False)

            self.occ_head.load_state_dict(self.teacher_model.occ_head.state_dict(), strict=False)

            # self.occ_fuser.load_state_dict(self.teacher_model.occ_fuser.state_dict(), strict=False)
            # self.pts_backbone.load_state_dict(self.teacher_model.pts_backbone.state_dict(), strict=False)
            # self.pts_neck.load_state_dict(self.teacher_model.pts_neck.state_dict(), strict=False)


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

        # set `fp16_enabled` flag
        # if hasattr(self, 'fp16_enabled') and self.fp16_enabled:
        #     for m in self.teacher_model.modules():
        #         if hasattr(m, 'fp16_enabled'):
        #             m.fp16_enabled = True
        with torch.no_grad():
           
            # img_feats, lss_feat, bev_backbone_feats
            # teacher_bev_feat = self.teacher_model.extract_bev_feat(
            #     img_inputs = img_inputs, img_metas = img_metas, points = points, **kwargs)
            teacher_bev_feat, teacher_bev_occ, teacher_visual_bev_feat, teacher_lidar_bev_feat = self.teacher_model.extract_bev_feat_occ(
                img_inputs = img_inputs, img_metas = img_metas, points = points, **kwargs
            )
            
            teacher_preds = self.teacher_model.simple_test_occ(teacher_bev_feat)
            
            # teacher_preds = self.teacher_model.simple_test(
            #     img = img_inputs, img_metas = img_metas, points = points, **kwargs)
           
        # print(points)
        img_feats, pts_feats, depth = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)

        # visualC = teacher_visual_bev_feat.shape[1]
        # lidarC = teacher_lidar_bev_feat.shape[1]
        # # print(teacher_visual_bev_feat.shape, teacher_lidar_bev_feat.shape)
        # student_visual_bev_feat = img_feats[0][:, :visualC, ...]
        # student_lidar_bev_feat = img_feats[0][:, visualC:visualC+lidarC, ...]
        # # print(student_visual_bev_feat.shape, student_lidar_bev_feat.shape) # visual+lidar bev feature

        losses = dict()
        voxel_semantics = kwargs['voxel_semantics']     # (B, Dx, Dy, Dz)
        mask_camera = kwargs['mask_camera']     # (B, Dx, Dy, Dz)
        mask_lidar = kwargs.get('mask_lidar')
        mask_infov = kwargs.get('mask_infov')

        # ious, mIoU = calculate_iou(np.stack(teacher_preds), voxel_semantics.cpu().numpy(), mask_camera.cpu().numpy())

        final_mask = torch.ones_like(voxel_semantics)
        if self.use_infov_mask:
            final_mask = torch.logical_and(mask_infov, final_mask)
        if self.use_lidar_mask:
            final_mask = torch.logical_and(mask_lidar, final_mask)
        if self.use_camera_mask:
            final_mask = torch.logical_and(mask_camera, final_mask)
        final_mask = final_mask.bool()

        occ_bev_feature = img_feats[0]

        # occ_bev_feature = self.occ_fuser(student_visual_bev_feat, student_lidar_bev_feat)
        
        # occ_bev_feature = self.pts_backbone(occ_bev_feature)
        # if self.with_pts_neck:
        #     occ_bev_feature = self.pts_neck(occ_bev_feature)
        
        # occ_bev_feature = occ_bev_feature[0]

        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)

        occ, loss_occ = self.forward_occ_train(occ_bev_feature, voxel_semantics, final_mask, img_metas)
        losses.update(loss_occ)
        
        # print(occ_bev_feature.shape, teacher_bev_feat.shape)
        if self.distill_type == 'naive_bev' or self.distill_type == 'fgbg_bev':
            student_feat = occ_bev_feature
            teacher_feat = teacher_bev_feat

            student_occ = occ
            teacher_occ = teacher_bev_occ
            # print(torch.max(student_occ), torch.min(student_occ), torch.max(teacher_occ), torch.min(teacher_occ))
        
        elif self.distill_type =='mae_distill_bidirectional_transformer':
            mask_ratio = 0.75
            # occ_bev_feature = self.downsample_maxpool(self.downsample_maxpool(occ_bev_feature))
            B, C, H, W = occ_bev_feature.shape
            code = occ_bev_feature.flatten(2,3).permute(0,2,1)
            # masking: length -> length * mask_ratio
            x, mask, ids_restore = self.random_masking(code, mask_ratio)
            # append mask tokens to sequence
            visible_tokens = x.clone()
            mask_tokens = self.transformer.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
            x = torch.cat([x, mask_tokens], dim=1)  # no cls token
            x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
            predicted_tokens = self.transformer(x)
            predicted_tokens = predicted_tokens.view(B, -1, C)
            predicted_tokens[mask==0] = visible_tokens.view(-1,C)
            predicted_tokens = predicted_tokens.view(B, H, W, C).permute(0,3,1,2)
            
            # predicted_tokens = F.interpolate(predicted_tokens, scale_factor=4,
            #                                 mode='bilinear', align_corners=True)
            student_feat = predicted_tokens
            teacher_feat = teacher_bev_feat
        
        if self.distill_type == 'fgbg_bev':
            fg_feat_loss_weights = 1
            bg_feat_loss_weights = 1
            empty_feat_loss_weights = 1

            sample_with_label = torch.from_numpy(np.array([img_meta['use_with_label'] for img_meta in img_metas])).to(student_feat.device)


            B = voxel_semantics.shape[0]
            # foreground_mask, background_mask, empty_mask = (voxel_semantics <= 10).any(dim=3), torch.logical_and((voxel_semantics > 10).any(dim=3), (voxel_semantics < 17).any(dim=3)), (voxel_semantics == 17).all(dim=3)
            foreground_mask, background_mask, empty_mask = [], [], []
            teacher_preds = torch.from_numpy(np.stack(teacher_preds, axis=0)).to(student_feat.device)

            for bs in range(B):
                if sample_with_label[bs]:
                    for_mask, back_mask, em_mask = (voxel_semantics[bs] <= 10).any(dim=2), torch.logical_and((voxel_semantics[bs] > 10).any(dim=2), (voxel_semantics[bs] < 17).any(dim=2)), (voxel_semantics[bs] == 17).all(dim=2)
          
                else:
                    for_mask, back_mask, em_mask = (teacher_preds[bs] <= 10).any(dim=2), torch.logical_and((teacher_preds[bs] > 10).any(dim=2), (teacher_preds[bs] < 17).any(dim=2)), (teacher_preds[bs] == 17).all(dim=2)
                foreground_mask.append(for_mask)
                background_mask.append(back_mask)
                empty_mask.append(em_mask)


            foreground_mask = torch.stack(foreground_mask)
            background_mask = torch.stack(background_mask)
            empty_mask = torch.stack(empty_mask)
            
            # # # distill visual bev feature loss
            # C = student_visual_bev_feat.shape[1]
            # losses['fg_distill_visual_bev_loss'] = (self.feat_criterion(student_visual_bev_feat, teacher_visual_bev_feat) * foreground_mask.unsqueeze(1).repeat(1,C,1,1)).sum() \
            #                 / torch.sum(foreground_mask) / C * fg_feat_loss_weights
            # losses['bg_distill_visual_bev_loss'] = (self.feat_criterion(student_visual_bev_feat, teacher_visual_bev_feat) * background_mask.unsqueeze(1).repeat(1,C,1,1)).sum() \
            #                 / torch.sum(background_mask) / C * bg_feat_loss_weights
            # losses['empty_distill_visual_bev_loss'] = (self.feat_criterion(student_visual_bev_feat, teacher_visual_bev_feat) * empty_mask.unsqueeze(1).repeat(1,C,1,1)).sum() \
            #                 / torch.sum(empty_mask) / C * empty_feat_loss_weights

            # # distill lidar bev feature loss
            # C = student_lidar_bev_feat.shape[1]
            # losses['fg_distill_lidar_bev_loss'] = (self.feat_criterion(student_lidar_bev_feat, teacher_lidar_bev_feat) * foreground_mask.unsqueeze(1).repeat(1,C,1,1)).sum() \
            #                 / torch.sum(foreground_mask) / C * fg_feat_loss_weights
            # losses['bg_distill_lidar_bev_loss'] = (self.feat_criterion(student_lidar_bev_feat, teacher_lidar_bev_feat) * background_mask.unsqueeze(1).repeat(1,C,1,1)).sum() \
            #                 / torch.sum(background_mask) / C * bg_feat_loss_weights
            # losses['empty_distill_lidar_bev_loss'] = (self.feat_criterion(student_lidar_bev_feat, teacher_lidar_bev_feat) * empty_mask.unsqueeze(1).repeat(1,C,1,1)).sum() \
            #                 / torch.sum(empty_mask) / C * empty_feat_loss_weights

            # distill bev feature loss
            C = student_feat.shape[1]
            if torch.sum(foreground_mask) > 0:
                losses['fg_distill_bev_loss'] = (self.feat_criterion(student_feat, teacher_feat) * foreground_mask.unsqueeze(1).repeat(1,C,1,1)).sum() \
                                / torch.sum(foreground_mask) / C * fg_feat_loss_weights
            else:
                losses['fg_distill_bev_loss'] = torch.tensor(0., requires_grad=True).to(student_feat.device)

            if torch.sum(background_mask) > 0:
                losses['bg_distill_bev_loss'] = (self.feat_criterion(student_feat, teacher_feat) * background_mask.unsqueeze(1).repeat(1,C,1,1)).sum() \
                                / torch.sum(background_mask) / C * bg_feat_loss_weights
            else:
                losses['bg_distill_bev_loss'] = torch.tensor(0., requires_grad=True).to(student_feat.device)

            if torch.sum(empty_mask) > 0:
                losses['empty_distill_bev_loss'] = (self.feat_criterion(student_feat, teacher_feat) * empty_mask.unsqueeze(1).repeat(1,C,1,1)).sum() \
                                / torch.sum(empty_mask) / C * empty_feat_loss_weights
            else:
                losses['empty_distill_bev_loss'] = torch.tensor(0., requires_grad=True).to(student_feat.device)

            # distill bev occ loss


            # foreground_mask_3d, background_mask_3d, empty_mask_3d = (voxel_semantics <= 10), torch.logical_and((voxel_semantics > 10), (voxel_semantics < 17)), (voxel_semantics == 17)


            foreground_mask_3d, background_mask_3d, empty_mask_3d = [], [], []

            for bs in range(B):
                if sample_with_label[bs]:
                    for_mask3d, back_mask3d, em_mask3d = (voxel_semantics[bs] <= 10), torch.logical_and((voxel_semantics[bs] > 10), (voxel_semantics[bs] < 17)), (voxel_semantics[bs] == 17)
                else:
                    for_mask3d, back_mask3d, em_mask3d = (teacher_preds[bs] <= 10), torch.logical_and((teacher_preds[bs] > 10), (teacher_preds[bs] < 17)), (teacher_preds[bs] == 17)
                foreground_mask_3d.append(for_mask3d)
                background_mask_3d.append(back_mask3d)
                empty_mask_3d.append(em_mask3d)


            foreground_mask_3d = torch.stack(foreground_mask_3d)
            background_mask_3d = torch.stack(background_mask_3d)
            empty_mask_3d = torch.stack(empty_mask_3d)

            if torch.sum(foreground_mask_3d) > 0:
                losses['fg_distill_occ_loss'] = ((1 - F.cosine_similarity(student_occ, teacher_occ, dim=-1)) * foreground_mask_3d).sum() / torch.sum(foreground_mask_3d) * fg_feat_loss_weights
            else:
                losses['fg_distill_occ_loss'] = torch.tensor(0., requires_grad=True).to(student_feat.device)

            if torch.sum(background_mask_3d) > 0:
                losses['bg_distill_occ_loss'] = ((1 - F.cosine_similarity(student_occ, teacher_occ, dim=-1)) * background_mask_3d).sum() / torch.sum(background_mask_3d) * bg_feat_loss_weights
            else:
                losses['bg_distill_occ_loss'] = torch.tensor(0., requires_grad=True).to(student_feat.device)

            if torch.sum(empty_mask_3d) > 0:
                losses['empty_distill_occ_loss'] = ((1 - F.cosine_similarity(student_occ, teacher_occ, dim=-1)) * empty_mask_3d).sum() / torch.sum(empty_mask_3d) * empty_feat_loss_weights
            else:
                losses['empty_distill_occ_loss'] = torch.tensor(0., requires_grad=True).to(student_feat.device)

            # distill bev occ loss
            # C = student_occ.shape[-1]
            # foreground_mask_3d, background_mask_3d, empty_mask_3d = (voxel_semantics <= 10), torch.logical_and((voxel_semantics > 10), (voxel_semantics < 17)), (voxel_semantics == 17)
            # losses['fg_distill_occ_loss'] = (self.feat_criterion(student_occ, teacher_occ) * foreground_mask_3d.unsqueeze(-1).repeat(1,1,1,1,C)).sum() / torch.sum(foreground_mask_3d) / C * fg_feat_loss_weights * 0.1
            # losses['bg_distill_occ_loss'] = (self.feat_criterion(student_occ, teacher_occ) * background_mask_3d.unsqueeze(-1).repeat(1,1,1,1,C)).sum() / torch.sum(background_mask_3d) / C * bg_feat_loss_weights * 0.1
            # losses['empty_distill_occ_loss'] = (self.feat_criterion(student_occ, teacher_occ) * empty_mask_3d.unsqueeze(-1).repeat(1,1,1,1,C)).sum() / torch.sum(empty_mask_3d) / C * empty_feat_loss_weights * 0.1

            # student_occ_logits = student_occ.softmax(dim=-1)
            # teacher_occ_logits = teacher_occ.softmax(dim=-1)

            # kl_diversity = (teacher_occ_logits * torch.log(teacher_occ_logits) - teacher_occ_logits * torch.log(student_occ_logits)).sum(dim=-1)
            # cross_entropy = (- teacher_occ_logits * F.log_softmax(student_occ, dim=-1)).sum(dim=-1)

            # losses['distill_cross_entropy_loss'] = cross_entropy.mean()
            # print(cross_entropy.mean())

        else:
            p = 1
            if p == 1:
                distill_loss = F.l1_loss(student_feat, teacher_feat) 
            elif p == 2:
                distill_loss = F.mse_loss(student_feat, teacher_feat)
            
            losses['distill_loss'] = distill_loss
         
        return losses

    def forward_occ_train(self, img_feats, voxel_semantics, mask_camera, img_metas):
        """
        Args:
            img_feats: (B, C, Dz, Dy, Dx) / (B, C, Dy, Dx)
            voxel_semantics: (B, Dx, Dy, Dz)
            mask_camera: (B, Dx, Dy, Dz)
        Returns:
        """
        sample_with_label = torch.from_numpy(np.array([img_meta['use_with_label'] for img_meta in img_metas])).to(img_feats.device)
        outs = self.occ_head(img_feats)

        if torch.sum(sample_with_label) == 0:
            return outs, {'loss_occ': torch.tensor(0., requires_grad=True).to(img_feats.device)}
        
        else:
        # assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
            loss_occ = self.occ_head.loss(
                outs[sample_with_label],  # (B, Dx, Dy, Dz, n_cls)
                voxel_semantics[sample_with_label],  # (B, Dx, Dy, Dz)
                mask_camera[sample_with_label],  # (B, Dx, Dy, Dz)
            )
            return outs, loss_occ

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

        # visualC = 32
        # lidarC = 512

        # student_visual_bev_feat = img_feats[0][:, :visualC, ...]
        # student_lidar_bev_feat = img_feats[0][:, visualC:visualC+lidarC, ...]

        # occ_bev_feature = self.occ_fuser(student_visual_bev_feat, student_lidar_bev_feat)
        
        # occ_bev_feature = self.pts_backbone(occ_bev_feature)
        # if self.with_pts_neck:
        #     occ_bev_feature = self.pts_neck(occ_bev_feature)
        
        # occ_bev_feature = occ_bev_feature[0]

        occ_bev_feature = img_feats[0]
        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)

        occ_list = self.simple_test_occ(occ_bev_feature, img_metas)    # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        return occ_list

    # def simple_test(self,  # this function is for testing teacher
    #                 points,
    #                 img_metas,
    #                 img=None,
    #                 rescale=False,
    #                 **kwargs):

    #     return self.teacher_model.simple_test(points,
    #                 img_metas,
    #                 img=img,
    #                 rescale=rescale,
    #                 **kwargs)

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

    @force_fp32(apply_to=('teacher_feat', 'student_feat', 'teacher_preds', 'student_preds', 'heatmaps'))
    def distill_loss(self, teacher_feat, student_feat, teacher_preds, student_preds,
                     heatmaps, anno_boxes, inds, masks, gt_bboxes_3d, gt_labels_3d,
                     canvas_feat, index):
        # FIXME for now, only support tensor distillation.
        # TODO: add list distillation in the future
        # for input of size (256.704)
        # typically bevdet feature is of size (256,128,128)
        # centerpoint feature is of size (384,128,128)
        assert isinstance(teacher_feat, torch.Tensor) and isinstance(student_feat, torch.Tensor)
        # ensure each pixel on teacher feature map and student feature map have the same field-of-view
        if self.distill_type == 'all':
            losses_distill = self.all_distill_loss(teacher_feat, student_feat, index)
        elif self.distill_type == 'foreground_background':
            losses_distill = self.foreground_background_distill_loss(teacher_feat, student_feat, gt_bboxes_3d, index)
        elif self.distill_type == 's2m2_ssd_heatmap':
            tp_mask, fp_mask, fn_mask = self.s2m2_ssd_mask(heatmaps, student_preds)
            losses_distill = self.s2m2_ssd_heatmap_distill_loss(teacher_preds, student_preds, tp_mask, fp_mask, fn_mask)
        # elif self.distill_type == 's2m2_ssd_reg':
        #     tp_mask, fp_mask, fn_mask = self.s2m2_ssd_mask(heatmaps, student_preds)
        elif self.distill_type == 's2m2_ssd_feature':
            tp_mask, fp_mask, fn_mask = self.s2m2_ssd_mask(heatmaps, student_preds)
            losses_distill = self.s2m2_ssd_feature_distill_loss(teacher_feat, student_feat, tp_mask, fp_mask, fn_mask, index)
        elif self.distill_type == 'gauss_focal_heatmap':
            losses_distill = self.gauss_focal_heatmap_distill_loss(teacher_preds, student_preds, heatmaps)
        elif self.distill_type == 'non_local':
            losses_distill = self.non_local_distill_loss(teacher_feat, student_feat, index)
        elif self.distill_type == 'affinity':
            losses_distill = self.affinity_distill_loss(teacher_feat, student_feat, index)
        elif self.distill_type == 'linfengzhang':
            losses_distill = self.linfengzhang_distill_loss(teacher_feat, student_feat, index)
        elif self.distill_type == 'fgd':
            losses_distill = self.fgd_distill_loss(teacher_feat, student_feat,
                                                   gt_bboxes_3d, gt_labels_3d,
                                                   canvas_feat,
                                                   heatmaps, teacher_preds, student_preds, index)
        else:
            raise NotImplementedError

        return losses_distill