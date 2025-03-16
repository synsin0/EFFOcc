import torch
from .bevdet import BEVDet
from mmdet.models import DETECTORS
from mmdet3d.models.utils import FFN
from mmdet3d.models.utils.spconv_voxelize import SPConvVoxelization
from mmcv.runner import force_fp32
from torch import nn
import torch.nn.functional as F

def generate_forward_transformation_matrix(bda, img_meta_dict=None):
    b = bda.size(0)
    hom_res = torch.eye(4)[None].repeat(b, 1, 1).to(bda.device)
    for i in range(b):
        hom_res[i, :3, :3] = bda[i][:3,:3]
    return hom_res


@DETECTORS.register_module()
class StreamDAL(BEVDet):
    def __init__(self, 
            do_history=True,
            interpolation_mode='bilinear',
            history_cat_num=1, # Number of history key frames to cat
            history_cat_conv_out_channels=None,
            history_cat_conv_out_channels_lidar=None,
            single_bev_num_channels = 64,
            single_lidar_num_channels = 512,
            ### Stereo?
            do_history_stereo_fusion=False,
            stereo_neck=None,
            history_stereo_prev_step=1,
            align_prev_bev=True,
        **kwargs):
        super(StreamDAL, self).__init__(**kwargs)

        self.align_prev_bev = align_prev_bev
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


        self.single_bev_num_channels = single_bev_num_channels

        #### Deal with history
        self.do_history = do_history
        if self.do_history:
            self.interpolation_mode = interpolation_mode

            self.history_cat_num = history_cat_num
            self.history_cam_sweep_freq = 0.5 # seconds between each frame
            history_cat_conv_out_channels = (history_cat_conv_out_channels 
                                            if history_cat_conv_out_channels is not None 
                                            else self.single_bev_num_channels)
            # Embed each sample with its relative temporal offset with current timestep
            self.history_keyframe_time_conv = nn.Sequential(
                nn.Conv2d(self.single_bev_num_channels + 1,
                        self.single_bev_num_channels,
                        kernel_size=1,
                        padding=0,
                        stride=1),
                nn.BatchNorm2d(self.single_bev_num_channels),
                nn.ReLU(inplace=True))

            # Then concatenate and send them through an MLP.
            self.history_keyframe_cat_conv = nn.Sequential(
                nn.Conv2d(self.single_bev_num_channels * (self.history_cat_num + 1),
                        history_cat_conv_out_channels,
                        kernel_size=1,
                        padding=0,
                        stride=1),
                nn.BatchNorm2d(history_cat_conv_out_channels),
                nn.ReLU(inplace=True))

            self.history_sweep_time = None

            self.history_bev = None
            self.history_seq_ids = None
            self.history_forward_augs = None
       


        self.single_lidar_num_channels = single_lidar_num_channels

        #### Deal with history
        self.do_history = do_history

        if self.do_history:
            self.interpolation_mode = interpolation_mode

            self.history_cat_num = history_cat_num
            self.history_lidar_sweep_freq = 0.5 # seconds between each frame
            history_cat_conv_out_channels_lidar = (history_cat_conv_out_channels_lidar 
                                            if history_cat_conv_out_channels_lidar is not None 
                                            else single_lidar_num_channels)
            # Embed each sample with its relative temporal offset with current timestep
            self.history_keyframe_time_conv_lidar = nn.Sequential(
                nn.Conv2d(single_lidar_num_channels + 1,
                        single_lidar_num_channels,
                        kernel_size=1,
                        padding=0,
                        stride=1),
                nn.BatchNorm2d(single_lidar_num_channels),
                nn.ReLU(inplace=True))

            # Then concatenate and send them through an MLP.
            self.history_keyframe_cat_conv_lidar = nn.Sequential(
                nn.Conv2d(single_lidar_num_channels * (self.history_cat_num + 1),
                        history_cat_conv_out_channels_lidar,
                        kernel_size=1,
                        padding=0,
                        stride=1),
                nn.BatchNorm2d(history_cat_conv_out_channels_lidar),
                nn.ReLU(inplace=True))

            self.history_sweep_time_lidar = None

            self.history_bev_lidar = None
            self.history_seq_ids_lidar = None
            self.history_forward_augs_lidar = None
        
        self.test_mode = True



    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        img = self.prepare_inputs(img)
        x, _  = self.image_encoder(img[0])
        return [x] + img[1:]

    def extract_feat(self, points, img, img_metas):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas)
        pts_feats = self.extract_pts_feat(points, img_feats, img_metas)
        


        return (img_feats, pts_feats)

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

        if self.test_mode == True:
            self.history_sweep_time_lidar = None

            self.history_bev_lidar = None
            self.history_seq_ids_lidar = None
            self.history_forward_augs_lidar = None

            self.history_sweep_time = None

            self.history_bev = None
            self.history_seq_ids = None
            self.history_forward_augs = None
            self.test_mode = False

        img_feats, pts_feats = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas)
        img_feats_bev = \
            self.img_view_transformer(img_feats + img_inputs[1:7],
                                      depth_from_lidar=kwargs['gt_depth'])

        # Fuse History
        img_feats_bev = list(img_feats_bev)
        pts_feats = list(pts_feats)
        
        if self.do_history:
            img_feats_bev[0] = self.fuse_history(img_feats_bev[0], img_metas, img_inputs[6])
            pts_feats[0] = self.fuse_history_lidar(pts_feats[0], img_metas, img_inputs[6])

        losses = dict()
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

    def simple_test(self,
                    points,
                    img_metas,
                    img_inputs=None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentaiton."""

        if self.test_mode == False:
            self.history_sweep_time_lidar = None

            self.history_bev_lidar = None
            self.history_seq_ids_lidar = None
            self.history_forward_augs_lidar = None

            self.history_sweep_time = None

            self.history_bev = None
            self.history_seq_ids = None
            self.history_forward_augs = None
            self.test_mode = True


        img_feats, pts_feats = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas)
        img_feats_bev = \
            self.img_view_transformer(img_feats + img_inputs[1:7],
                                      depth_from_lidar=kwargs['gt_depth'][0])
        

        img_feats_bev = list(img_feats_bev)
        pts_feats = list(pts_feats)
        
        if self.do_history:
            img_feats_bev[0] = self.fuse_history(img_feats_bev[0], img_metas, img_inputs[6])
            pts_feats[0] = self.fuse_history_lidar(pts_feats[0], img_metas, img_inputs[6])


        bbox_list = [dict() for _ in range(len(img_metas))]
        bbox_pts = self.simple_test_pts([img_feats, pts_feats, img_feats_bev],
                                        img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list


    @force_fp32()
    def fuse_history(self, curr_bev, img_metas, bda): # align features with 3d shift
        if curr_bev is None: return None
        voxel_feat = True  if len(curr_bev.shape) == 5 else False
        if voxel_feat:
            curr_bev = curr_bev.permute(0, 1, 4, 2, 3) # n, c, z, h, w
        
        seq_ids = torch.LongTensor([
            single_img_metas['sequence_group_idx'] 
            for single_img_metas in img_metas]).to(curr_bev.device)
        start_of_sequence = torch.BoolTensor([
            single_img_metas['start_of_sequence'] 
            for single_img_metas in img_metas]).to(curr_bev.device)
        forward_augs = generate_forward_transformation_matrix(bda)
      
        # print('sqe_ids', seq_ids, ' start_of_sequence ', start_of_sequence.tolist(), ' index ', img_metas[0]['index'], img_metas[0]['scene_name'])

        curr_to_prev_ego_rt = torch.stack([
            single_img_metas['curr_to_prev_ego_rt']
            for single_img_metas in img_metas]).to(curr_bev)

        if not self.align_prev_bev:
            curr_to_prev_ego_rt = torch.eye(4).repeat(curr_to_prev_ego_rt.size(0), 1, 1).to(curr_bev)

        ## Deal with first batch
        if self.history_bev is None:
            self.history_bev = curr_bev.clone()
            self.history_seq_ids = seq_ids.clone()
            self.history_forward_augs = forward_augs.clone()

            # Repeat the first frame feature to be history
            if voxel_feat:
                self.history_bev = curr_bev.repeat(1, self.history_cat_num, 1, 1, 1) 
            else:
                self.history_bev = curr_bev.repeat(1, self.history_cat_num, 1, 1)
            # All 0s, representing current timestep.
            self.history_sweep_time = curr_bev.new_zeros(curr_bev.shape[0], self.history_cat_num)


        self.history_bev = self.history_bev.detach()

        assert self.history_bev.dtype == torch.float32

        ## Deal with the new sequences
        # First, sanity check. For every non-start of sequence, history id and seq id should be same.

        assert (self.history_seq_ids != seq_ids)[~start_of_sequence].sum() == 0, \
                "{}, {}, {}".format(self.history_seq_ids, seq_ids, start_of_sequence)

        ## Replace all the new sequences' positions in history with the curr_bev information
        self.history_sweep_time += 1 # new timestep, everything in history gets pushed back one.
        if start_of_sequence.sum()>0:
            if voxel_feat:    
                self.history_bev[start_of_sequence] = curr_bev[start_of_sequence].repeat(1, self.history_cat_num, 1, 1, 1)
            else:
                self.history_bev[start_of_sequence] = curr_bev[start_of_sequence].repeat(1, self.history_cat_num, 1, 1)
            
            self.history_sweep_time[start_of_sequence] = 0 # zero the new sequence timestep starts
            self.history_seq_ids[start_of_sequence] = seq_ids[start_of_sequence]
            self.history_forward_augs[start_of_sequence] = forward_augs[start_of_sequence]

        ## Get grid idxs & grid2bev first.
        if voxel_feat:
            n, c_, z, h, w = curr_bev.shape
        else:
            n, c_, h, w = curr_bev.shape
            z = 1

        # Generate grid
        xs = torch.linspace(0, w - 1, w, dtype=curr_bev.dtype, device=curr_bev.device).view(1, w, 1).expand(h, w, z)
        ys = torch.linspace(0, h - 1, h, dtype=curr_bev.dtype, device=curr_bev.device).view(h, 1, 1).expand(h, w, z)
        zs = torch.linspace(0, z - 1, z, dtype=curr_bev.dtype, device=curr_bev.device).view(1, 1, z).expand(h, w, z)
        grid = torch.stack(
            (xs, ys,  zs, torch.ones_like(xs)), -1).view(1, h, w, z, 4).expand(n, h, w, z, 4).view(n, h, w, z, 4, 1)

        # This converts BEV indices to meters
        # IMPORTANT: the feat2bev[0, 3] is changed from feat2bev[0, 2] because previous was 2D rotation
        # which has 2-th index as the hom index. Now, with 3D hom, 3-th is hom
        feat2bev = torch.zeros((4,4),dtype=grid.dtype).to(grid)
        feat2bev[0, 0] = self.img_view_transformer.dx[0]
        feat2bev[1, 1] = self.img_view_transformer.dx[1]
        feat2bev[2, 2] = self.img_view_transformer.dx[2]
        feat2bev[0, 3] = self.img_view_transformer.bx[0] - self.img_view_transformer.dx[0] / 2.
        feat2bev[1, 3] = self.img_view_transformer.bx[1] - self.img_view_transformer.dx[1] / 2.
        feat2bev[2, 3] = self.img_view_transformer.bx[2] - self.img_view_transformer.dx[2] / 2.
        feat2bev[3, 3] = 1
        feat2bev = feat2bev.view(1,4,4)
        
        ## Get flow for grid sampling.
        # The flow is as follows. Starting from grid locations in curr bev, transform to BEV XY11,
        # backward of current augmentations, curr lidar to prev lidar, forward of previous augmentations,
        # transform to previous grid locations.
        rt_flow = (torch.inverse(feat2bev) @ self.history_forward_augs @ curr_to_prev_ego_rt
                   @ torch.inverse(forward_augs) @ feat2bev)
        grid = rt_flow.view(n, 1, 1, 1, 4, 4) @ grid
        

        # normalize and sample
        if voxel_feat:
            normalize_factor = torch.tensor([w - 1.0, h - 1.0, z - 1.0], dtype=curr_bev.dtype, device=curr_bev.device)
            grid = grid[:,:,:,:, :3,0] / normalize_factor.view(1, 1, 1, 1, 3) * 2.0 - 1.0
        else:
            normalize_factor = torch.tensor([w - 1.0, h - 1.0], dtype=curr_bev.dtype, device=curr_bev.device)
            grid = grid[:,:,:,:, :2,0] / normalize_factor.view(1, 1, 1, 1, 2) * 2.0 - 1.0           

        tmp_bev = self.history_bev
        if voxel_feat: 
            n, mc, z, h, w = tmp_bev.shape
            tmp_bev = tmp_bev.reshape(n, mc, z, h, w)
            grid = grid.to(curr_bev.dtype).permute(0, 3, 1, 2, 4)
        else:
            grid = grid.to(curr_bev.dtype).squeeze(-2)

        # save_tensor(tmp_bev[0].clamp(min=-1, max=1).reshape(4, 80, 128, 128).abs().mean(1), f'curr_{self.count}_pre.png')
        sampled_history_bev = F.grid_sample(tmp_bev, grid, align_corners=True, mode=self.interpolation_mode)
        # save_tensor(sampled_history_bev[0].clamp(min=-1, max=1).reshape(4, 80, 128, 128).abs().mean(1), f'curr_{self.count}_after.png')
        # save_tensor(curr_bev.clamp(min=-1, max=1).abs().mean(1), f'curr_{self.count}.png')
        # self.count += 1
        # if self.count == 10:

        ## Update history
        # Add in current frame to features & timestep
        self.history_sweep_time = torch.cat(
            [self.history_sweep_time.new_zeros(self.history_sweep_time.shape[0], 1), self.history_sweep_time],
            dim=1) # B x (1 + T)

        if voxel_feat:
            sampled_history_bev = sampled_history_bev.reshape(n, mc, z, h, w)
            curr_bev = curr_bev.reshape(n, c_, z, h, w)
        feats_cat = torch.cat([curr_bev, sampled_history_bev], dim=1) # B x (1 + T) * 80 x H x W or B x (1 + T) * 80 xZ x H x W 

        # Reshape and concatenate features and timestep
        feats_to_return = feats_cat.reshape(
                feats_cat.shape[0], self.history_cat_num + 1, self.single_bev_num_channels, *feats_cat.shape[2:]) # B x (1 + T) x 80 x H x W
        if voxel_feat:
            feats_to_return = torch.cat(
            [feats_to_return, self.history_sweep_time[:, :, None, None, None, None].repeat(
                1, 1, 1, *feats_to_return.shape[3:]) * self.history_cam_sweep_freq
            ], dim=2) # B x (1 + T) x 81 x Z x H x W
        else:
            feats_to_return = torch.cat(
            [feats_to_return, self.history_sweep_time[:, :, None, None, None].repeat(
                1, 1, 1, feats_to_return.shape[3], feats_to_return.shape[4]) * self.history_cam_sweep_freq
            ], dim=2) # B x (1 + T) x 81 x H x W

        # Time conv
        feats_to_return = self.history_keyframe_time_conv(
            feats_to_return.reshape(-1, *feats_to_return.shape[2:])).reshape(
                feats_to_return.shape[0], feats_to_return.shape[1], -1, *feats_to_return.shape[3:]) # B x (1 + T) x 80 xZ x H x W

        # Cat keyframes & conv
        feats_to_return = self.history_keyframe_cat_conv(
            feats_to_return.reshape(
                feats_to_return.shape[0], -1, *feats_to_return.shape[3:])) # B x C x H x W or B x C x Z x H x W
        
        self.history_bev = feats_cat[:, :-self.single_bev_num_channels, ...].detach().clone()
        self.history_sweep_time = self.history_sweep_time[:, :-1]
        self.history_forward_augs = forward_augs.clone()
        if voxel_feat:
            feats_to_return = feats_to_return.permute(0, 1, 3, 4, 2)
        if not self.do_history:
            self.history_bev = None
        return feats_to_return.clone()




    @force_fp32()
    def fuse_history_lidar(self, curr_bev, img_metas, bda): # align features with 3d shift
        if curr_bev is None: return None
        voxel_feat = True  if len(curr_bev.shape) == 5 else False
        if voxel_feat:
            curr_bev = curr_bev.permute(0, 1, 4, 2, 3) # n, c, z, h, w
        
        seq_ids_lidar = torch.LongTensor([
            single_img_metas['sequence_group_idx'] 
            for single_img_metas in img_metas]).to(curr_bev.device)
        start_of_sequence_lidar = torch.BoolTensor([
            single_img_metas['start_of_sequence'] 
            for single_img_metas in img_metas]).to(curr_bev.device)
        forward_augs_lidar = generate_forward_transformation_matrix(bda)
      
        # print('sqe_ids', seq_ids, ' start_of_sequence ', start_of_sequence.tolist(), ' index ', img_metas[0]['index'], img_metas[0]['scene_name'])

        curr_to_prev_ego_rt = torch.stack([
            single_img_metas['curr_to_prev_ego_rt']
            for single_img_metas in img_metas]).to(curr_bev)

        if not self.align_prev_bev:
            curr_to_prev_ego_rt = torch.eye(4).repeat(curr_to_prev_ego_rt.size(0), 1, 1).to(curr_bev)

        ## Deal with first batch
        if self.history_bev_lidar is None:
            self.history_bev_lidar = curr_bev.clone()
            self.history_seq_ids_lidar = seq_ids_lidar.clone()
            self.history_forward_augs_lidar = forward_augs_lidar.clone()

            # Repeat the first frame feature to be history
            if voxel_feat:
                self.history_bev_lidar = curr_bev.repeat(1, self.history_cat_num, 1, 1, 1) 
            else:
                self.history_bev_lidar = curr_bev.repeat(1, self.history_cat_num, 1, 1)
            # All 0s, representing current timestep.
            self.history_sweep_time_lidar = curr_bev.new_zeros(curr_bev.shape[0], self.history_cat_num)


        self.history_bev_lidar = self.history_bev_lidar.detach()

        assert self.history_bev_lidar.dtype == torch.float32

        ## Deal with the new sequences
        # First, sanity check. For every non-start of sequence, history id and seq id should be same.

        assert (self.history_seq_ids_lidar != seq_ids_lidar)[~start_of_sequence_lidar].sum() == 0, \
                "{}, {}, {}".format(self.history_seq_ids_lidar, seq_ids_lidar, start_of_sequence_lidar)

        ## Replace all the new sequences' positions in history with the curr_bev information
        self.history_sweep_time_lidar += 1 # new timestep, everything in history gets pushed back one.
        if start_of_sequence_lidar.sum()>0:
            if voxel_feat:    
                self.history_bev_lidar[start_of_sequence_lidar] = curr_bev[start_of_sequence_lidar].repeat(1, self.history_cat_num, 1, 1, 1)
            else:
                self.history_bev_lidar[start_of_sequence_lidar] = curr_bev[start_of_sequence_lidar].repeat(1, self.history_cat_num, 1, 1)
            
            self.history_sweep_time_lidar[start_of_sequence_lidar] = 0 # zero the new sequence timestep starts
            self.history_seq_ids_lidar[start_of_sequence_lidar] = seq_ids_lidar[start_of_sequence_lidar]
            self.history_forward_augs_lidar[start_of_sequence_lidar] = forward_augs_lidar[start_of_sequence_lidar]

        ## Get grid idxs & grid2bev first.
        if voxel_feat:
            n, c_, z, h, w = curr_bev.shape
        else:
            n, c_, h, w = curr_bev.shape
            z = 1

        # Generate grid
        xs = torch.linspace(0, w - 1, w, dtype=curr_bev.dtype, device=curr_bev.device).view(1, w, 1).expand(h, w, z)
        ys = torch.linspace(0, h - 1, h, dtype=curr_bev.dtype, device=curr_bev.device).view(h, 1, 1).expand(h, w, z)
        zs = torch.linspace(0, z - 1, z, dtype=curr_bev.dtype, device=curr_bev.device).view(1, 1, z).expand(h, w, z)
        grid = torch.stack(
            (xs, ys,  zs, torch.ones_like(xs)), -1).view(1, h, w, z, 4).expand(n, h, w, z, 4).view(n, h, w, z, 4, 1)

        # This converts BEV indices to meters
        # IMPORTANT: the feat2bev[0, 3] is changed from feat2bev[0, 2] because previous was 2D rotation
        # which has 2-th index as the hom index. Now, with 3D hom, 3-th is hom
        feat2bev = torch.zeros((4,4),dtype=grid.dtype).to(grid)
        feat2bev[0, 0] = self.img_view_transformer.dx[0]
        feat2bev[1, 1] = self.img_view_transformer.dx[1]
        feat2bev[2, 2] = self.img_view_transformer.dx[2]
        feat2bev[0, 3] = self.img_view_transformer.bx[0] - self.img_view_transformer.dx[0] / 2.
        feat2bev[1, 3] = self.img_view_transformer.bx[1] - self.img_view_transformer.dx[1] / 2.
        feat2bev[2, 3] = self.img_view_transformer.bx[2] - self.img_view_transformer.dx[2] / 2.
        feat2bev[3, 3] = 1
        feat2bev = feat2bev.view(1,4,4)
        
        ## Get flow for grid sampling.
        # The flow is as follows. Starting from grid locations in curr bev, transform to BEV XY11,
        # backward of current augmentations, curr lidar to prev lidar, forward of previous augmentations,
        # transform to previous grid locations.
        rt_flow = (torch.inverse(feat2bev) @ self.history_forward_augs_lidar @ curr_to_prev_ego_rt
                   @ torch.inverse(forward_augs_lidar) @ feat2bev)
        grid = rt_flow.view(n, 1, 1, 1, 4, 4) @ grid
        

        # normalize and sample
        if voxel_feat:
            normalize_factor = torch.tensor([w - 1.0, h - 1.0, z - 1.0], dtype=curr_bev.dtype, device=curr_bev.device)
            grid = grid[:,:,:,:, :3,0] / normalize_factor.view(1, 1, 1, 1, 3) * 2.0 - 1.0
        else:
            normalize_factor = torch.tensor([w - 1.0, h - 1.0], dtype=curr_bev.dtype, device=curr_bev.device)
            grid = grid[:,:,:,:, :2,0] / normalize_factor.view(1, 1, 1, 1, 2) * 2.0 - 1.0           

        tmp_bev = self.history_bev_lidar
        if voxel_feat: 
            n, mc, z, h, w = tmp_bev.shape
            tmp_bev = tmp_bev.reshape(n, mc, z, h, w)
            grid = grid.to(curr_bev.dtype).permute(0, 3, 1, 2, 4)
        else:
            grid = grid.to(curr_bev.dtype).squeeze(-2)

        # save_tensor(tmp_bev[0].clamp(min=-1, max=1).reshape(4, 80, 128, 128).abs().mean(1), f'curr_{self.count}_pre.png')
        sampled_history_bev_lidar = F.grid_sample(tmp_bev, grid, align_corners=True, mode=self.interpolation_mode)
        # save_tensor(sampled_history_bev_lidar[0].clamp(min=-1, max=1).reshape(4, 80, 128, 128).abs().mean(1), f'curr_{self.count}_after.png')
        # save_tensor(curr_bev.clamp(min=-1, max=1).abs().mean(1), f'curr_{self.count}.png')
        # self.count += 1
        # if self.count == 10:

        ## Update history
        # Add in current frame to features & timestep
        self.history_sweep_time_lidar = torch.cat(
            [self.history_sweep_time_lidar.new_zeros(self.history_sweep_time_lidar.shape[0], 1), self.history_sweep_time_lidar],
            dim=1) # B x (1 + T)

        if voxel_feat:
            sampled_history_bev_lidar = sampled_history_bev_lidar.reshape(n, mc, z, h, w)
            curr_bev = curr_bev.reshape(n, c_, z, h, w)
        feats_cat = torch.cat([curr_bev, sampled_history_bev_lidar], dim=1) # B x (1 + T) * 80 x H x W or B x (1 + T) * 80 xZ x H x W 

        # Reshape and concatenate features and timestep
        feats_to_return = feats_cat.reshape(
                feats_cat.shape[0], self.history_cat_num + 1, self.single_lidar_num_channels, *feats_cat.shape[2:]) # B x (1 + T) x 80 x H x W
        if voxel_feat:
            feats_to_return = torch.cat(
            [feats_to_return, self.history_sweep_time_lidar[:, :, None, None, None, None].repeat(
                1, 1, 1, *feats_to_return.shape[3:]) * self.history_lidar_sweep_freq
            ], dim=2) # B x (1 + T) x 81 x Z x H x W
        else:
            feats_to_return = torch.cat(
            [feats_to_return, self.history_sweep_time_lidar[:, :, None, None, None].repeat(
                1, 1, 1, feats_to_return.shape[3], feats_to_return.shape[4]) * self.history_lidar_sweep_freq
            ], dim=2) # B x (1 + T) x 81 x H x W

        # Time conv
        feats_to_return = self.history_keyframe_time_conv_lidar(
            feats_to_return.reshape(-1, *feats_to_return.shape[2:])).reshape(
                feats_to_return.shape[0], feats_to_return.shape[1], -1, *feats_to_return.shape[3:]) # B x (1 + T) x 80 xZ x H x W

        # Cat keyframes & conv
        feats_to_return = self.history_keyframe_cat_conv_lidar(
            feats_to_return.reshape(
                feats_to_return.shape[0], -1, *feats_to_return.shape[3:])) # B x C x H x W or B x C x Z x H x W
        
        self.history_bev_lidar = feats_cat[:, :-self.single_lidar_num_channels, ...].detach().clone()
        self.history_sweep_time_lidar = self.history_sweep_time_lidar[:, :-1]
        self.history_forward_augs_lidar = forward_augs_lidar.clone()
        if voxel_feat:
            feats_to_return = feats_to_return.permute(0, 1, 3, 4, 2)
        if not self.do_history:
            self.history_bev_lidar = None
        return feats_to_return.clone()