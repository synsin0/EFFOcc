_base_ = [
    '../_base_/datasets/nus-3d.py',
    '../_base_/schedules/cyclic_20e.py',
    '../_base_/default_runtime.py'
]
'''

'''
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -3.0, 51.2, 51.2, 5.0]
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']

data_config = {
    'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
    'Ncams': 5,
    'input_size': (192, 544),
    'src_size': (900, 1600),

    # Augmentation
    'resize': (-0.06, 0.44),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'random_crop_height': True,
    'vflip':True,
    'resize_test': 0.04,

    'pmd': dict(
        brightness_delta=32,
        contrast_lower=0.5,
        contrast_upper=1.5,
        saturation_lower=0.5,
        saturation_upper=1.5,
        hue_delta=18,
        rate=0.5
    )
}

grid_config = {
    'x': [-51.2, 51.2, 0.8],
    'y': [-51.2, 51.2, 0.8],
    'z': [-3, 5, 8],
    'depth': [1.0, 60.0, 0.5],
}

# Model
voxel_size = [0.1, 0.1, 0.2]

feat_bev_img_dim = 32
img_feat_dim = 128
model = dict(
    type='DAL',
    use_grid_mask=True,
    # camera
    img_backbone=dict(
        pretrained='torchvision://resnet18',
        # pretrained="/mnt/cfs/algorithm/junjie.huang/models/resnet18-f37072fd.pth",
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style='pytorch'),
    img_neck=dict(
        type='CustomFPN',
        in_channels=[128, 256, 512],
        out_channels=img_feat_dim,
        num_outs=1,
        start_level=0,
        out_ids=[0]),
    img_view_transformer=dict(
        type='LSSViewTransformer',
        grid_config=grid_config,
        input_size=data_config['input_size'],
        in_channels=img_feat_dim,
        out_channels=feat_bev_img_dim,
        downsample=8,
        with_depth_from_lidar=True),
    # lidar
    pts_voxel_layer=dict(
        max_num_points=10, voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        max_voxels=(90000, 120000)),
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=5),
    pts_middle_encoder=dict(
        type='SparseEncoder',
        in_channels=5,
        sparse_shape=[41, 1024, 1024],
        output_channels=128,
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 32),
                          (32, 32, 64),
                          (64, 64, 128),
                          (128, 128)),
        encoder_paddings=((0, 0, 1),
                          (0, 0, 1),
                          (0, 0, [0, 1, 1]),
                          (0, 0)),
        block_type='basicblock'),
    pts_backbone=dict(
        type='SECOND',
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),

    # head
    pts_bbox_head=dict(
        type='DALHead',

        # DAL
        feat_bev_img_dim=feat_bev_img_dim,
        img_feat_dim=img_feat_dim,
        sparse_fuse_layers=2,
        dense_fuse_layers=2,
        instance_attn=False,

        # Transfusion
        num_proposals=200,
        in_channels=512,
        hidden_channel=128,
        num_classes=10,
        num_decoder_layers=1,
        num_heads=8,
        nms_kernel_size=3,
        ffn_channel=256,
        dropout=0.1,
        bn_momentum=0.1,
        activation='relu',
        auxiliary=True,
        common_heads=dict(
            center=[2, 2],
            height=[1, 2],
            dim=[3, 2],
            rot=[2, 2],
            vel=[2, 2]),
        bbox_coder=dict(
            type='TransFusionBBoxCoder',
            pc_range=point_cloud_range[:2],
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            score_threshold=0.0,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            code_size=10),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            reduction='mean',
            loss_weight=1.0),
        loss_heatmap=dict(
            type='GaussianFocalLoss', reduction='mean', loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25)),
    train_cfg=dict(pts=dict(
        dataset='nuScenes',
        point_cloud_range=point_cloud_range,
        grid_size=[1024, 1024, 40],
        voxel_size=voxel_size,
        out_size_factor=8,
        gaussian_overlap=0.1,
        min_radius=2,
        pos_weight=-1,
        code_weights=[
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
        ],
        assigner=dict(
            type='HungarianAssigner3D',
            iou_calculator=dict(
                type='BboxOverlaps3D', coordinate='lidar'),
            cls_cost=dict(
                type='FocalLossCost',
                gamma=2.0,
                alpha=0.25,
                weight=0.15),
            reg_cost=dict(type='BBoxBEVL1Cost', weight=0.25),
            iou_cost=dict(type='IoU3DCost', weight=0.25)))),
    test_cfg=dict(pts=dict(
        dataset='nuScenes',
        grid_size=[1024, 1024, 40],
        img_feat_downsample=8,
        out_size_factor=8,
        voxel_size=voxel_size[:2],
        pc_range=point_cloud_range[:2],
        nms_type=None)),
)

dataset_type = 'NuScenesDatasetLiDARCoord'
data_root = 'data/nuscenes/'

file_client_args = dict(backend='disk')

bda_aug_conf = dict(
    rot_lim=(-22.5 * 2, 22.5 * 2),
    scale_lim=(0.9, 1.1),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5,
    tran_lim=[0.5, 0.5, 0.5]
)

db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'bevdetv3-nuscenes_dbinfos_train.pkl',
    # info_path= data_root + 'nuscenes_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(
            car=5,
            truck=5,
            bus=5,
            trailer=5,
            construction_vehicle=5,
            traffic_cone=5,
            barrier=5,
            motorcycle=5,
            bicycle=5,
            pedestrian=5)),
    classes=class_names,
    sample_groups=dict(
        car=2,
        truck=3,
        construction_vehicle=7,
        bus=4,
        trailer=6,
        barrier=2,
        motorcycle=6,
        bicycle=6,
        pedestrian=2,
        traffic_cone=2),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args))

train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True, opencv_pp=True, use_lidar_coord=True,
        data_config=data_config),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True),
    dict(type='PipelinePlaceHolder'),
    # dict(type='LoadAnnotations'),
    dict(type='LoadAnnotations3D'),
    dict(type='ObjectSample', db_sampler=db_sampler),
    dict(type='VelocityAug'),
    dict(
        type='BEVAug',
        bda_aug_conf=bda_aug_conf,
        classes=class_names),
    dict(type='PointToMultiViewDepthFusion', downsample=1,
         grid_config=grid_config,  use_lidar_coord=True,),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d',
                                 'img_inputs', 'gt_depth',
                                 'gt_bboxes_ignore'
                                 ])
]

test_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=False, opencv_pp=True, use_lidar_coord=True,
        data_config=data_config),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True),
    dict(type='PipelinePlaceHolder'),
    # dict(type='LoadAnnotations'),
    dict(type='LoadAnnotations3D'),
    dict(type='BEVAug',
         bda_aug_conf=bda_aug_conf,
         classes=class_names,
         is_train=False),
    dict(type='PointToMultiViewDepthFusion', downsample=1,
         grid_config=grid_config, use_lidar_coord=True,),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img_inputs', 'gt_depth'])
        ])
]

input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

data = dict(
    samples_per_gpu=16,  # for 16 GPU
    workers_per_gpu=8,
    train=dict(
        type='CBGSDataset',
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'bevdetv3-nuscenes_infos_train.pkl',
            pipeline=train_pipeline,
            classes=class_names,
            test_mode=False,
            use_valid_flag=True,
            modality=input_modality,
            img_info_prototype='bevdet',
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='LiDAR')),
    val=dict(type=dataset_type, pipeline=test_pipeline, classes=class_names,
             modality=input_modality,
             ann_file=data_root + 'bevdetv3-nuscenes_infos_val.pkl',
             img_info_prototype='bevdet'),
    test=dict(type=dataset_type, pipeline=test_pipeline, classes=class_names,
              modality=input_modality,
              ann_file=data_root + 'bevdetv3-nuscenes_infos_val.pkl',
              img_info_prototype='bevdet'))

evaluation = dict(interval=1, start=16, pipeline=test_pipeline)
optimizer = dict(type='AdamW', lr=2e-4, weight_decay=0.01)  # for 64 total batch size
two_stage = True
runner = dict(type='TwoStageRunner', max_epochs=20)

# {'pts_bbox_NuScenes/car_AP_dist_0.5': 0.7722, 'pts_bbox_NuScenes/car_AP_dist_1.0': 0.8771, 'pts_bbox_NuScenes/car_AP_dist_2.0': 0.9111, 'pts_bbox_NuScenes/car_AP_dist_4.0': 0.928, 'pts_bbox_NuScenes/car_trans_err': 0.177, 'pts_bbox_NuScenes/car_scale_err': 0.153, 'pts_bbox_NuScenes/car_orient_err': 0.1002, 'pts_bbox_NuScenes/car_vel_err': 0.2117, 'pts_bbox_NuScenes/car_attr_err': 0.1824, 'pts_bbox_NuScenes/mATE': 0.2941, 'pts_bbox_NuScenes/mASE': 0.2545, 'pts_bbox_NuScenes/mAOE': 0.3203, 'pts_bbox_NuScenes/mAVE': 0.1966, 'pts_bbox_NuScenes/mAAE': 0.1841, 'pts_bbox_NuScenes/truck_AP_dist_0.5': 0.4327, 'pts_bbox_NuScenes/truck_AP_dist_1.0': 0.6184, 'pts_bbox_NuScenes/truck_AP_dist_2.0': 0.7006, 'pts_bbox_NuScenes/truck_AP_dist_4.0': 0.7472, 'pts_bbox_NuScenes/truck_trans_err': 0.3191, 'pts_bbox_NuScenes/truck_scale_err': 0.1879, 'pts_bbox_NuScenes/truck_orient_err': 0.1247, 'pts_bbox_NuScenes/truck_vel_err': 0.1822, 'pts_bbox_NuScenes/truck_attr_err': 0.2071, 'pts_bbox_NuScenes/construction_vehicle_AP_dist_0.5': 0.0309, 'pts_bbox_NuScenes/construction_vehicle_AP_dist_1.0': 0.2076, 'pts_bbox_NuScenes/construction_vehicle_AP_dist_2.0': 0.4285, 'pts_bbox_NuScenes/construction_vehicle_AP_dist_4.0': 0.5679, 'pts_bbox_NuScenes/construction_vehicle_trans_err': 0.7402, 'pts_bbox_NuScenes/construction_vehicle_scale_err': 0.4376, 'pts_bbox_NuScenes/construction_vehicle_orient_err': 0.8387, 'pts_bbox_NuScenes/construction_vehicle_vel_err': 0.1238, 'pts_bbox_NuScenes/construction_vehicle_attr_err': 0.298, 'pts_bbox_NuScenes/bus_AP_dist_0.5': 0.5104, 'pts_bbox_NuScenes/bus_AP_dist_1.0': 0.7643, 'pts_bbox_NuScenes/bus_AP_dist_2.0': 0.8822, 'pts_bbox_NuScenes/bus_AP_dist_4.0': 0.9043, 'pts_bbox_NuScenes/bus_trans_err': 0.3378, 'pts_bbox_NuScenes/bus_scale_err': 0.1837, 'pts_bbox_NuScenes/bus_orient_err': 0.1081, 'pts_bbox_NuScenes/bus_vel_err': 0.3236, 'pts_bbox_NuScenes/bus_attr_err': 0.2342, 'pts_bbox_NuScenes/trailer_AP_dist_0.5': 0.1251, 'pts_bbox_NuScenes/trailer_AP_dist_1.0': 0.3997, 'pts_bbox_NuScenes/trailer_AP_dist_2.0': 0.6036, 'pts_bbox_NuScenes/trailer_AP_dist_4.0': 0.7307, 'pts_bbox_NuScenes/trailer_trans_err': 0.5143, 'pts_bbox_NuScenes/trailer_scale_err': 0.212, 'pts_bbox_NuScenes/trailer_orient_err': 0.4683, 'pts_bbox_NuScenes/trailer_vel_err': 0.1676, 'pts_bbox_NuScenes/trailer_attr_err': 0.1726, 'pts_bbox_NuScenes/barrier_AP_dist_0.5': 0.5806, 'pts_bbox_NuScenes/barrier_AP_dist_1.0': 0.6946, 'pts_bbox_NuScenes/barrier_AP_dist_2.0': 0.7412, 'pts_bbox_NuScenes/barrier_AP_dist_4.0': 0.7559, 'pts_bbox_NuScenes/barrier_trans_err': 0.2126, 'pts_bbox_NuScenes/barrier_scale_err': 0.2868, 'pts_bbox_NuScenes/barrier_orient_err': 0.0759, 'pts_bbox_NuScenes/barrier_vel_err': nan, 'pts_bbox_NuScenes/barrier_attr_err': nan, 'pts_bbox_NuScenes/motorcycle_AP_dist_0.5': 0.6199, 'pts_bbox_NuScenes/motorcycle_AP_dist_1.0': 0.7432, 'pts_bbox_NuScenes/motorcycle_AP_dist_2.0': 0.7568, 'pts_bbox_NuScenes/motorcycle_AP_dist_4.0': 0.7653, 'pts_bbox_NuScenes/motorcycle_trans_err': 0.1961, 'pts_bbox_NuScenes/motorcycle_scale_err': 0.2354, 'pts_bbox_NuScenes/motorcycle_orient_err': 0.2793, 'pts_bbox_NuScenes/motorcycle_vel_err': 0.2364, 'pts_bbox_NuScenes/motorcycle_attr_err': 0.2749, 'pts_bbox_NuScenes/bicycle_AP_dist_0.5': 0.5204, 'pts_bbox_NuScenes/bicycle_AP_dist_1.0': 0.5624, 'pts_bbox_NuScenes/bicycle_AP_dist_2.0': 0.5678, 'pts_bbox_NuScenes/bicycle_AP_dist_4.0': 0.5802, 'pts_bbox_NuScenes/bicycle_trans_err': 0.1749, 'pts_bbox_NuScenes/bicycle_scale_err': 0.2512, 'pts_bbox_NuScenes/bicycle_orient_err': 0.521, 'pts_bbox_NuScenes/bicycle_vel_err': 0.1363, 'pts_bbox_NuScenes/bicycle_attr_err': 0.0125, 'pts_bbox_NuScenes/pedestrian_AP_dist_0.5': 0.8597, 'pts_bbox_NuScenes/pedestrian_AP_dist_1.0': 0.8753, 'pts_bbox_NuScenes/pedestrian_AP_dist_2.0': 0.8855, 'pts_bbox_NuScenes/pedestrian_AP_dist_4.0': 0.8949, 'pts_bbox_NuScenes/pedestrian_trans_err': 0.1368, 'pts_bbox_NuScenes/pedestrian_scale_err': 0.272, 'pts_bbox_NuScenes/pedestrian_orient_err': 0.366, 'pts_bbox_NuScenes/pedestrian_vel_err': 0.1912, 'pts_bbox_NuScenes/pedestrian_attr_err': 0.0913, 'pts_bbox_NuScenes/traffic_cone_AP_dist_0.5': 0.7436, 'pts_bbox_NuScenes/traffic_cone_AP_dist_1.0': 0.7622, 'pts_bbox_NuScenes/traffic_cone_AP_dist_2.0': 0.78, 'pts_bbox_NuScenes/traffic_cone_AP_dist_4.0': 0.8054, 'pts_bbox_NuScenes/traffic_cone_trans_err': 0.1322, 'pts_bbox_NuScenes/traffic_cone_scale_err': 0.3254, 'pts_bbox_NuScenes/traffic_cone_orient_err': nan, 'pts_bbox_NuScenes/traffic_cone_vel_err': nan, 'pts_bbox_NuScenes/traffic_cone_attr_err': nan, 'pts_bbox_NuScenes/NDS': 0.7080107994521956, 'pts_bbox_NuScenes/mAP': 0.665937523730591}