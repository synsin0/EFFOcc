_base_ = ['../_base_/datasets/nus-3d.py',
          '../_base_/default_runtime.py']

plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

data_config = {
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams':
    6,
    'input_size': (256, 704),
    'src_size': (900, 1600),

    # Augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

grid_config = {
    'x': [-40, 40, 0.4],
    'y': [-40, 40, 0.4],
    'z': [-1, 5.4, 6.4],
    'depth': [1.0, 45.0, 0.5],
}

point_cloud_range = [-40, -40, -1, 40, 40, 5.4]

voxel_size = [0.05, 0.05, 0.16]

numC_Trans = 64

img_feat_dim = 128

model = dict(
    type='FlashFusionOCCFlow',
    img_backbone=dict(
        # pretrained='torchvision://resnet18',
        pretrained="ckpts/torchvision/resnet18-f37072fd.pth",
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
        out_channels=numC_Trans,
        sid=False,
        collapse_z=True,
        downsample=8,
        with_depth_from_lidar=True),
    # img_bev_encoder_backbone=dict(
    #     type='CustomResNet',
    #     numC_input=numC_Trans,
    #     num_channels=[numC_Trans * 2, numC_Trans * 4, numC_Trans * 8]),
    # img_bev_encoder_neck=dict(
    #     type='FPN_LSS',
    #     in_channels=numC_Trans * 8 + numC_Trans * 2,
    #     out_channels=256),

   # lidar
    pts_voxel_layer=dict(
        max_num_points=10, voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        max_voxels=(90000, 120000)),
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=5),
    pts_middle_encoder=dict(
        type='SparseEncoder',
        in_channels=5,
        sparse_shape=[41, 1600, 1600],
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
    
    occ_fuser=dict(
        type='ConvFuser2D',
        in_channels=[64, 256],
        out_channels=256,
    ),

    occ_head=dict(
        type='BEVOCCFlowHead2D',
        in_dim=512,
        out_dim=256,
        Dz=16,
        use_mask=True,
        num_classes=18,
        use_predicter=True,
        class_wise=False,
        loss_occ=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            ignore_index=255,
            loss_weight=1.0
        ),
        loss_flow=dict(type='L1Loss', loss_weight=0.25),
    )
)

# Data
dataset_type = 'NuScenesDatasetOccpancyFlow'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

bda_aug_conf = dict(
    rot_lim=(-0., 0.),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5
)

train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config=data_config,
        sequential=False),
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
    dict(type='ToEgo'), 
    # dict(
    #     type='LoadAnnotationsBEVDepth',
    #     bda_aug_conf=bda_aug_conf,
    #     classes=class_names,
    #     is_train=True),
    dict(type='LoadOccGTFromFileV2'),
    dict(type='LoadAnnotations'),
    dict(
        type='BEVAug',
        bda_aug_conf=bda_aug_conf,
        classes=class_names),

    dict(type='PointToMultiViewDepthFusion', downsample=1, grid_config=grid_config),

    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D', keys=['points', 'img_inputs', 'gt_depth', 'voxel_semantics',
                                'voxel_flow','voxel_instances'])
]

test_pipeline = [
    dict(type='PrepareImageInputs', data_config=data_config, sequential=False),
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
    dict(type='ToEgo'),
    # dict(
    #     type='LoadAnnotationsBEVDepth',
    #     bda_aug_conf=bda_aug_conf,
    #     classes=class_names,
    #     is_train=False),
    dict(type='LoadAnnotations'),
    dict(type='BEVAug',
         bda_aug_conf=bda_aug_conf,
         classes=class_names,
         is_train=False),
    dict(type='PointToMultiViewDepthFusion', downsample=1,
         grid_config=grid_config),
    # dict(
    #     type='LoadPointsFromFile',
    #     coord_type='LIDAR',
    #     load_dim=5,
    #     use_dim=5,
    #     file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
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

share_data_config = dict(
    type=dataset_type,
    data_root=data_root,
    classes=class_names,
    modality=input_modality,
    stereo=False,
    filter_empty_gt=False,
    img_info_prototype='bevdet',
)

test_data_config = dict(
    pipeline=test_pipeline,
    ann_file=data_root + 'bevdetv3-nuscenes_infos_val_occflow.pkl')

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        data_root=data_root,
        ann_file=data_root + 'bevdetv3-nuscenes_infos_train_occflow.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        test_mode=False,
        use_valid_flag=True,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=test_data_config,
    test=test_data_config)

for key in ['val', 'train', 'test']:
    data[key].update(share_data_config)

# Optimizer
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=1e-2)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[24, ])
runner = dict(type='EpochBasedRunner', max_epochs=24)

custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
    ),
]

load_from = "ckpts/dal/dal-tiny-map66.9-nds71.1.pth"
# fp16 = dict(loss_scale='dynamic')
evaluation = dict(interval=1, start=24, pipeline=test_pipeline)
checkpoint_config = dict(interval=1, max_keep_ckpts=5)


# 562
# 【2024-03-25 15:55:43】+----------------------+-------+-------+-------+-------+
# 563
# 【2024-03-25 15:55:43】| Class Names | IoU@1 | IoU@2 | IoU@4 | AVE |
# 564
# 【2024-03-25 15:55:43】+----------------------+-------+-------+-------+-------+
# 565
# 【2024-03-25 15:55:43】| car | 0.613 | 0.657 | 0.671 | 2.846 |
# 566
# 【2024-03-25 15:55:43】| truck | 0.505 | 0.549 | 0.569 | 1.769 |
# 567
# 【2024-03-25 15:55:43】| trailer | 0.468 | 0.505 | 0.533 | 1.998 |
# 568
# 【2024-03-25 15:55:43】| bus | 0.618 | 0.675 | 0.708 | 2.730 |
# 569
# 【2024-03-25 15:55:43】| construction_vehicle | 0.340 | 0.380 | 0.393 | 0.136 |
# 570
# 【2024-03-25 15:55:43】| bicycle | 0.094 | 0.095 | 0.095 | 3.831 |
# 571
# 【2024-03-25 15:55:43】| motorcycle | 0.331 | 0.339 | 0.341 | 3.013 |
# 572
# 【2024-03-25 15:55:43】| pedestrian | 0.512 | 0.525 | 0.530 | 0.925 |
# 573
# 【2024-03-25 15:55:43】| traffic_cone | 0.270 | 0.275 | 0.278 | nan |
# 574
# 【2024-03-25 15:55:43】| barrier | 0.449 | 0.470 | 0.481 | nan |
# 575
# 【2024-03-25 15:55:43】| driveable_surface | 0.453 | 0.544 | 0.650 | nan |
# 576
# 【2024-03-25 15:55:43】| other_flat | 0.233 | 0.261 | 0.282 | nan |
# 577
# 【2024-03-25 15:55:43】| sidewalk | 0.275 | 0.315 | 0.353 | nan |
# 578
# 【2024-03-25 15:55:43】| terrain | 0.284 | 0.348 | 0.405 | nan |
# 579
# 【2024-03-25 15:55:43】| manmade | 0.557 | 0.593 | 0.625 | nan |
# 580
# 【2024-03-25 15:55:43】| vegetation | 0.518 | 0.589 | 0.636 | nan |
# 581
# 【2024-03-25 15:55:43】+----------------------+-------+-------+-------+-------+
# 582
# 【2024-03-25 15:55:43】| MEAN | 0.407 | 0.445 | 0.472 | 2.156 |
# 583
# 【2024-03-25 15:55:43】+----------------------+-------+-------+-------+-------+
# 584
# 【2024-03-25 15:55:43】 --- Occ score: 0.39727509237479