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
    type='FlashFusionOCC',
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
        type='BEVOCCHead2D',
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
    )
)

# Data
dataset_type = 'NuScenesDatasetOccpancy'
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
    dict(type='LoadOccGTFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='BEVAug',
        bda_aug_conf=bda_aug_conf,
        classes=class_names),

    dict(type='PointToMultiViewDepthFusion', downsample=1, grid_config=grid_config),

    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D', keys=['points', 'img_inputs', 'gt_depth', 'voxel_semantics',
                                'mask_lidar', 'mask_camera'])
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
    use_lidar=False,
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
    ann_file=data_root + 'bevdetv3-nuscenes_infos_val.pkl')

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        data_root=data_root,
        ann_file=data_root + 'bevdetv3-nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        test_mode=False,
        use_valid_flag=True,
        load_sequences=140,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=test_data_config,
    test=test_data_config)

for key in ['val', 'train', 'test']:
    data[key].update(share_data_config)

# Optimizer
optimizer = dict(type='AdamW', lr=4e-4, weight_decay=1e-2)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[24, ])
runner = dict(type='EpochBasedRunner', max_epochs=48)

custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
    ),
]

# load_from = "ckpts/bevdet-r50-cbgs.pth"
# fp16 = dict(loss_scale='dynamic')
evaluation = dict(interval=1, start=40, pipeline=test_pipeline)
checkpoint_config = dict(interval=1, max_keep_ckpts=5)


# 2023-12-15 05:44:19,677 - mmdet - INFO - Epoch(val) [23][753]	others: 8.6700, barrier: 53.0400, bicycle: 13.0000, bus: 51.5800, car: 59.9100, construction_vehicle: 30.9900, motorcycle: 22.7000, pedestrian: 46.3600, traffic_cone: 31.0900, trailer: 45.6300, truck: 48.5800, driveable_surface: 80.7100, other_flat: 40.7000, sidewalk: 52.9100, terrain: 57.4000, manmade: 69.6000, vegetation: 68.6600, mIoU: 45.9700
