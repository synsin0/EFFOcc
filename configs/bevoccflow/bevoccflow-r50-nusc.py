_base_ = ['../_base_/datasets/nus-3d.py',
          '../_base_/default_runtime.py']

plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

# pose_file = 'data/occ3d-waymo/cam_infos.pkl'
# data_root='data/occ3d-waymo/kitti_format/'
# occ_gt_data_root='data/occ3d-waymo/voxel04/training/'
# val_pose_file='data/occ3d-waymo/cam_infos_vali.pkl'
# occ_val_gt_data_root='data/occ3d-waymo/voxel04/validation/'

# num_views = 5
# FREE_LABEL = 23
# load_interval = 5

# class_weight_binary = [5.314075572339673, 1]
# class_weight_multiclass = [
#     21.996729830048952,
#     7.504469780801267, 10.597629961083673, 12.18107968968811, 15.143940258446506, 13.035521328502758, 
#     9.861234292376812, 13.64431851057796, 15.121236434460473, 21.996729830048952, 6.201671013759701, 
#     5.7420517938838325, 9.768712859518626, 3.4607400626606317, 4.152268220983671, 1.000000000000000,
# ]

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
    'z': [-1, 5.4, 0.4],
    'depth': [1.0, 45.0, 0.5],
}
point_cloud_range = [-40, -40, -1, 40, 40, 5.4]

voxel_size = [0.05, 0.05, 0.16]

numC_Trans = 32


model = dict(
    type='BEVDetOCCFlow',
    num_classes = 18,
    use_infov_mask=False,
    use_lidar_mask=False,
    use_camera_mask=False,
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style='pytorch',
        pretrained='torchvision://resnet50',
    ),
    img_neck=dict(
        type='CustomFPN',
        in_channels=[1024, 2048],
        out_channels=256,
        num_outs=1,
        start_level=0,
        out_ids=[0]),
    img_view_transformer=dict(
        type='LSSViewTransformer',
        grid_config=grid_config,
        input_size=data_config['input_size'],
        in_channels=256,
        out_channels=numC_Trans,
        sid=False,
        collapse_z=False,
        downsample=16),
    img_bev_encoder_backbone=dict(
        type='CustomResNet3D',
        with_cp = True,
        numC_input=numC_Trans,
        num_layer=[1, 2, 4],
        num_channels=[numC_Trans,numC_Trans*2,numC_Trans*4],
        stride=[1,2,2],
        backbone_output_ids=[0,1,2],
        ),
    img_bev_encoder_neck=dict(
        type='LSSFPN3D',
        in_channels=numC_Trans*7,
        out_channels=numC_Trans
    ),
    loss_occ=dict(
        type='CrossEntropyLoss',
        use_sigmoid=False,
        loss_weight=1.0),
    loss_flow=dict(type='L1Loss', loss_weight=0.25),
    use_mask=False,
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
        sequential=False,
        ),
    dict(type='LoadOccGTFromFileV2'),
    dict(type='LoadAnnotations'),
    dict(
        type='BEVAug',
        bda_aug_conf=bda_aug_conf,
        classes=class_names),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(type='PointToMultiViewDepth', downsample=1, grid_config=grid_config),        
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D', keys=['img_inputs', 'points', 'voxel_semantics', 'voxel_flow','voxel_instances'])
]

test_pipeline = [
    dict(type='PrepareImageInputs', data_config=data_config, sequential=False),
    # dict(type='LoadOccGTFromFileWaymo',data_root=occ_val_gt_data_root, num_classes = num_classes, free_label=FREE_LABEL, use_larger=True, crop_x=False),
    dict(type='LoadAnnotations'),

    dict(type='BEVAug',
         bda_aug_conf=bda_aug_conf,
         classes=class_names,
         is_train=False),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
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
            dict(type='Collect3D', keys=['points','img_inputs'])
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
    # use_infov_mask=True,
    # use_lidar_mask=False,
    # use_camera_mask=True,
)

test_data_config = dict(
    pipeline=test_pipeline,
    ann_file=data_root + 'bevdetv3-nuscenes_infos_val_occflow.pkl')

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
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
optimizer = dict(type='AdamW', lr=2e-4, weight_decay=1e-2)
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

load_from = "ckpts/bevdet/bevdet-r50-cbgs.pth"
# fp16 = dict(loss_scale='dynamic')
evaluation = dict(interval=1, start=24, pipeline=test_pipeline)
checkpoint_config = dict(interval=1, max_keep_ckpts=5)


# 579
# 【2024-03-25 16:28:02】+----------------------+-------+-------+-------+-------+
# 580
# 【2024-03-25 16:28:02】| Class Names | IoU@1 | IoU@2 | IoU@4 | AVE |
# 581
# 【2024-03-25 16:28:02】+----------------------+-------+-------+-------+-------+
# 582
# 【2024-03-25 16:28:02】| car | 0.433 | 0.512 | 0.540 | 2.391 |
# 583
# 【2024-03-25 16:28:02】| truck | 0.295 | 0.390 | 0.432 | 1.651 |
# 584
# 【2024-03-25 16:28:02】| trailer | 0.200 | 0.243 | 0.275 | 3.619 |
# 585
# 【2024-03-25 16:28:02】| bus | 0.436 | 0.537 | 0.596 | 2.780 |
# 586
# 【2024-03-25 16:28:02】| construction_vehicle | 0.209 | 0.275 | 0.299 | 0.078 |
# 587
# 【2024-03-25 16:28:02】| bicycle | 0.000 | 0.000 | 0.000 | nan |
# 588
# 【2024-03-25 16:28:02】| motorcycle | 0.053 | 0.065 | 0.069 | 2.334 |
# 589
# 【2024-03-25 16:28:02】| pedestrian | 0.101 | 0.120 | 0.125 | 0.886 |
# 590
# 【2024-03-25 16:28:02】| traffic_cone | 0.037 | 0.041 | 0.043 | nan |
# 591
# 【2024-03-25 16:28:02】| barrier | 0.267 | 0.313 | 0.333 | nan |
# 592
# 【2024-03-25 16:28:02】| driveable_surface | 0.444 | 0.530 | 0.634 | nan |
# 593
# 【2024-03-25 16:28:02】| other_flat | 0.211 | 0.248 | 0.279 | nan |
# 594
# 【2024-03-25 16:28:02】| sidewalk | 0.213 | 0.254 | 0.298 | nan |
# 595
# 【2024-03-25 16:28:02】| terrain | 0.205 | 0.268 | 0.328 | nan |
# 596
# 【2024-03-25 16:28:02】| manmade | 0.161 | 0.206 | 0.231 | nan |
# 597
# 【2024-03-25 16:28:02】| vegetation | 0.131 | 0.201 | 0.262 | nan |
# 598
# 【2024-03-25 16:28:02】+----------------------+-------+-------+-------+-------+
# 599
# 【2024-03-25 16:28:02】| MEAN | 0.212 | 0.263 | 0.296 | 1.963 |
# 600
# 【2024-03-25 16:28:02】+----------------------+-------+-------+-------+-------+
# 601
# 【2024-03-25 16:28:02】 --- Occ score: 0.23141508667169183