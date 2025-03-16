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
    'input_size': (512, 1408),
    'src_size': (900, 1600),

    # Augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

grid_config = {
    'x': [-51.2, 51.2, 0.2],
    'y': [-51.2, 51.2, 0.2],
    'z': [-5.0, 3.0, 8.0],
    'depth': [1.0, 60.0, 1.0],
}

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

voxel_size = [0.025, 0.025, 0.2]

empty_idx = 0  # noise 0-->255
num_cls = 17  # 0 free, 1-16 obj
occ_size = [512, 512, 40]
visible_mask = False

numC_Trans = 64

img_feat_dim = 128

model = dict(
    type='FlashFusionOCC',
    use_camera_mask = False,
    mode = 'openoccupancy',
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style='pytorch',
        pretrained='torchvision://resnet50',
    ),
    img_neck=dict(
        type='CustomFPN',
        in_channels=[512, 1024, 2048],
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
        with_cp = True,
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
        sparse_shape=[41, 4096, 4096],
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
        in_channels=[numC_Trans, 256],
        out_channels=256,
    ),

    occ_head=dict(
        type='BEVOCCHead2DOpenOccupancy',
        in_dim=512,
        out_dim=256,
        Dz=40,
        use_mask=False,
        num_classes=17,
        use_predicter=True,
        class_wise=False,
        empty_idx=empty_idx,
    )
)

# Data
dataset_type = 'NuScenesDatasetOccpancyOpenOccupancy'
data_root = 'data/nuscenes/'
occ_path = "./data/nuscenes/nuScenes-Occupancy-v0.1"

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
        use_lidar_coord=True,
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
    # dict(type='ToEgo'), 
    # dict(
    #     type='LoadAnnotationsBEVDepth',
    #     bda_aug_conf=bda_aug_conf,
    #     classes=class_names,
    #     is_train=True),
    dict(type='LoadOpenOccupancy', to_float32=True, use_semantic=True, occ_path=occ_path, grid_size=occ_size, use_vel=False,
            unoccupied=empty_idx, pc_range=point_cloud_range, cal_visible=visible_mask),
    dict(type='LoadAnnotations'),
    dict(
        type='BEVAug',
        bda_aug_conf=bda_aug_conf,
        classes=class_names),

    dict(type='PointToMultiViewDepthFusion', downsample=1, grid_config=grid_config, use_lidar_coord=True),

    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D', keys=['points', 'img_inputs', 'gt_depth', 'voxel_semantics'])
]

test_pipeline = [
    dict(type='PrepareImageInputs', data_config=data_config, sequential=False, use_lidar_coord=True),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True),
    # dict(type='ToEgo'),
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
         grid_config=grid_config, use_lidar_coord=True),
    # dict(
    #     type='LoadPointsFromFile',
    #     coord_type='LIDAR',
    #     load_dim=5,
    #     use_dim=5,
    #     file_client_args=file_client_args),
    dict(type='LoadOpenOccupancy', to_float32=True, use_semantic=True, occ_path=occ_path, grid_size=occ_size, use_vel=False,
        unoccupied=empty_idx, pc_range=point_cloud_range, cal_visible=visible_mask),
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
            dict(type='Collect3D', keys=['points', 'img_inputs', 'gt_depth','voxel_semantics'])
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
    ann_file=data_root + 'bevdetv3-nuscenes_infos_val.pkl')

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=8,
    train=dict(
        data_root=data_root,
        ann_file=data_root + 'bevdetv3-nuscenes_infos_train.pkl',
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
runner = dict(type='EpochBasedRunner', max_epochs=15)

custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
    ),
]

load_from = "ckpts/dal/dal-tiny-map66.9-nds71.1.pth"
# fp16 = dict(loss_scale='dynamic')
evaluation = dict(interval=1, start=13, pipeline=test_pipeline)
checkpoint_config = dict(interval=1, max_keep_ckpts=5)


# ===> per class IoU of 6019 samples:
# ===> others - IoU = 10.57
# ===> barrier - IoU = 56.16
# ===> bicycle - IoU = 21.73
# ===> bus - IoU = 58.68
# ===> car - IoU = 63.16
# ===> construction_vehicle - IoU = 31.98
# ===> motorcycle - IoU = 37.71
# ===> pedestrian - IoU = 55.4
# ===> traffic_cone - IoU = 36.15
# ===> trailer - IoU = 45.87
# ===> truck - IoU = 50.81
# ===> driveable_surface - IoU = 81.02
# ===> other_flat - IoU = 39.07
# ===> sidewalk - IoU = 53.08
# ===> terrain - IoU = 57.15
# ===> manmade - IoU = 70.41
# ===> vegetation - IoU = 68.9
# ===> mIoU of 6019 samples: 49.29
# {'others': 10.57, 'barrier': 56.16, 'bicycle': 21.73, 'bus': 58.68, 'car': 63.16, 'construction_vehicle': 31.98, 'motorcycle': 37.71, 'pedestrian': 55.4, 'traffic_cone': 36.15, 'trailer': 45.87, 'truck': 50.81, 'driveable_surface': 81.02, 'other_flat': 39.07, 'sidewalk': 53.08, 'terrain': 57.15, 'manmade': 70.41, 'vegetation': 68.9, 'mIoU': 49.29}