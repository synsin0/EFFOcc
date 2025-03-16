_base_ = ['../_base_/datasets/nus-3d.py',
          '../_base_/default_runtime.py']

plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

pose_file = 'data/occ3d-waymo/cam_infos.pkl'
data_root='data/occ3d-waymo/kitti_format/'
occ_gt_data_root='data/occ3d-waymo/voxel04/training/'
val_pose_file='data/occ3d-waymo/cam_infos_vali.pkl'
occ_val_gt_data_root='data/occ3d-waymo/voxel04/validation/'

num_views = 5
FREE_LABEL = 23
load_interval = 1

class_weight_binary = [5.314075572339673, 1]
class_weight_multiclass = [
    21.996729830048952,
    7.504469780801267, 10.597629961083673, 12.18107968968811, 15.143940258446506, 13.035521328502758, 
    9.861234292376812, 13.64431851057796, 15.121236434460473, 21.996729830048952, 6.201671013759701, 
    5.7420517938838325, 9.768712859518626, 3.4607400626606317, 4.152268220983671, 1.000000000000000,
]

data_config = {
    'cams': [
        '0', '1', '2', '3', '4'
    ],
    'Ncams':
    5,
    'input_size': (256, 704),
    'src_size': (1280, 1920),

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

num_classes = 16

img_feat_dim = 128

model = dict(
    type='FlashFusionOCC',
    use_infov_mask=True,
    use_lidar_mask=False,
    use_camera_mask=True,

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
        # class_weight = class_weight_multiclass,
        in_dim=512,
        out_dim=256,
        Dz=16,
        use_mask=True,
        num_classes = num_classes,
        use_predicter=True,
        class_wise=False,
        loss_occ=dict(
            type='CrossEntropyOHEMLoss',
            class_weight=class_weight_multiclass,
            use_sigmoid=False,
            use_mask=False,
            loss_weight=1.0, 
            top_ratio=0.2,
            top_weight=4.0,
        ),
    )
)

# Data
dataset_type = 'WaymoDatasetOccupancy'
data_root = 'data/occ3d-waymo/kitti_format/'
file_client_args = dict(backend='disk')

bda_aug_conf = dict(
    rot_lim=(-0., 0.),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5
)


train_pipeline = [
    dict(
        type='PrepareImageInputsWaymo',
        is_train=True,
        data_config=data_config,
        sequential=False,
        ),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=6,
        use_dim=5,
        tanh_dim=[3, 4],
        file_client_args=file_client_args),
    # dict(
    #     type='LoadPointsFromMultiSweeps',
    #     sweeps_num=9,
    #     use_dim=[0, 1, 2, 3, 4],
    #     file_client_args=file_client_args,
    #     pad_empty_sweeps=True,
    #     remove_close=True),
    # dict(type='ToEgo'),
    dict(type='LoadOccGTFromFileWaymo',data_root=occ_gt_data_root, num_classes = num_classes, free_label=FREE_LABEL, use_larger=True, crop_x=False),
    dict(type='LoadAnnotations3D'),
    dict(
        type='BEVAug',
        bda_aug_conf=bda_aug_conf,
        classes=class_names),
    dict(type='PointToMultiViewDepthFusionWaymo', downsample=1, grid_config=grid_config),

    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D', keys=['img_inputs', 'points', 'voxel_semantics', 'mask_lidar','mask_camera','mask_infov', 'gt_depth'])
]

test_pipeline = [
    dict(type='PrepareImageInputsWaymo', data_config=data_config, sequential=False),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=6,
        use_dim=5,
        tanh_dim=[3, 4],
        file_client_args=file_client_args),
    # dict(type='ToEgo'),
    # dict(type='LoadOccGTFromFileWaymo',data_root=occ_val_gt_data_root, num_classes = num_classes, free_label=FREE_LABEL, use_larger=True, crop_x=False),
    dict(type='LoadAnnotations3D'),
    dict(type='BEVAug',
         bda_aug_conf=bda_aug_conf,
         classes=class_names,
         is_train=False),
    dict(type='PointToMultiViewDepthFusionWaymo', downsample=1, grid_config=grid_config),

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
    use_infov_mask=True,
    use_lidar_mask=False,
    use_camera_mask=True,
)

test_data_config = dict(
    pipeline=test_pipeline,
    ann_file=data_root + 'waymo_infos_val.pkl')

# data = dict(
#     samples_per_gpu=2,
#     workers_per_gpu=8,
#     train=dict(
#         data_root=data_root,
#         ann_file=data_root + 'waymo_infos_train.pkl',
#         pipeline=train_pipeline,
#         classes=class_names,
#         test_mode=False,
#         use_valid_flag=True,
#         # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
#         # and box_type_3d='Depth' in sunrgbd and scannet dataset.
#         box_type_3d='LiDAR'),
#     val=test_data_config,
#     test=test_data_config)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        load_interval=load_interval,
        num_views=num_views,
        split='training',
        ann_file=data_root + 'waymo_infos_train.pkl',
        pose_file=pose_file,
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        occ_gt_data_root = occ_gt_data_root,
        use_larger = True,
        
        # use_valid_flag=True,
        # history_len=queue_length,
        # bev_size=(bev_h_, bev_w_),
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=dict(type=dataset_type,
             data_root=data_root,
             split='training',
             ann_file=data_root + 'waymo_infos_val.pkl',
             pose_file=val_pose_file,
             num_views=num_views,
             pipeline=test_pipeline,  #bev_size=(bev_h_, bev_w_),
             test_mode=True,
             occ_gt_data_root = occ_val_gt_data_root,
             use_larger = True,
             classes=class_names, modality=input_modality, samples_per_gpu=1),
    test=dict(type=dataset_type,
            data_root=data_root,
            split='training',
            num_views=num_views,
            ann_file=data_root + 'waymo_infos_val.pkl',
            pose_file=val_pose_file,
            pipeline=test_pipeline, #bev_size=(bev_h_, bev_w_),
            classes=class_names, modality=input_modality,
            test_mode=False,
            box_type_3d='LiDAR',
            occ_gt_data_root = occ_val_gt_data_root,
            use_larger = True,
    ),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)


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

# load_from = "ckpts/bevdet-r50-cbgs.pth"
# fp16 = dict(loss_scale='dynamic')
evaluation = dict(interval=1, start=18, pipeline=test_pipeline)
checkpoint_config = dict(interval=1, max_keep_ckpts=5)


# 2023-12-27 02:09:46,342 - mmdet - INFO - Epoch(val) [8][4999]	TYPE_GENERALOBJECT: 11.4300, TYPE_VEHICLE: 61.7200, TYPE_BICYCLIST: 50.2200, TYPE_PEDESTRIAN: 37.0000, TYPE_SIGN: 36.7500, TYPE_TRAFFIC_LIGHT: 30.9700, TYPE_POLE: 46.2100, TYPE_CONSTRUCTION_CONE: 34.1300, TYPE_BICYCLE: 18.5700, TYPE_MOTORCYCLE: 0.0000, TYPE_BUILDING: 64.8900, TYPE_VEGETATION: 58.3600, TYPE_TREE_TRUNK: 39.4200, TYPE_ROAD: 83.3800, TYPE_WALKABLE: 72.1700, mIoU: 43.0200
