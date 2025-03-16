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
    'Ncams': 6,
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

voxel_size = [0.1, 0.1, 0.2]

numC_Trans = 80
multi_adj_frame_id_cfg = (1, 1+1, 1)


model = dict(
    type='FlashBEVStereo4DOCCFlow',
    align_after_view_transfromation=False,
    num_adj=len(range(*multi_adj_frame_id_cfg)),
    img_backbone=dict(
        pretrained='torchvision://resnet50',
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style='pytorch'),
    img_neck=dict(
        type='CustomFPN',
        in_channels=[1024, 2048],
        out_channels=256,
        num_outs=1,
        start_level=0,
        out_ids=[0]),
    img_view_transformer=dict(
        type='LSSViewTransformerBEVStereo',
        grid_config=grid_config,
        input_size=data_config['input_size'],
        in_channels=256,
        out_channels=numC_Trans,
        sid=True,
        loss_depth_weight=0.05,
        depthnet_cfg=dict(use_dcn=False,
                          aspp_mid_channels=96,
                          stereo=True,
                          bias=5.),
        downsample=16),
    img_bev_encoder_backbone=dict(
        type='CustomResNet',
        numC_input=numC_Trans * (len(range(*multi_adj_frame_id_cfg))+1),
        num_channels=[numC_Trans * 2, numC_Trans * 4, numC_Trans * 8]),
    img_bev_encoder_neck=dict(
        type='FPN_LSS',
        in_channels=numC_Trans * 8 + numC_Trans * 2,
        out_channels=256),
    pre_process=dict(
        type='CustomResNet',
        numC_input=numC_Trans,
        num_layer=[1, ],
        num_channels=[numC_Trans, ],
        stride=[1, ],
        backbone_output_ids=[0, ]),
    occ_head=dict(
        type='BEVOCCFlowHead2D',
        in_dim=256,
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
    flip_dy_ratio=0.5)

train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config=data_config,
        sequential=True),
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
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(type='PointToMultiViewDepth', downsample=1, grid_config=grid_config),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D', keys=['img_inputs', 'gt_depth', 'voxel_semantics',
                                   'voxel_flow','voxel_instances'])
]

test_pipeline = [
    dict(type='PrepareImageInputs', data_config=data_config, sequential=True),
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
            dict(type='Collect3D', keys=['points', 'img_inputs'])
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
    stereo=True,
    filter_empty_gt=False,
    img_info_prototype='bevdet4d',
    multi_adj_frame_id_cfg=multi_adj_frame_id_cfg,
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

load_from = "./ckpts/bevdet/bevdet-r50-4d-stereo-cbgs.pth"
# fp16 = dict(loss_scale='dynamic')
evaluation = dict(interval=1, start=24, pipeline=test_pipeline)
checkpoint_config = dict(interval=1, max_keep_ckpts=5)


# 562
# 【2024-03-25 15:52:01】+----------------------+-------+-------+-------+-------+
# 563
# 【2024-03-25 15:52:01】| Class Names | IoU@1 | IoU@2 | IoU@4 | AVE |
# 564
# 【2024-03-25 15:52:01】+----------------------+-------+-------+-------+-------+
# 565
# 【2024-03-25 15:52:01】| car | 0.497 | 0.561 | 0.585 | 1.931 |
# 566
# 【2024-03-25 15:52:01】| truck | 0.366 | 0.452 | 0.485 | 1.600 |
# 567
# 【2024-03-25 15:52:01】| trailer | 0.248 | 0.320 | 0.404 | 2.680 |
# 568
# 【2024-03-25 15:52:01】| bus | 0.475 | 0.567 | 0.621 | 2.716 |
# 569
# 【2024-03-25 15:52:01】| construction_vehicle | 0.249 | 0.328 | 0.349 | 0.114 |
# 570
# 【2024-03-25 15:52:01】| bicycle | 0.110 | 0.117 | 0.118 | 1.066 |
# 571
# 【2024-03-25 15:52:01】| motorcycle | 0.146 | 0.188 | 0.194 | 2.353 |
# 572
# 【2024-03-25 15:52:01】| pedestrian | 0.229 | 0.248 | 0.254 | 0.846 |
# 573
# 【2024-03-25 15:52:01】| traffic_cone | 0.172 | 0.178 | 0.183 | nan |
# 574
# 【2024-03-25 15:52:01】| barrier | 0.387 | 0.423 | 0.439 | nan |
# 575
# 【2024-03-25 15:52:01】| driveable_surface | 0.431 | 0.520 | 0.630 | nan |
# 576
# 【2024-03-25 15:52:01】| other_flat | 0.216 | 0.252 | 0.281 | nan |
# 577
# 【2024-03-25 15:52:01】| sidewalk | 0.233 | 0.274 | 0.315 | nan |
# 578
# 【2024-03-25 15:52:01】| terrain | 0.226 | 0.288 | 0.346 | nan |
# 579
# 【2024-03-25 15:52:01】| manmade | 0.313 | 0.362 | 0.397 | nan |
# 580
# 【2024-03-25 15:52:01】| vegetation | 0.207 | 0.293 | 0.356 | nan |
# 581
# 【2024-03-25 15:52:01】+----------------------+-------+-------+-------+-------+
# 582
# 【2024-03-25 15:52:01】| MEAN | 0.282 | 0.336 | 0.372 | 1.664 |
# 583
# 【2024-03-25 15:52:01】+----------------------+-------+-------+-------+-------+
# 584
# 【2024-03-25 15:52:01】 --- Occ score: 0.2968795272596858