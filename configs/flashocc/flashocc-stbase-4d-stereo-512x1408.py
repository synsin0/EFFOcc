_base_ = ['../_base_/datasets/nus-3d.py',
          '../_base_/default_runtime.py']
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

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

# Model
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
    type='BEVStereo4DOCC',
    align_after_view_transfromation=False,
    num_adj=len(range(*multi_adj_frame_id_cfg)),
    img_backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=224,
        patch_size=4,
        window_size=12,
        mlp_ratio=4,
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        strides=(4, 2, 2, 2),
        out_indices=(2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        use_abs_pos_embed=False,
        return_stereo_feat=True,
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='LN', requires_grad=True),
        pretrain_style='official',
        output_missing_index_as_none=False),
    img_neck=dict(
        type='FPN_LSS',
        in_channels=512 + 1024,
        out_channels=512,
        # with_cp=False,
        extra_upsample=None,
        input_feature_index=(0, 1),
        scale_factor=2),
    img_view_transformer=dict(
        type='LSSViewTransformerBEVStereo',
        grid_config=grid_config,
        input_size=data_config['input_size'],
        in_channels=512,
        out_channels=numC_Trans,
        sid=False,
        collapse_z=True,
        loss_depth_weight=0.05,
        depthnet_cfg=dict(use_dcn=False,
                          aspp_mid_channels=96,
                          stereo=True,
                          bias=5.),
        downsample=16),
    img_bev_encoder_backbone=dict(
        type='CustomResNet',
        with_cp=True,
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
        type='BEVOCCHead2D',
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
    flip_dy_ratio=0.5)

train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config=data_config,
        sequential=True),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=True),
    dict(type='LoadOccGTFromFile'),
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
                                'mask_lidar','mask_camera'])
]

test_pipeline = [
    dict(type='PrepareImageInputs', data_config=data_config, sequential=True),
    dict(
        type='LoadAnnotationsBEVDepth',
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
    classes=class_names,
    modality=input_modality,
    stereo=True,
    filter_empty_gt=False,
    img_info_prototype='bevdet4d',
    multi_adj_frame_id_cfg=multi_adj_frame_id_cfg,
)

test_data_config = dict(
    data_root=data_root,
    pipeline=test_pipeline,
    ann_file=data_root + 'bevdetv2-nuscenes_infos_val.pkl')

data = dict(
    samples_per_gpu=1,  # with 32 GPU
    workers_per_gpu=4,
    train=dict(
        data_root=data_root,
        ann_file=data_root + 'bevdetv2-nuscenes_infos_train.pkl',
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
    dict(
        type='SyncbnControlHook',
        syncbn_start_epoch=0,
    ),
]

evaluation = dict(interval=6, start=0, pipeline=test_pipeline)
checkpoint_config = dict(interval=1, max_keep_ckpts=3)
load_from="ckpts/bevdet-stbase-4d-stereo-512x1408-cbgs.pth"
# fp16 = dict(loss_scale='dynamic')


# ===> per class IoU of 6019 samples:
# ===> others - IoU = 13.42
# ===> barrier - IoU = 51.07
# ===> bicycle - IoU = 27.68
# ===> bus - IoU = 51.57
# ===> car - IoU = 56.22
# ===> construction_vehicle - IoU = 27.27
# ===> motorcycle - IoU = 29.98
# ===> pedestrian - IoU = 29.93
# ===> traffic_cone - IoU = 29.8
# ===> trailer - IoU = 37.77
# ===> truck - IoU = 43.52
# ===> driveable_surface - IoU = 83.81
# ===> other_flat - IoU = 46.55
# ===> sidewalk - IoU = 56.15
# ===> terrain - IoU = 59.56
# ===> manmade - IoU = 50.84
# ===> vegetation - IoU = 44.67
# ===> mIoU of 6019 samples: 43.52


# 705
# 【2023-12-12 14:36:47】===> per class IoU of 6019 samples:
# 706
# 【2023-12-12 14:36:47】===> others - IoU = 13.4
# 707
# 【2023-12-12 14:36:47】===> barrier - IoU = 50.98
# 708
# 【2023-12-12 14:36:47】===> bicycle - IoU = 27.63
# 709
# 【2023-12-12 14:36:47】===> bus - IoU = 51.59
# 710
# 【2023-12-12 14:36:47】===> car - IoU = 56.1
# 711
# 【2023-12-12 14:36:47】===> construction_vehicle - IoU = 27.24
# 712
# 【2023-12-12 14:36:47】===> motorcycle - IoU = 29.98
# 713
# 【2023-12-12 14:36:47】===> pedestrian - IoU = 29.89
# 714
# 【2023-12-12 14:36:47】===> traffic_cone - IoU = 29.59
# 715
# 【2023-12-12 14:36:47】===> trailer - IoU = 37.62
# 716
# 【2023-12-12 14:36:47】===> truck - IoU = 43.48
# 717
# 【2023-12-12 14:36:47】===> driveable_surface - IoU = 83.72
# 718
# 【2023-12-12 14:36:47】===> other_flat - IoU = 46.35
# 719
# 【2023-12-12 14:36:47】===> sidewalk - IoU = 56.11
# 720
# 【2023-12-12 14:36:47】===> terrain - IoU = 59.59
# 721
# 【2023-12-12 14:36:47】===> manmade - IoU = 50.74
# 722
# 【2023-12-12 14:36:47】===> vegetation - IoU = 44.64
# 723
# 【2023-12-12 14:36:47】===> mIoU of 6019 samples: 43.45
# 724
# 【2023-12-12 14:36:47】{'mIoU': array([0.13401251, 0.50975851, 0.27632565, 0.51590772, 0.56098359,
# 725
# 【2023-12-12 14:36:47】 0.27236905, 0.29975315, 0.29892711, 0.29588309, 0.37619141,
# 726
# 【2023-12-12 14:36:47】 0.43482615, 0.83716681, 0.46350534, 0.56105589, 0.59594136,
# 727
# 【2023-12-12 14:36:47】 0.50739079, 0.44641156, 0.91032558])}
