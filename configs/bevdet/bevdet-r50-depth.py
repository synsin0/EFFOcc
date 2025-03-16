# epoch20_ema.pth
# Evaluating bboxes of pts_bbox
# mAP: 0.3052
# mATE: 0.7215
# mASE: 0.2848
# mAOE: 0.6714
# mAVE: 0.7117
# mAAE: 0.2862
# NDS: 0.3850
# Eval time: 114.1s

# Per-class results:
# Object Class	AP	ATE	ASE	AOE	AVE	AAE
# car	0.542	0.489	0.159	0.108	0.811	0.216
# truck	0.246	0.698	0.220	0.138	0.675	0.238
# bus	0.330	0.770	0.205	0.111	1.371	0.312
# trailer	0.119	1.121	0.232	0.467	0.440	0.131
# construction_vehicle	0.046	0.982	0.517	1.273	0.096	0.392
# pedestrian	0.346	0.750	0.304	1.396	0.837	0.730
# motorcycle	0.242	0.724	0.280	0.952	1.001	0.213
# bicycle	0.226	0.612	0.286	1.443	0.463	0.057
# traffic_cone	0.524	0.499	0.346	nan	nan	nan
# barrier	0.431	0.569	0.300	0.156	nan	nan
# {'pts_bbox_NuScenes/car_AP_dist_0.5': 0.2036, 'pts_bbox_NuScenes/car_AP_dist_1.0': 0.491, 'pts_bbox_NuScenes/car_AP_dist_2.0': 0.6942, 'pts_bbox_NuScenes/car_AP_dist_4.0': 0.7788, 'pts_bbox_NuScenes/car_trans_err': 0.4894, 'pts_bbox_NuScenes/car_scale_err': 0.1589, 'pts_bbox_NuScenes/car_orient_err': 0.1075, 'pts_bbox_NuScenes/car_vel_err': 0.8108, 'pts_bbox_NuScenes/car_attr_err': 0.2159, 'pts_bbox_NuScenes/mATE': 0.7215, 'pts_bbox_NuScenes/mASE': 0.2848, 'pts_bbox_NuScenes/mAOE': 0.6714, 'pts_bbox_NuScenes/mAVE': 0.7117, 'pts_bbox_NuScenes/mAAE': 0.2862, 'pts_bbox_NuScenes/truck_AP_dist_0.5': 0.0295, 'pts_bbox_NuScenes/truck_AP_dist_1.0': 0.1601, 'pts_bbox_NuScenes/truck_AP_dist_2.0': 0.3398, 'pts_bbox_NuScenes/truck_AP_dist_4.0': 0.4537, 'pts_bbox_NuScenes/truck_trans_err': 0.6984, 'pts_bbox_NuScenes/truck_scale_err': 0.2198, 'pts_bbox_NuScenes/truck_orient_err': 0.1379, 'pts_bbox_NuScenes/truck_vel_err': 0.6754, 'pts_bbox_NuScenes/truck_attr_err': 0.2384, 'pts_bbox_NuScenes/construction_vehicle_AP_dist_0.5': 0.0, 'pts_bbox_NuScenes/construction_vehicle_AP_dist_1.0': 0.0059, 'pts_bbox_NuScenes/construction_vehicle_AP_dist_2.0': 0.068, 'pts_bbox_NuScenes/construction_vehicle_AP_dist_4.0': 0.1108, 'pts_bbox_NuScenes/construction_vehicle_trans_err': 0.9821, 'pts_bbox_NuScenes/construction_vehicle_scale_err': 0.5174, 'pts_bbox_NuScenes/construction_vehicle_orient_err': 1.2728, 'pts_bbox_NuScenes/construction_vehicle_vel_err': 0.0963, 'pts_bbox_NuScenes/construction_vehicle_attr_err': 0.3922, 'pts_bbox_NuScenes/bus_AP_dist_0.5': 0.025, 'pts_bbox_NuScenes/bus_AP_dist_1.0': 0.2213, 'pts_bbox_NuScenes/bus_AP_dist_2.0': 0.4817, 'pts_bbox_NuScenes/bus_AP_dist_4.0': 0.593, 'pts_bbox_NuScenes/bus_trans_err': 0.7701, 'pts_bbox_NuScenes/bus_scale_err': 0.205, 'pts_bbox_NuScenes/bus_orient_err': 0.111, 'pts_bbox_NuScenes/bus_vel_err': 1.3707, 'pts_bbox_NuScenes/bus_attr_err': 0.3118, 'pts_bbox_NuScenes/trailer_AP_dist_0.5': 0.0, 'pts_bbox_NuScenes/trailer_AP_dist_1.0': 0.0182, 'pts_bbox_NuScenes/trailer_AP_dist_2.0': 0.1696, 'pts_bbox_NuScenes/trailer_AP_dist_4.0': 0.2883, 'pts_bbox_NuScenes/trailer_trans_err': 1.1207, 'pts_bbox_NuScenes/trailer_scale_err': 0.2318, 'pts_bbox_NuScenes/trailer_orient_err': 0.4667, 'pts_bbox_NuScenes/trailer_vel_err': 0.4397, 'pts_bbox_NuScenes/trailer_attr_err': 0.1311, 'pts_bbox_NuScenes/barrier_AP_dist_0.5': 0.1647, 'pts_bbox_NuScenes/barrier_AP_dist_1.0': 0.4194, 'pts_bbox_NuScenes/barrier_AP_dist_2.0': 0.5381, 'pts_bbox_NuScenes/barrier_AP_dist_4.0': 0.6016, 'pts_bbox_NuScenes/barrier_trans_err': 0.5694, 'pts_bbox_NuScenes/barrier_scale_err': 0.3001, 'pts_bbox_NuScenes/barrier_orient_err': 0.1559, 'pts_bbox_NuScenes/barrier_vel_err': nan, 'pts_bbox_NuScenes/barrier_attr_err': nan, 'pts_bbox_NuScenes/motorcycle_AP_dist_0.5': 0.0608, 'pts_bbox_NuScenes/motorcycle_AP_dist_1.0': 0.2108, 'pts_bbox_NuScenes/motorcycle_AP_dist_2.0': 0.322, 'pts_bbox_NuScenes/motorcycle_AP_dist_4.0': 0.3737, 'pts_bbox_NuScenes/motorcycle_trans_err': 0.7239, 'pts_bbox_NuScenes/motorcycle_scale_err': 0.2797, 'pts_bbox_NuScenes/motorcycle_orient_err': 0.9525, 'pts_bbox_NuScenes/motorcycle_vel_err': 1.0008, 'pts_bbox_NuScenes/motorcycle_attr_err': 0.2133, 'pts_bbox_NuScenes/bicycle_AP_dist_0.5': 0.0761, 'pts_bbox_NuScenes/bicycle_AP_dist_1.0': 0.2236, 'pts_bbox_NuScenes/bicycle_AP_dist_2.0': 0.2858, 'pts_bbox_NuScenes/bicycle_AP_dist_4.0': 0.3199, 'pts_bbox_NuScenes/bicycle_trans_err': 0.6119, 'pts_bbox_NuScenes/bicycle_scale_err': 0.2862, 'pts_bbox_NuScenes/bicycle_orient_err': 1.4427, 'pts_bbox_NuScenes/bicycle_vel_err': 0.4626, 'pts_bbox_NuScenes/bicycle_attr_err': 0.0569, 'pts_bbox_NuScenes/pedestrian_AP_dist_0.5': 0.1276, 'pts_bbox_NuScenes/pedestrian_AP_dist_1.0': 0.3018, 'pts_bbox_NuScenes/pedestrian_AP_dist_2.0': 0.4372, 'pts_bbox_NuScenes/pedestrian_AP_dist_4.0': 0.5178, 'pts_bbox_NuScenes/pedestrian_trans_err': 0.7499, 'pts_bbox_NuScenes/pedestrian_scale_err': 0.3036, 'pts_bbox_NuScenes/pedestrian_orient_err': 1.3956, 'pts_bbox_NuScenes/pedestrian_vel_err': 0.8375, 'pts_bbox_NuScenes/pedestrian_attr_err': 0.73, 'pts_bbox_NuScenes/traffic_cone_AP_dist_0.5': 0.2276, 'pts_bbox_NuScenes/traffic_cone_AP_dist_1.0': 0.5204, 'pts_bbox_NuScenes/traffic_cone_AP_dist_2.0': 0.6437, 'pts_bbox_NuScenes/traffic_cone_AP_dist_4.0': 0.704, 'pts_bbox_NuScenes/traffic_cone_trans_err': 0.4994, 'pts_bbox_NuScenes/traffic_cone_scale_err': 0.3457, 'pts_bbox_NuScenes/traffic_cone_orient_err': nan, 'pts_bbox_NuScenes/traffic_cone_vel_err': nan, 'pts_bbox_NuScenes/traffic_cone_attr_err': nan, 'pts_bbox_NuScenes/NDS': 0.38504845466873744, 'pts_bbox_NuScenes/mAP': 0.3052291602879023}

# epoch20.pth

# mAP: 0.2838
# mATE: 0.7604
# mASE: 0.2953
# mAOE: 0.7000
# mAVE: 0.7780
# mAAE: 0.2697
# NDS: 0.3616
# Eval time: 126.6s

# Per-class results:
# Object Class	AP	ATE	ASE	AOE	AVE	AAE
# car	0.499	0.549	0.163	0.123	0.903	0.225
# truck	0.217	0.755	0.228	0.128	0.775	0.242
# bus	0.296	0.812	0.220	0.140	1.612	0.377
# trailer	0.077	1.142	0.269	0.602	0.542	0.095
# construction_vehicle	0.043	1.050	0.513	1.376	0.094	0.371
# pedestrian	0.341	0.751	0.309	1.406	0.877	0.591
# motorcycle	0.228	0.755	0.281	0.951	0.997	0.223
# bicycle	0.226	0.670	0.308	1.397	0.425	0.033
# traffic_cone	0.506	0.521	0.356	nan	nan	nan
# barrier	0.404	0.598	0.308	0.176	nan	nan
# {'pts_bbox_NuScenes/car_AP_dist_0.5': 0.1513, 'pts_bbox_NuScenes/car_AP_dist_1.0': 0.4183, 'pts_bbox_NuScenes/car_AP_dist_2.0': 0.661, 'pts_bbox_NuScenes/car_AP_dist_4.0': 0.7643, 'pts_bbox_NuScenes/car_trans_err': 0.5487, 'pts_bbox_NuScenes/car_scale_err': 0.1627, 'pts_bbox_NuScenes/car_orient_err': 0.1234, 'pts_bbox_NuScenes/car_vel_err': 0.9029, 'pts_bbox_NuScenes/car_attr_err': 0.2253, 'pts_bbox_NuScenes/mATE': 0.7604, 'pts_bbox_NuScenes/mASE': 0.2953, 'pts_bbox_NuScenes/mAOE': 0.7, 'pts_bbox_NuScenes/mAVE': 0.778, 'pts_bbox_NuScenes/mAAE': 0.2697, 'pts_bbox_NuScenes/truck_AP_dist_0.5': 0.0128, 'pts_bbox_NuScenes/truck_AP_dist_1.0': 0.1231, 'pts_bbox_NuScenes/truck_AP_dist_2.0': 0.3019, 'pts_bbox_NuScenes/truck_AP_dist_4.0': 0.4307, 'pts_bbox_NuScenes/truck_trans_err': 0.7547, 'pts_bbox_NuScenes/truck_scale_err': 0.2278, 'pts_bbox_NuScenes/truck_orient_err': 0.1285, 'pts_bbox_NuScenes/truck_vel_err': 0.7749, 'pts_bbox_NuScenes/truck_attr_err': 0.2425, 'pts_bbox_NuScenes/construction_vehicle_AP_dist_0.5': 0.0, 'pts_bbox_NuScenes/construction_vehicle_AP_dist_1.0': 0.002, 'pts_bbox_NuScenes/construction_vehicle_AP_dist_2.0': 0.0635, 'pts_bbox_NuScenes/construction_vehicle_AP_dist_4.0': 0.107, 'pts_bbox_NuScenes/construction_vehicle_trans_err': 1.0502, 'pts_bbox_NuScenes/construction_vehicle_scale_err': 0.5128, 'pts_bbox_NuScenes/construction_vehicle_orient_err': 1.3756, 'pts_bbox_NuScenes/construction_vehicle_vel_err': 0.0938, 'pts_bbox_NuScenes/construction_vehicle_attr_err': 0.3712, 'pts_bbox_NuScenes/bus_AP_dist_0.5': 0.0199, 'pts_bbox_NuScenes/bus_AP_dist_1.0': 0.1545, 'pts_bbox_NuScenes/bus_AP_dist_2.0': 0.4446, 'pts_bbox_NuScenes/bus_AP_dist_4.0': 0.5666, 'pts_bbox_NuScenes/bus_trans_err': 0.8121, 'pts_bbox_NuScenes/bus_scale_err': 0.2196, 'pts_bbox_NuScenes/bus_orient_err': 0.1402, 'pts_bbox_NuScenes/bus_vel_err': 1.6122, 'pts_bbox_NuScenes/bus_attr_err': 0.3772, 'pts_bbox_NuScenes/trailer_AP_dist_0.5': 0.0, 'pts_bbox_NuScenes/trailer_AP_dist_1.0': 0.0107, 'pts_bbox_NuScenes/trailer_AP_dist_2.0': 0.0972, 'pts_bbox_NuScenes/trailer_AP_dist_4.0': 0.2006, 'pts_bbox_NuScenes/trailer_trans_err': 1.1424, 'pts_bbox_NuScenes/trailer_scale_err': 0.2686, 'pts_bbox_NuScenes/trailer_orient_err': 0.6024, 'pts_bbox_NuScenes/trailer_vel_err': 0.542, 'pts_bbox_NuScenes/trailer_attr_err': 0.0946, 'pts_bbox_NuScenes/barrier_AP_dist_0.5': 0.131, 'pts_bbox_NuScenes/barrier_AP_dist_1.0': 0.3895, 'pts_bbox_NuScenes/barrier_AP_dist_2.0': 0.5122, 'pts_bbox_NuScenes/barrier_AP_dist_4.0': 0.5833, 'pts_bbox_NuScenes/barrier_trans_err': 0.5984, 'pts_bbox_NuScenes/barrier_scale_err': 0.3077, 'pts_bbox_NuScenes/barrier_orient_err': 0.1762, 'pts_bbox_NuScenes/barrier_vel_err': nan, 'pts_bbox_NuScenes/barrier_attr_err': nan, 'pts_bbox_NuScenes/motorcycle_AP_dist_0.5': 0.0397, 'pts_bbox_NuScenes/motorcycle_AP_dist_1.0': 0.19, 'pts_bbox_NuScenes/motorcycle_AP_dist_2.0': 0.3153, 'pts_bbox_NuScenes/motorcycle_AP_dist_4.0': 0.3675, 'pts_bbox_NuScenes/motorcycle_trans_err': 0.7554, 'pts_bbox_NuScenes/motorcycle_scale_err': 0.2806, 'pts_bbox_NuScenes/motorcycle_orient_err': 0.9509, 'pts_bbox_NuScenes/motorcycle_vel_err': 0.9968, 'pts_bbox_NuScenes/motorcycle_attr_err': 0.2225, 'pts_bbox_NuScenes/bicycle_AP_dist_0.5': 0.0604, 'pts_bbox_NuScenes/bicycle_AP_dist_1.0': 0.2058, 'pts_bbox_NuScenes/bicycle_AP_dist_2.0': 0.2963, 'pts_bbox_NuScenes/bicycle_AP_dist_4.0': 0.3428, 'pts_bbox_NuScenes/bicycle_trans_err': 0.6697, 'pts_bbox_NuScenes/bicycle_scale_err': 0.3081, 'pts_bbox_NuScenes/bicycle_orient_err': 1.3968, 'pts_bbox_NuScenes/bicycle_vel_err': 0.4246, 'pts_bbox_NuScenes/bicycle_attr_err': 0.0335, 'pts_bbox_NuScenes/pedestrian_AP_dist_0.5': 0.1138, 'pts_bbox_NuScenes/pedestrian_AP_dist_1.0': 0.2877, 'pts_bbox_NuScenes/pedestrian_AP_dist_2.0': 0.4352, 'pts_bbox_NuScenes/pedestrian_AP_dist_4.0': 0.5266, 'pts_bbox_NuScenes/pedestrian_trans_err': 0.7513, 'pts_bbox_NuScenes/pedestrian_scale_err': 0.3091, 'pts_bbox_NuScenes/pedestrian_orient_err': 1.4061, 'pts_bbox_NuScenes/pedestrian_vel_err': 0.8768, 'pts_bbox_NuScenes/pedestrian_attr_err': 0.5911, 'pts_bbox_NuScenes/traffic_cone_AP_dist_0.5': 0.2018, 'pts_bbox_NuScenes/traffic_cone_AP_dist_1.0': 0.4945, 'pts_bbox_NuScenes/traffic_cone_AP_dist_2.0': 0.6302, 'pts_bbox_NuScenes/traffic_cone_AP_dist_4.0': 0.6977, 'pts_bbox_NuScenes/traffic_cone_trans_err': 0.5207, 'pts_bbox_NuScenes/traffic_cone_scale_err': 0.3563, 'pts_bbox_NuScenes/traffic_cone_orient_err': nan, 'pts_bbox_NuScenes/traffic_cone_vel_err': nan, 'pts_bbox_NuScenes/traffic_cone_attr_err': nan, 'pts_bbox_NuScenes/NDS': 0.36155209106018993, 'pts_bbox_NuScenes/mAP': 0.28378908025147453}


_base_ = ['../_base_/datasets/nus-3d.py', '../_base_/default_runtime.py']
# Global
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
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

# Model
grid_config = {
    'x': [-51.2, 51.2, 0.8],
    'y': [-51.2, 51.2, 0.8],
    'z': [-5, 3, 8],
    'depth': [1.0, 60.0, 0.5],
}

voxel_size = [0.1, 0.1, 0.2]

numC_Trans = 80


model = dict(
    type='BEVDepth',
    img_backbone=dict(
        pretrained='torchvision://resnet50',
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style='pytorch'),
    img_neck=dict(
        type='CustomFPN',
        in_channels=[1024, 2048],
        out_channels=512,
        num_outs=1,
        start_level=0,
        out_ids=[0]),
    img_view_transformer=dict(
        type='LSSViewTransformerBEVDepth',
        grid_config=grid_config,
        input_size=data_config['input_size'],
        in_channels=512,
        out_channels=numC_Trans,
        depthnet_cfg=dict(use_dcn=False, aspp_mid_channels=96),
        downsample=16),
    img_bev_encoder_backbone=dict(
        type='CustomResNet',
        numC_input=numC_Trans,
        num_channels=[numC_Trans * 2, numC_Trans * 4, numC_Trans * 8]),
    img_bev_encoder_neck=dict(
        type='FPN_LSS',
        in_channels=numC_Trans * 8 + numC_Trans * 2,
        out_channels=256),
    pts_bbox_head=dict(
        type='CenterHead',
        in_channels=256,
        tasks=[
            dict(num_class=10, class_names=['car', 'truck',
                                            'construction_vehicle',
                                            'bus', 'trailer',
                                            'barrier',
                                            'motorcycle', 'bicycle',
                                            'pedestrian', 'traffic_cone']),
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            pc_range=point_cloud_range[:2],
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            code_size=9),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean', loss_weight=6.),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=1.5),
        norm_bbox=True),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            point_cloud_range=point_cloud_range,
            grid_size=[1024, 1024, 40],
            voxel_size=voxel_size,
            out_size_factor=8,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])),
    test_cfg=dict(
        pts=dict(
            pc_range=point_cloud_range[:2],
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            pre_max_size=1000,
            post_max_size=500,

            # Scale-NMS
            nms_type=['rotate'],
            nms_thr=[0.2],
            nms_rescale_factor=[[1.0, 0.7, 0.7, 0.4, 0.55,
                                 1.1, 1.0, 1.0, 1.5, 3.5]]
        )
    )
)

# Data
dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

bda_aug_conf = dict(
    rot_lim=(-22.5, 22.5),
    scale_lim=(0.95, 1.05),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)

train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config=data_config,
        sequential=False),
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
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D', keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d',
                                'gt_depth'])
]

test_pipeline = [
    dict(type='PrepareImageInputs', data_config=data_config, sequential=False),
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
    classes=class_names,
    modality=input_modality,
    img_info_prototype='bevdet',
)

test_data_config = dict(
    pipeline=test_pipeline,
    ann_file=data_root + 'bevdetv3-nuscenes_infos_val.pkl')

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=8,
    train=dict(
        # type='CBGSDataset',
        # dataset=dict(
        data_root=data_root,
        ann_file=data_root + 'bevdetv3-nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        test_mode=False,
        use_valid_flag=True,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR',
        # )
        ),
    val=test_data_config,
    test=test_data_config)

for key in ['val', 'test']:
    data[key].update(share_data_config)
# data['train']['dataset'].update(share_data_config)
data['train'].update(share_data_config)

# Optimizer
optimizer = dict(type='AdamW', lr=2e-4, weight_decay=1e-2)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[20,])
runner = dict(type='EpochBasedRunner', max_epochs=20)

custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
    ),
    dict(
        type='SequentialControlHook',
        temporal_start_epoch=2,
    ),
]

# fp16 = dict(loss_scale='dynamic')
