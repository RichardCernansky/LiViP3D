point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
class_names = [
    'car', 'truck', 'bus', 'trailer', 'motorcycle', 'bicycle', 'pedestrian'
]
dataset_type = 'NuScenesTrackDatasetRadar'
data_root = 'data/nuscenes/'
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
file_client_args = dict(backend='disk')
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='InstanceRangeFilter',
        point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
    dict(
        type='NormalizeMultiviewImage',
        mean=[103.53, 116.28, 123.675],
        std=[1.0, 1.0, 1.0],
        to_rgb=False),
    dict(type='PadMultiViewImage', size_divisor=32)
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles'),
    dict(
        type='NormalizeMultiviewImage',
        mean=[103.53, 116.28, 123.675],
        std=[1.0, 1.0, 1.0],
        to_rgb=False),
    dict(type='PadMultiViewImage', size_divisor=32)
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type='NuScenesTrackDatasetRadar',
        data_root='data/nuscenes/',
        ann_file='data/nuscenes/nuscenes_tracking_infos_train.pkl',
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadPointsFromMultiSweeps',
                sweeps_num=10,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True),
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[-0.3925, 0.3925],
                scale_ratio_range=[0.95, 1.05],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
            dict(
                type='PointsRangeFilter',
                point_cloud_range=[-50, -50, -5, 50, 50, 3]),
            dict(
                type='ObjectRangeFilter',
                point_cloud_range=[-50, -50, -5, 50, 50, 3]),
            dict(
                type='ObjectNameFilter',
                classes=[
                    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                    'barrier'
                ]),
            dict(type='PointShuffle'),
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                    'barrier'
                ]),
            dict(
                type='Collect3D',
                keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
        ],
        classes=[
            'car', 'truck', 'bus', 'trailer', 'motorcycle', 'bicycle',
            'pedestrian'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=False,
        box_type_3d='LiDAR',
        num_frames_per_sample=3,
        pipeline_single=[
            dict(type='LoadMultiViewImageFromFiles'),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True),
            dict(
                type='InstanceRangeFilter',
                point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
            dict(
                type='NormalizeMultiviewImage',
                mean=[103.53, 116.28, 123.675],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(type='PadMultiViewImage', size_divisor=32)
        ],
        pipeline_post=[
            dict(type='FormatBundle3DTrack'),
            dict(
                type='Collect3D',
                keys=[
                    'gt_bboxes_3d', 'gt_labels_3d', 'instance_inds', 'img',
                    'timestamp', 'l2g_r_mat', 'l2g_t', 'pred_matrix',
                    'polyline_spans', 'mapping', 'instance_idx_2_labels'
                ])
        ],
        use_valid_flag=True,
        camera_types=['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT'],
        do_pred=True),
    val=dict(
        type='NuScenesTrackDatasetRadar',
        data_root='data/nuscenes/',
        ann_file='data/nuscenes/nuscenes_tracking_infos_val.pkl',
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadPointsFromMultiSweeps',
                sweeps_num=10,
                file_client_args=dict(backend='disk')),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='GlobalRotScaleTrans',
                        rot_range=[0, 0],
                        scale_ratio_range=[1.0, 1.0],
                        translation_std=[0, 0, 0]),
                    dict(type='RandomFlip3D'),
                    dict(
                        type='PointsRangeFilter',
                        point_cloud_range=[-50, -50, -5, 50, 50, 3]),
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=[
                            'car', 'truck', 'trailer', 'bus',
                            'construction_vehicle', 'bicycle', 'motorcycle',
                            'pedestrian', 'traffic_cone', 'barrier'
                        ],
                        with_label=False),
                    dict(type='Collect3D', keys=['points'])
                ])
        ],
        classes=[
            'car', 'truck', 'bus', 'trailer', 'motorcycle', 'bicycle',
            'pedestrian'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        box_type_3d='LiDAR',
        pipeline_single=[
            dict(type='LoadMultiViewImageFromFiles'),
            dict(
                type='NormalizeMultiviewImage',
                mean=[103.53, 116.28, 123.675],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(type='PadMultiViewImage', size_divisor=32)
        ],
        pipeline_post=[
            dict(type='FormatBundle3DTrack'),
            dict(
                type='Collect3D',
                keys=[
                    'img', 'timestamp', 'l2g_r_mat', 'l2g_t', 'pred_matrix',
                    'polyline_spans', 'mapping', 'instance_idx_2_labels'
                ])
        ],
        num_frames_per_sample=1,
        camera_types=['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT'],
        do_pred=True),
    test=dict(
        type='NuScenesTrackDatasetRadar',
        data_root='data/nuscenes/',
        ann_file='data/nuscenes/nuscenes_tracking_infos_val.pkl',
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadPointsFromMultiSweeps',
                sweeps_num=10,
                file_client_args=dict(backend='disk')),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='GlobalRotScaleTrans',
                        rot_range=[0, 0],
                        scale_ratio_range=[1.0, 1.0],
                        translation_std=[0, 0, 0]),
                    dict(type='RandomFlip3D'),
                    dict(
                        type='PointsRangeFilter',
                        point_cloud_range=[-50, -50, -5, 50, 50, 3]),
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=[
                            'car', 'truck', 'trailer', 'bus',
                            'construction_vehicle', 'bicycle', 'motorcycle',
                            'pedestrian', 'traffic_cone', 'barrier'
                        ],
                        with_label=False),
                    dict(type='Collect3D', keys=['points'])
                ])
        ],
        classes=[
            'car', 'truck', 'bus', 'trailer', 'motorcycle', 'bicycle',
            'pedestrian'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        box_type_3d='LiDAR',
        pipeline_single=[
            dict(type='LoadMultiViewImageFromFiles'),
            dict(
                type='NormalizeMultiviewImage',
                mean=[103.53, 116.28, 123.675],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(type='PadMultiViewImage', size_divisor=32)
        ],
        pipeline_post=[
            dict(type='FormatBundle3DTrack'),
            dict(
                type='Collect3D',
                keys=[
                    'img', 'timestamp', 'l2g_r_mat', 'l2g_t', 'pred_matrix',
                    'polyline_spans', 'mapping', 'instance_idx_2_labels'
                ])
        ],
        num_frames_per_sample=1,
        camera_types=['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT'],
        do_pred=True))
evaluation = dict(interval=24)
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'work_dirs/vip3d_resnet50_3cam'
load_from = 'ckpt_init/detr3d_resnet50.pth'
resume_from = None
workflow = [('train', 1)]
plugin = True
plugin_dir = 'plugin/'
voxel_size = [0.2, 0.2, 8]
img_norm_cfg = dict(
    mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
model = dict(
    type='ViP3D',
    use_grid_mask=True,
    num_classes=7,
    num_query=300,
    bbox_coder=dict(
        type='DETRTrack3DCoder',
        post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        max_num=100,
        num_classes=7),
    fix_feats=False,
    score_thresh=0.4,
    filter_score_thresh=0.35,
    use_lidar=False,
    qim_args=dict(
        qim_type='QIMBase',
        merger_dropout=0,
        update_query_pos=True,
        fp_ratio=0.3,
        random_drop=0.1),
    mem_cfg=dict(
        memory_bank_type='MemoryBank',
        memory_bank_score_thresh=0.0,
        memory_bank_len=4),
    img_backbone=dict(
        type='ResNet',
        with_cp=False,
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True)),
    loss_cfg=dict(
        type='ClipMatcher',
        num_classes=7,
        weight_dict=None,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
        assigner=dict(
            type='HungarianAssigner3DTrack',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25)),
    img_neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs=True,
        num_outs=4,
        norm_cfg=dict(type='BN2d'),
        relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='DeformableDETR3DCamHeadTrackPlusRaw',
        num_classes=7,
        in_channels=256,
        num_cams=3,
        num_feature_levels=4,
        with_box_refine=True,
        transformer=dict(
            type='Detr3DCamTrackTransformer',
            decoder=dict(
                type='Detr3DCamTrackPlusTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='Detr3DCrossAtten',
                            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                            num_points=1,
                            embed_dims=256,
                            num_cams=3)
                    ],
                    feedforward_channels=512,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True,
            offset=-0.5)),
    do_pred=True,
    relative_pred=True,
    agents_layer_0=True,
    add_branch=True,
    predictor=dict(
        hidden_size=128,
        laneGCN=True,
        decoder=dict(
            variety_loss=True, variety_loss_prob=True, hidden_size=128)),
    train_cfg=dict(
        pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=[0.2, 0.2, 8],
            point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            out_size_factor=4,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            assigner=dict(
                type='HungarianAssigner3D',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                iou_cost=dict(type='GIoU3DCost', weight=0.0),
                pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]))))
train_pipeline_post = [
    dict(type='FormatBundle3DTrack'),
    dict(
        type='Collect3D',
        keys=[
            'gt_bboxes_3d', 'gt_labels_3d', 'instance_inds', 'img',
            'timestamp', 'l2g_r_mat', 'l2g_t', 'pred_matrix', 'polyline_spans',
            'mapping', 'instance_idx_2_labels'
        ])
]
test_pipeline_post = [
    dict(type='FormatBundle3DTrack'),
    dict(
        type='Collect3D',
        keys=[
            'img', 'timestamp', 'l2g_r_mat', 'l2g_t', 'pred_matrix',
            'polyline_spans', 'mapping', 'instance_idx_2_labels'
        ])
]
optimizer = dict(
    type='AdamW',
    lr=0.0002,
    paramwise_cfg=dict(custom_keys=dict(img_backbone=dict(lr_mult=0.1))),
    weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.3333333333333333,
    min_lr_ratio=0.001)
total_epochs = 24
runner = dict(type='EpochBasedRunner', max_epochs=24)
find_unused_parameters = True
fp16 = dict(loss_scale='dynamic')
gpu_ids = range(0, 1)
