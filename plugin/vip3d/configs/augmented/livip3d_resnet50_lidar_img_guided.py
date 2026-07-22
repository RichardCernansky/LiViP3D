_base_ = [
    '../_base_/nus-3d.py',
    '../_base_/default_runtime.py'
]
# TransFusion Table 7 ablation, "w/o Fusion" cell: image-guided query init
# ON, SMCA feature fusion OFF. Stage 2 of the paper's 2-stage scheme --
# loads directly from stage 1's (augmented/livip3d_resnet50_lidar_only.py)
# finished checkpoint. No point-cloud geometric augmentation here: this
# stage's image-guided heatmap (img_hm_task*) is built from a LiDAR-BEV/
# image cross-attention projection computed from calibration matrices
# that are NOT re-derived after augmentation (see dataset.py get_data_info),
# so flip/rotate/scale would desync LiDAR geometry from the image
# projection this stage is training. Trained identically to the
# "w/o Guide" and full-model siblings (fix_lidar=True, 6 epochs) so the
# ablation only isolates the img-guided/SMCA toggles.
workflow = [('train', 1)]
plugin = True
plugin_dir = 'plugin/'

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

class_names = [
    'car', 'truck', 'bus', 'trailer',
    'motorcycle', 'bicycle', 'pedestrian',
]
prediction_eval_classes = [
    'car', 'truck', 'bus', 'trailer',
    'motorcycle', 'bicycle', 'pedestrian',
]

input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

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
    fix_feats=True,   # camera backbone frozen — img guided uses frozen features
    fix_lidar=True,   # stage 1 LiDAR backbone already converged; only train new modules
    score_thresh=0.4,
    filter_score_thresh=0.35,
    use_lidar=True,
    lidar_bev_channels=384,
    lidar_voxel_size=[0.2, 0.2, 8],
    lidar_out_size_factor=2,
    pts_voxel_layer=dict(
        max_num_points=20,
        voxel_size=[0.2, 0.2, 8],
        max_voxels=(30000, 40000),
        point_cloud_range=point_cloud_range),
    pts_voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=5,
        feat_channels=[64],
        with_distance=False,
        voxel_size=[0.2, 0.2, 8],
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        point_cloud_range=point_cloud_range),
    pts_middle_encoder=dict(
        type='PointPillarsScatter',
        in_channels=64,
        output_shape=[512, 512]),
    pts_backbone=dict(
        type='SECOND',
        in_channels=64,
        out_channels=[128, 256],
        layer_nums=[3, 5],
        layer_strides=[2, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[128, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=False),
    qim_args=dict(
        qim_type='QIMBase',
        merger_dropout=0, update_query_pos=True,
        fp_ratio=0.3, random_drop=0.1),
    mem_cfg=dict(
        memory_bank_type='MemoryBank',
        memory_bank_score_thresh=0.0,
        memory_bank_len=4,
    ),
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
            pc_range=point_cloud_range),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
    ),
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
        type='TransFusionDetHead',
        num_classes=7,
        in_channels=256,
        num_cams=6,
        num_feature_levels=4,
        transformer=dict(
            type='TransFusionTransformer',
            decoder=dict(
                type='TransFusionTransformerDecoder',
                embed_dims=256,
                num_heads=8,
                ffn_dims=512,
                dropout=0.1,
                use_smca=False,
                lidar_bev_attn=dict(
                    type='LiDARBEVDeformCrossAtten',
                    embed_dims=256,
                    num_heads=8,
                    num_points=4,
                    bev_in_channels=384,
                    dropout=0.1,
                    pc_range=point_cloud_range),
                smca_attn=dict(
                    type='SMCACrossAtten',
                    embed_dims=256,
                    num_heads=8,
                    num_cams=6,
                    num_levels=4,
                    pc_range=point_cloud_range,
                    dropout=0.1),
            )),
        pc_range=point_cloud_range,
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True,
            offset=-0.5),
    ),
    debug=False,
    bev_vis=True,
    vis_interval=20,
    use_img_guided=True,
    use_smca=False,
    do_pred=True,
    relative_pred=True,
    agents_layer_0=True,
    add_branch=True,
    predictor=dict(
        hidden_size=128,
        laneGCN=True,
        decoder=dict(
            variety_loss=True,
            variety_loss_prob=True,
            hidden_size=128,
        ),
    ),
    train_cfg=dict(
        pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=2,
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
                pc_range=point_cloud_range)
        )
    ),
)

dataset_type = 'NuScenesTrackDatasetRadar'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles'),
    dict(type='ResizeMultiViewKeepRatio', scale=(960, 544), keep_ratio=True),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        load_dim=5,
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='InstanceRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
]
train_pipeline_post = [
    dict(type='FormatBundle3DTrack'),
    dict(type='Collect3D', keys=[
        'gt_bboxes_3d', 'gt_labels_3d', 'instance_inds', 'img',
        'points', 'timestamp', 'l2g_r_mat', 'l2g_t',
        'pred_matrix', 'polyline_spans', 'mapping', 'instance_idx_2_labels']),
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles'),
    dict(type='ResizeMultiViewKeepRatio', scale=(960, 544), keep_ratio=True),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        load_dim=5,
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
]
test_pipeline_post = [
    dict(type='FormatBundle3DTrack'),
    dict(type='Collect3D', keys=[
        'gt_bboxes_3d', 'gt_labels_3d',
        'points', 'img', 'timestamp', 'l2g_r_mat', 'l2g_t',
        'pred_matrix', 'polyline_spans', 'mapping', 'instance_idx_2_labels']),
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        num_frames_per_sample=3,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_tracking_infos_train.pkl',
        pipeline_single=train_pipeline,
        pipeline_post=train_pipeline_post,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        box_type_3d='LiDAR',
        camera_types=['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'],
        do_pred=True),
    val=dict(
        type=dataset_type,
        pipeline_single=test_pipeline,
        pipeline_post=test_pipeline_post,
        classes=class_names,
        modality=input_modality,
        ann_file=data_root + 'nuscenes_tracking_infos_val.pkl',
        num_frames_per_sample=1,
        camera_types=['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'],
        do_pred=True),
    test=dict(
        type=dataset_type,
        pipeline_single=test_pipeline,
        pipeline_post=test_pipeline_post,
        classes=class_names,
        modality=input_modality,
        ann_file=data_root + 'nuscenes_tracking_infos_val.pkl',
        num_frames_per_sample=1,
        camera_types=['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'],
        do_pred=True))

optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.0),  # frozen
            'img_neck':     dict(lr_mult=0.0),  # frozen
            'pts_backbone': dict(lr_mult=0.1),
            'pts_neck':     dict(lr_mult=0.1),
            'heatmap_head': dict(lr_mult=0.1),
            'hm_task0':    dict(lr_mult=0.1),
            'hm_task1':    dict(lr_mult=0.1),
            'hm_task2':    dict(lr_mult=0.1),
            'hm_task4':    dict(lr_mult=0.1),
            'hm_task5':    dict(lr_mult=0.1),
            # img_bev_proj and img_hm_* use base lr (2e-4) — new modules, full lr
        }),
    weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)

total_epochs = 6
evaluation = dict(interval=6)
runner = dict(type='EpochBasedRunner', max_epochs=6)

find_unused_parameters = True
load_from = 'work_dirs/s1-livip3d_lidar_only_augmented/epoch_20.pth'
# fp16 = dict(loss_scale='dynamic')
