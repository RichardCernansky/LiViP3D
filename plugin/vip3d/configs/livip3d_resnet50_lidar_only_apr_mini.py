_base_ = ['./livip3d_resnet50_lidar_only_apr.py']

# Fast validation only — points the exact same model/APR config at a small,
# contiguous 200/80-sample slice of the existing trainval infos (no v1.0-mini
# raw data is present in this environment, so this is the practical equivalent:
# same pipeline, same everything, just enough samples to iterate in minutes
# instead of days). Not meant to produce a real trained checkpoint.
data = dict(
    train=dict(ann_file='data/nuscenes/nuscenes_tracking_infos_train_mini.pkl'),
    val=dict(ann_file='data/nuscenes/nuscenes_tracking_infos_val_mini.pkl'),
    test=dict(ann_file='data/nuscenes/nuscenes_tracking_infos_val_mini.pkl'),
)
