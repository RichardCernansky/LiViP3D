"""Build a LIFT-style temporal GT-paste database.

The existing GT database (nuscenes_dbinfos_10sweeps_withvelo.pkl, built by
mmdetection3d/tools/data_converter/create_gt_database.py) stores exactly one
box + one recentered point-cloud snapshot per object -- a pasted object can
only ever be replayed as a single frozen or rigidly-extrapolated shape.

This script groups objects by their real identity (`instance_inds`, already
present and scene-stable in nuscenes_tracking_infos_train.pkl -- see
tools/data_converter/nusc_tracking.py:277,291) and stores each instance's own
box+points across up to `--t-max` CONSECUTIVE real keyframes, so a pasted
object can replay its genuine recorded appearance and motion (LIFT paper,
"LIFT: Learning 4D LiDAR Image Fusion Transformer", CVPR 2022, Sec 3.3)
instead of a single snapshot.

Two-pass design:
  Pass 1: walk every keyframe once (same point-loading cost as the existing
          single-frame create_gt_database.py -- LoadPointsFromFile +
          LoadPointsFromMultiSweeps(sweeps_num=10), matching this repo's own
          train_pipeline exactly), crop+recenter every object's points via
          box_np_ops.points_in_rbbox (same as create_gt_database.py:255,295),
          convert box+points to a GLOBAL anchor using THAT frame's own ego
          pose (reusing pipeline.py's _heading_local_to_global -- the same
          formula already verified for the ego-motion reprojection fix),
          write points to disk, and keep per-frame per-object metadata in
          memory.
  Pass 2: for each (instance, frame) pair, walk forward up to `--t-max`
          frames within the same scene collecting that instance's available
          future history (LIFT's {O_{t'-Δt}} candidate sequence, Sec 3.3),
          and emit one db_info per starting frame (mirroring
          create_gt_database.py's "one entry per object per keyframe"
          density -- the same real object contributes multiple, overlapping
          windows across its lifetime, same as it already contributes
          multiple independent single-frame entries today).

Usage:
    # quick sanity check on a small subset before committing to a full build
    python tools/create_temporal_gt_database.py --limit 500

    # full build (nuScenes trainval, ~28k keyframes; expect a large output --
    # see the printed size estimate before letting this run to completion)
    python tools/create_temporal_gt_database.py
"""
import argparse
import os
import pickle
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, 'mmdetection3d'))

import mmcv  # noqa: E402
from mmcv.utils import build_from_cfg  # noqa: E402
from mmdet.datasets.builder import PIPELINES  # noqa: E402
from mmdet3d.core.bbox import box_np_ops  # noqa: E402

import plugin.vip3d  # noqa: E402  registers NuScenesTrackDatasetRadar + pipeline transforms
from plugin.vip3d.dataset import NuScenesTrackDatasetRadar  # noqa: E402
from plugin.vip3d.pipeline import _heading_local_to_global  # noqa: E402

CLASS_NAMES = ['car', 'truck', 'bus', 'trailer', 'motorcycle', 'bicycle', 'pedestrian']


def build_dataset(ann_file, data_root):
    modality = dict(use_lidar=True, use_camera=False, use_radar=False,
                     use_map=False, use_external=False)
    return NuScenesTrackDatasetRadar(
        ann_file=ann_file,
        data_root=data_root,
        classes=CLASS_NAMES,
        modality=modality,
        box_type_3d='LiDAR',
        test_mode=False,
        use_valid_flag=True,
        camera_types=None,
        pipeline_single=None,
        pipeline_post=None,
    )


def build_points_loader(file_client_args):
    load_cfg = dict(
        type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5,
        use_dim=[0, 1, 2, 3, 4], file_client_args=file_client_args)
    sweeps_cfg = dict(
        type='LoadPointsFromMultiSweeps', load_dim=5, sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4], file_client_args=file_client_args,
        pad_empty_sweeps=True, remove_close=True)
    loader = build_from_cfg(load_cfg, PIPELINES)
    sweeps = build_from_cfg(sweeps_cfg, PIPELINES)
    return lambda input_dict: sweeps(loader(input_dict))


def is_same_scene(dataset, i, j):
    """True iff frames i and j=i+1 of dataset.data_infos are the same real
    nuScenes scene. Calls the dataset's OWN scene-boundary heuristic
    (dataset.py's is_the_same_scene) with arguments chosen so it compares
    exactly this adjacent pair, instead of re-deriving the heuristic here --
    so this script can never silently drift from what the training loop
    itself considers a scene boundary."""
    assert j == i + 1
    return dataset.is_the_same_scene(i, j + 1, 0)


def crop_frame(input_dict, frame_idx, out_dir, data_root):
    """Crop+recenter every valid object's points in one keyframe, convert
    box+points to a GLOBAL anchor using this frame's own ego pose, and write
    each object's points to disk EXACTLY ONCE (Pass 2 references the path,
    it never re-writes the bytes -- a window spanning several frames must
    not duplicate the same on-disk point cloud once per window it appears
    in). Returns a list of per-object dicts (one list entry per valid
    object)."""
    points = input_dict['points'].tensor.numpy()  # [N, 5]: x,y,z,intensity,ring
    ann = input_dict['ann_info']
    boxes = ann['gt_bboxes_3d'].tensor.numpy()  # [M, 9]
    labels = ann['gt_labels_3d']
    names = ann['gt_names']
    instance_ids = ann['instance_inds']

    R = input_dict['l2g_r_mat']
    t = input_dict['l2g_t']
    ts = input_dict['timestamp']

    records = []
    if len(boxes) == 0:
        return records

    in_box = box_np_ops.points_in_rbbox(points[:, :3], boxes[:, :7])
    ctr_g_all = boxes[:, :3] @ R.T + t
    yaw_g_all = _heading_local_to_global(boxes[:, 6], R)

    for k in range(len(boxes)):
        if labels[k] < 0:
            continue  # class not in CLASS_NAMES
        pts_k = points[in_box[:, k]]
        pts_g = pts_k[:, :3] @ R.T + t
        pts_extra = pts_k[:, 3:]
        points_gc = np.concatenate([pts_g, pts_extra], axis=1).astype(np.float32)

        path = os.path.join(out_dir, f'{frame_idx}_{names[k]}_{instance_ids[k]}.bin')
        points_gc.tofile(path)

        records.append(dict(
            instance_id=int(instance_ids[k]), name=str(names[k]),
            box3d_lidar=boxes[k].astype(np.float32),
            ctr_g=ctr_g_all[k].astype(np.float32),
            yaw_g=float(yaw_g_all[k]),
            points_path=os.path.relpath(path, data_root),
            points_dim=pts_k.shape[1],
            num_points=int(len(points_gc)),
            timestamp=float(ts)))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='data/nuscenes/')
    parser.add_argument('--ann-file', default='data/nuscenes/nuscenes_tracking_infos_train.pkl')
    parser.add_argument('--out-dir', default='data/nuscenes/nuscenes_gt_database_temporal')
    parser.add_argument('--out-pkl', default='data/nuscenes/nuscenes_dbinfos_temporal_10sweeps_withvelo.pkl')
    parser.add_argument('--t-max', type=int, default=4)
    parser.add_argument('--limit', type=int, default=None,
                         help='only process the first N keyframes, for a quick sanity check before the full build')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    dataset = build_dataset(args.ann_file, args.data_root)
    points_loader = build_points_loader(dict(backend='disk'))

    n = len(dataset.data_infos)
    if args.limit is not None:
        n = min(n, args.limit)

    # ---- Pass 1: one pass over all keyframes, crop+recenter+globalize ----
    # Points for each (frame, object) are written to disk exactly once here.
    frame_records = [None] * n
    print(f'Pass 1: cropping objects from {n} keyframes...')
    total_points_bytes = 0
    for j in mmcv.track_iter_progress(list(range(n))):
        input_dict = dataset.get_data_info(j)
        input_dict = points_loader(input_dict)
        records = crop_frame(input_dict, j, args.out_dir, args.data_root)
        total_points_bytes += sum(r['num_points'] * r['points_dim'] * 4 for r in records)
        frame_records[j] = records

    # ---- Pass 2: window each (instance, start-frame) forward up to t_max ----
    # Pure bookkeeping over the metadata already built in Pass 1 -- no point
    # data is touched or re-written here, only path references are copied.
    print('Pass 2: assembling temporal windows...')
    all_db_infos = {name: [] for name in CLASS_NAMES}
    for i in mmcv.track_iter_progress(list(range(n))):
        for rec0 in frame_records[i]:
            frames = [dict(ctr_g=rec0['ctr_g'].tolist(), yaw_g=rec0['yaw_g'], dt=0.0,
                            points_path=rec0['points_path'], points_dim=rec0['points_dim'])]
            j = i
            while len(frames) < args.t_max:
                j_next = j + 1
                if j_next >= n or not is_same_scene(dataset, j, j_next):
                    break
                match = next((r for r in frame_records[j_next]
                              if r['instance_id'] == rec0['instance_id']), None)
                if match is None:
                    break
                frames.append(dict(
                    ctr_g=match['ctr_g'].tolist(), yaw_g=match['yaw_g'],
                    dt=match['timestamp'] - rec0['timestamp'],
                    points_path=match['points_path'], points_dim=match['points_dim']))
                j = j_next

            db_info = dict(
                name=rec0['name'],
                box3d_lidar=rec0['box3d_lidar'],
                num_points_in_gt=rec0['num_points'],
                frames=frames)
            all_db_infos[rec0['name']].append(db_info)

    for name, infos in all_db_infos.items():
        print(f'  {name}: {len(infos)} windows')
    print(f'Total points on disk: {total_points_bytes / 1e9:.2f} GB')

    with open(args.out_pkl, 'wb') as f:
        pickle.dump(all_db_infos, f)
    print(f'Wrote {args.out_pkl}')


if __name__ == '__main__':
    main()
