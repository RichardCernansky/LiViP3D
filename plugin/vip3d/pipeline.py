import os

import numpy as np
from mmcv.parallel import DataContainer as DC
import torch
from mmdet3d.core.bbox import BaseInstance3DBoxes
from mmdet3d.core.points import BasePoints
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import to_tensor
from mmdet3d.datasets.pipelines import DefaultFormatBundle
from mmdet3d.datasets.pipelines import (GlobalRotScaleTrans, RandomFlip3D,
                                        ObjectSample)
from mmdet3d.datasets.pipelines.dbsampler import DataBaseSampler
from mmdet3d.datasets.builder import OBJECTSAMPLERS
import mmcv
from nuscenes.utils.data_classes import RadarPointCloud


@OBJECTSAMPLERS.register_module()
class TemporalDataBaseSampler(DataBaseSampler):
    """DataBaseSampler that returns each sampled object's own multi-frame
    history (built by tools/create_temporal_gt_database.py) instead of a
    single box+points snapshot, so a pasted object can replay its real
    recorded appearance and motion ("LIFT: Learning 4D LiDAR Image Fusion
    Transformer", CVPR 2022, Sec 3.3) rather than a frozen or rigidly-
    extrapolated shape.

    Everything about WHICH objects get sampled and WHERE they're placed in
    the current scene is unchanged from DataBaseSampler -- sample_class_v2's
    BEV collision test and BatchSampler class-balancing only ever look at
    each candidate's frame-0 `box3d_lidar`, exactly as they do today, so
    those are inherited untouched. Only the points-loading tail of
    sample_all is overridden: instead of loading and flattening one
    snapshot per object, this loads and keeps each object's own `frames`
    sequence intact (still in the GLOBAL coordinates the database stores
    them in -- re-anchoring onto the current scene happens downstream, in
    TrackConsistentObjectSample, which is the only place with access to the
    training clip's own ego pose)."""

    def sample_all(self, gt_bboxes, gt_labels, img=None):
        sample_num_per_class = []
        for class_name, max_sample_num in zip(self.sample_classes,
                                               self.sample_max_nums):
            class_label = self.cat2label[class_name]
            sampled_num = int(max_sample_num -
                              np.sum([n == class_label for n in gt_labels]))
            sampled_num = np.round(self.rate * sampled_num).astype(np.int64)
            sample_num_per_class.append(sampled_num)

        sampled = []
        sampled_gt_bboxes = []
        avoid_coll_boxes = gt_bboxes

        for class_name, sampled_num in zip(self.sample_classes,
                                           sample_num_per_class):
            if sampled_num > 0:
                sampled_cls = self.sample_class_v2(class_name, sampled_num,
                                                    avoid_coll_boxes)
                sampled += sampled_cls
                if len(sampled_cls) > 0:
                    if len(sampled_cls) == 1:
                        sampled_gt_box = sampled_cls[0]['box3d_lidar'][np.newaxis, ...]
                    else:
                        sampled_gt_box = np.stack(
                            [s['box3d_lidar'] for s in sampled_cls], axis=0)
                    sampled_gt_bboxes += [sampled_gt_box]
                    avoid_coll_boxes = np.concatenate(
                        [avoid_coll_boxes, sampled_gt_box], axis=0)

        if len(sampled) == 0:
            return None

        sampled_gt_bboxes = np.concatenate(sampled_gt_bboxes, axis=0)
        gt_labels_3d = np.array(
            [self.cat2label[s['name']] for s in sampled], dtype=np.long)

        sequences = []
        for info in sampled:
            frames = []
            for fr in info['frames']:
                file_path = os.path.join(
                    self.data_root, fr['points_path']) if self.data_root else fr['points_path']
                points = self.points_loader(dict(pts_filename=file_path))['points']
                frames.append(dict(
                    dt=fr['dt'],
                    ctr_g=np.array(fr['ctr_g'], dtype=np.float32),
                    yaw_g=float(fr['yaw_g']),
                    points=points))
            sequences.append(frames)

        return dict(
            gt_labels_3d=gt_labels_3d,
            gt_bboxes_3d=sampled_gt_bboxes,
            sequences=sequences,
            group_ids=np.arange(gt_bboxes.shape[0],
                                 gt_bboxes.shape[0] + len(sampled)))


@PIPELINES.register_module()
class FormatBundle3DTrack(DefaultFormatBundle):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields for voxels,
    "gt_bboxes", "gt_labels", "gt_masks" and
    "gt_semantic_seg".
    These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    """

    def __init__(self, with_gt=True, with_label=True):
        super(FormatBundle3DTrack, self).__init__()
        self.with_gt = with_gt
        self.with_label = with_label

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        # Format 3D data
        if 'points' in results:
            points_cat = []
            for point in results['points']:
                assert isinstance(point, BasePoints)
                points_cat.append(point.tensor)
            # results['points'] = DC(torch.stack(points_cat, dim=0))
            results['points'] = DC(points_cat)

        if 'img' in results:
            imgs_list = results['img']
            imgs_cat_list = []
            for imgs_frame in imgs_list:
                imgs = [img.transpose(2, 0, 1) for img in imgs_frame]
                imgs = np.ascontiguousarray(np.stack(imgs, axis=0))
                imgs_cat_list.append(to_tensor(imgs))

            results['img'] = DC(torch.stack(imgs_cat_list, dim=0), stack=True)

        for key in [
            'proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels',
            'gt_labels_3d', 'attr_labels', 'pts_instance_mask',
            'pts_semantic_mask', 'centers2d', 'depths',
        ]:
            if key not in results:
                continue
            if isinstance(results[key], list):
                results[key] = DC([to_tensor(res) for res in results[key]])
            else:
                results[key] = DC(to_tensor(results[key]))
        if 'gt_bboxes_3d' in results:
            results['gt_bboxes_3d'] = DC(results['gt_bboxes_3d'],
                                         cpu_only=True)

        if 'instance_inds' in results:
            instance_inds = [torch.tensor(_t) for _t in results['instance_inds']]
            results['instance_inds'] = DC(instance_inds)

        keys = ['l2g_r_mat', 'l2g_t', 'radar']
        for key in keys:
            if key in results:
                results[key] = DC(torch.tensor(results[key], dtype=torch.float))

        for key in ['pred_matrix', 'polyline_spans', 'mapping', 'instance_idx_2_labels']:
            if key in results:
                results[key] = DC(results[key], cpu_only=True)

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(class_names={self.class_names}, '
        repr_str += f'with_gt={self.with_gt}, with_label={self.with_label})'
        return repr_str


@PIPELINES.register_module()
class InstanceRangeFilter(object):
    """Filter objects by the range.

    Args:
        point_cloud_range (list[float]): Point cloud range.
    """

    def __init__(self, point_cloud_range):
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)
        self.bev_range = self.pcd_range[[0, 1, 3, 4]]

    def __call__(self, input_dict):
        """Call function to filter objects by the range.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d' \
                keys are updated in the result dict.
        """
        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        gt_labels_3d = input_dict['gt_labels_3d']
        instance_inds = input_dict['ann_info']['instance_inds']
        mask = gt_bboxes_3d.in_range_bev(self.bev_range)
        gt_bboxes_3d = gt_bboxes_3d[mask]
        # mask is a torch tensor but gt_labels_3d is still numpy array
        # using mask to index gt_labels_3d will cause bug when
        # len(gt_labels_3d) == 1, where mask=1 will be interpreted
        # as gt_labels_3d[1] and cause out of index error
        gt_labels_3d = gt_labels_3d[mask.numpy().astype(np.bool)]
        instance_inds = instance_inds[mask.numpy().astype(np.bool)]

        # limit rad to [-pi, pi]
        gt_bboxes_3d.limit_yaw(offset=0.5, period=2 * np.pi)
        input_dict['gt_bboxes_3d'] = gt_bboxes_3d
        input_dict['gt_labels_3d'] = gt_labels_3d
        input_dict['ann_info']['instance_inds'] = instance_inds

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(point_cloud_range={self.pcd_range.tolist()})'
        return repr_str


@PIPELINES.register_module()
class TrackConsistentGlobalRotScaleTrans(GlobalRotScaleTrans):
    """GlobalRotScaleTrans that reuses one random draw across every frame
    of a training sample.

    NuScenesTrackDatasetRadar.prepare_train_data runs pipeline_single
    independently per frame of the multi-frame tracking window. Drawing a
    fresh random rotation/scale per frame would desync object positions
    across the window, which the QIM/memory-bank tracking head assumes are
    coherent. `input_dict['aug_state']` is a dict shared by the caller
    across all frames of one sample: the first frame draws and caches the
    values, later frames reuse them.
    """

    def _random_scale(self, input_dict):
        aug_state = input_dict.get('aug_state')
        if aug_state is not None and 'pcd_scale_factor' in aug_state:
            input_dict['pcd_scale_factor'] = aug_state['pcd_scale_factor']
            return
        super()._random_scale(input_dict)
        if aug_state is not None:
            aug_state['pcd_scale_factor'] = input_dict['pcd_scale_factor']

    def _rot_bbox_points(self, input_dict):
        aug_state = input_dict.get('aug_state')
        if aug_state is not None and 'noise_rotation' in aug_state:
            noise_rotation = aug_state['noise_rotation']
        else:
            noise_rotation = np.random.uniform(self.rot_range[0],
                                               self.rot_range[1])
            if aug_state is not None:
                aug_state['noise_rotation'] = noise_rotation

        if len(input_dict['bbox3d_fields']) == 0:
            rot_mat_T = input_dict['points'].rotate(noise_rotation)
            input_dict['pcd_rotation'] = rot_mat_T
            return

        for key in input_dict['bbox3d_fields']:
            if len(input_dict[key].tensor) != 0:
                points, rot_mat_T = input_dict[key].rotate(
                    noise_rotation, input_dict['points'])
                input_dict['points'] = points
                input_dict['pcd_rotation'] = rot_mat_T


@PIPELINES.register_module()
class SyncEgoPoseToAugmentation(object):
    """Re-express `l2g_r_mat` / `l2g_t` for the augmented scene.

    GlobalRotScaleTrans and RandomFlip3D rewrite `points` and `gt_bboxes_3d`
    in each frame's LOCAL coordinates but leave the ego pose describing the
    scene as it was BEFORE augmentation. That pose is how the model relates
    consecutive frames (vip3d.py:379-381, the track-propagation step), so
    leaving it stale makes a world-static object appear to jump between
    frames.

    The rotation is applied about the LOCAL origin -- i.e. about the ego
    vehicle -- so in map terms each frame's scene is rotated about a
    different point, because ego moves. Sharing one random draw across the
    clip (which TrackConsistentGlobalRotScaleTrans already does) does not
    help: same angle, different pivot. Measured spread across a 3-frame clip
    with the config's own ranges reaches ~37m, against a 2m association
    threshold.

    `l2g_t` is deliberately NOT rotated or flipped: it records where the ego
    vehicle physically was, which no relabelling of axes can change. Only
    scale touches it, since scaling the scene rescales all distances
    including the ego trajectory.

    Must run AFTER every geometric augmentation (it needs their composed
    effect) and BEFORE Collect3D (which discards `pcd_rotation` and the flip
    flags). Registering it in train_pipeline also makes it hard to enable
    augmentation and silently forget the correction.
    """

    def __call__(self, input_dict):
        if 'l2g_r_mat' not in input_dict:
            return input_dict
        assert 'lidar2img' not in input_dict, (
            'SyncEgoPoseToAugmentation does not correct camera projection '
            'matrices -- point-cloud augmentation must not be combined with '
            'the image/fusion configs.')

        # `pcd_rotation` is mmdet3d's rot_mat_T: already transposed for
        # post-multiplication (points transform as p @ rot_mat_T), which is
        # exactly the form needed here. It is absent when a frame carries no
        # boxes, since _rot_bbox_points then rotates nothing at all.
        aug = np.asarray(
            input_dict.get('pcd_rotation', np.eye(3)), dtype=np.float32)
        if input_dict.get('pcd_horizontal_flip'):
            aug = aug @ np.diag([1., -1., 1.]).astype(np.float32)   # y -> -y
        if input_dict.get('pcd_vertical_flip'):
            aug = aug @ np.diag([-1., 1., 1.]).astype(np.float32)   # x -> -x

        input_dict['l2g_r_mat'] = (
            np.asarray(input_dict['l2g_r_mat'], dtype=np.float32) @ aug)

        scale = float(input_dict.get('pcd_scale_factor', 1.0))
        if scale != 1.0:
            input_dict['l2g_t'] = (
                np.asarray(input_dict['l2g_t'], dtype=np.float32) * scale)
        return input_dict

    def __repr__(self):
        return self.__class__.__name__ + '()'


@PIPELINES.register_module()
class TrackConsistentRandomFlip3D(RandomFlip3D):
    """RandomFlip3D that reuses one random flip decision across every frame
    of a training sample. See TrackConsistentGlobalRotScaleTrans for why.

    Must be used with sync_2d=False since no image flip is coordinated
    here (images aren't loaded in the LiDAR-only stage this is meant for).
    """

    def __call__(self, input_dict):
        assert not self.sync_2d, \
            'TrackConsistentRandomFlip3D requires sync_2d=False'
        aug_state = input_dict.get('aug_state')
        if aug_state is not None:
            if 'pcd_horizontal_flip' in aug_state:
                input_dict['pcd_horizontal_flip'] = aug_state[
                    'pcd_horizontal_flip']
            if 'pcd_vertical_flip' in aug_state:
                input_dict['pcd_vertical_flip'] = aug_state[
                    'pcd_vertical_flip']
        result = super().__call__(input_dict)
        if aug_state is not None:
            aug_state.setdefault('pcd_horizontal_flip',
                                 input_dict['pcd_horizontal_flip'])
            aug_state.setdefault('pcd_vertical_flip',
                                 input_dict['pcd_vertical_flip'])
        return result


def _heading_local_to_global(yaw_l, R):
    """Rotate a batch of local heading angles (about +z) into the frame
    whose local->parent rotation matrix is R, by rotating each box's local
    heading unit vector through R and re-deriving yaw from the result's
    xy-projection (matches nuscenes-devkit's own quaternion_yaw pattern;
    ignores any out-of-plane component from ego roll/pitch, same as every
    other BEV box-yaw computation in this codebase)."""
    c, s = np.cos(yaw_l), np.sin(yaw_l)
    v_l = np.stack([c, s, np.zeros_like(c)], axis=-1)
    v_g = v_l @ R.T
    return np.arctan2(v_g[:, 1], v_g[:, 0])


def _heading_global_to_local(yaw_g, R):
    """Inverse of _heading_local_to_global: parent-frame heading -> local
    heading, using R's own transpose (R is orthonormal), matching the
    global->local point convention used throughout this codebase
    (`p_local = (p_global - t) @ R`, see vip3d.py)."""
    c, s = np.cos(yaw_g), np.sin(yaw_g)
    v_g = np.stack([c, s, np.zeros_like(c)], axis=-1)
    v_l = v_g @ R
    return np.arctan2(v_l[:, 1], v_l[:, 0])


def _rot_z(theta):
    """3x3 rotation matrix for a rotation of `theta` radians about +z,
    written in the same `p_new = p_old @ R.T + t` form used throughout this
    file so it composes directly with _heading_local_to_global-style code."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# Ray-bin width for the occlusion filter below. Finer than either is the
# safe direction (it converges toward "keep everything," not toward wrongly
# dropping real returns), so these are picked finer than the actual
# nuScenes 32-beam LiDAR's real resolution rather than matched to it --
# matching the real per-beam elevation angles exactly would need a fixed
# 32-value lookup table instead of a uniform bin, which this doesn't do.
_OCCLUSION_AZIMUTH_BIN = np.deg2rad(0.15)
_OCCLUSION_ELEVATION_BIN = np.deg2rad(0.5)


# Pasted objects need an `instance_inds` entry, and it has two hard
# constraints:
#
#   1. It must never collide with a real nuScenes instance index. loss.py's
#      `obj_idx_to_gt_idx` is a plain dict comprehension keyed on the id, so a
#      collision silently rebinds a real track onto a pasted box -- no error,
#      just wrong targets.
#   2. It must be >= 0. The tracking code tests the SIGN of `obj_idxes` to mean
#      "is this a real track" -- 16 such tests across loss.py, qim.py and
#      vip3d.py. A negative id reads everywhere as "no track here", so a slot
#      stamped with one is neither continued (loss.py:296 `>= 0`) nor eligible
#      for rebirth (loss.py:304 `== -1`); it just burns a query slot for the
#      rest of the clip while being supervised as background.
#
# Real indices are nuScenes instance-table rows (~64k across trainval), so this
# base clears them by three orders of magnitude.
PASTE_SENTINEL_BASE = 10_000_000


def is_paste_sentinel(instance_inds):
    """Mask of which instance indices refer to pasted objects."""
    return np.asarray(instance_inds) >= PASTE_SENTINEL_BASE


def _ray_bins(xyz):
    """(bin key, range) per point, for points already in one frame's own
    local coordinates -- ego sits at the origin, so no pose is needed."""
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    az = np.floor(np.arctan2(y, x) / _OCCLUSION_AZIMUTH_BIN).astype(np.int64)
    el = np.floor(np.arctan2(z, np.sqrt(x ** 2 + y ** 2))
                  / _OCCLUSION_ELEVATION_BIN).astype(np.int64)
    return az * 4096 + (el + 2048), r      # one int key; |el| < 2048 by construction


def _visible_paste_mask(paste_xyz, real_xyz):
    """Mask over PASTED points: False where real geometry lies nearer along
    the same ray, i.e. where the sensor could not have seen the paste.

    This is LIFT Sec 3.3's "filter out the occluded point" of the pasted
    pattern (following PointAugmenting), and it is deliberately
    ONE-DIRECTIONAL: real points are read to build the depth reference and
    are never removed, so no choice of bin resolution or sweep count can
    cost real data. The reverse direction -- a paste shadowing real geometry
    behind it -- is not done: it is the part that would delete real points,
    it is not what the paper describes, and the interior case is already
    handled by remove_points_in_boxes.
    """
    if len(paste_xyz) == 0 or len(real_xyz) == 0:
        return np.ones(len(paste_xyz), dtype=bool)

    real_key, real_r = _ray_bins(real_xyz)
    paste_key, paste_r = _ray_bins(paste_xyz)

    order = np.argsort(real_r)                       # nearest real return per bin
    keys, first = np.unique(real_key[order], return_index=True)
    nearest = real_r[order][first]

    idx = np.clip(np.searchsorted(keys, paste_key), 0, len(keys) - 1)
    occluded = (keys[idx] == paste_key) & (nearest[idx] < paste_r)
    return ~occluded


@PIPELINES.register_module()
class TrackConsistentObjectSample(ObjectSample):
    """ObjectSample (copy-paste GT sampling) that replays the same pasted
    objects across every frame of a training sample, and keeps
    `ann_info['instance_inds']` in sync with them.

    NuScenesTrackDatasetRadar reads `example['instance_inds']` from
    `ann_info` after the pipeline runs (see prepare_train_data_single), so
    a plain ObjectSample would silently desync gt_bboxes_3d/gt_labels_3d
    (now longer) from instance_inds (still the pre-paste length) and crash
    or misalign the tracking loss downstream.

    Frame-independent pasting (drawing a fresh db_sampler call per frame)
    would give each pasted object a lifetime of exactly one frame, so the
    tracker only ever sees it as a one-off, never-continued detection --
    never gets to practice continuing a track for it. Instead, the first
    frame of the sample draws the paste once (db_sampler.sample_all,
    expected to be a TemporalDataBaseSampler) and replays each pasted
    object's own REAL recorded multi-frame trajectory (built by
    tools/create_temporal_gt_database.py from consecutive real keyframes of
    that object in its source scene) into every frame of the clip -- a
    moving source object keeps moving, a parked one stays parked, both
    coming from what was actually recorded rather than an assumption or a
    fabricated constant. Each object's trajectory is re-anchored once, at
    draw time, so its first frame lands exactly on the collision-tested
    spot sample_class_v2 chose for it in this scene, then the SAME rigid
    delta carries the rest of its real relative motion along with it. If
    the training clip runs longer than an object's available real history,
    its last real frame is held (there is no more real data to draw on).
    Every frame's replay is expressed via that frame's own `l2g_r_mat`/
    `l2g_t`, so this is decoupled from ego's own motion throughout --
    naively replaying one frame's local coordinates verbatim across a clip
    where ego moves would instead make every pasted object silently travel
    at exactly ego's own velocity, which is what this avoids.

    Each pasted object gets one shared instance id (>= PASTE_SENTINEL_BASE,
    see that constant for why the sign matters) reused at every frame of the
    clip, so the tracker treats it as a normal track: matched while it's
    "present" in the sample, then marked disappeared once the sample ends,
    since the id never reappears in a later sample.

    Also supports being turned off at runtime (`enabled = False`) so an
    epoch-based hook can implement the "fade" strategy: fade the GT
    sampling out for the last few epochs of training since pasted objects
    sit in physically implausible spots and can distort the real data
    distribution if used for the whole schedule.
    """

    _next_sentinel = PASTE_SENTINEL_BASE
    _cache_key = 'object_sample'

    def __init__(self, *args, min_paste_points=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.enabled = True
        # A pasted object left with no visible points is dropped entirely
        # rather than supervised as a box with no evidence under it.
        #
        # Deliberately 1, not the db_sampler's filter_by_min_points=5: the
        # dataset itself keeps real GT boxes down to 0 lidar points
        # (use_valid_flag only needs lidar+radar > 0), and 58% of real
        # bicycles / 57% of real pedestrians have fewer than 5 -- their
        # median is 4. Anything above 1 would hold pasted objects to a
        # stricter standard than the real annotations they imitate, and
        # would bite hardest on exactly the sparse rare classes this
        # augmentation exists to boost.
        self.min_paste_points = min_paste_points

    def _draw_and_cache(self, input_dict, aug_state):
        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        gt_labels_3d = input_dict['gt_labels_3d']
        sampled_dict = self.db_sampler.sample_all(                     # frame 0 only; collision test sees only THIS frame
            gt_bboxes_3d.tensor.numpy(), gt_labels_3d, img=None)

        if sampled_dict is None:
            if aug_state is not None:
                aug_state[self._cache_key] = None                      # cache the miss so later frames don't redraw
            return None

        boxes_l0 = np.nan_to_num(sampled_dict['gt_bboxes_3d'], nan=0.0)  # ~0.3% of DB entries have NaN vx/vy
        n_sampled = len(sampled_dict['gt_labels_3d'])

        sentinels = np.arange(                                         # one shared id per pasted object, reused at
            TrackConsistentObjectSample._next_sentinel,                 # every frame so the tracker follows it
            TrackConsistentObjectSample._next_sentinel + n_sampled)
        TrackConsistentObjectSample._next_sentinel += n_sampled

        # Re-anchor each object's OWN real trajectory (recorded in global
        # coordinates in its source scene -- see
        # tools/create_temporal_gt_database.py and TemporalDataBaseSampler)
        # onto the collision-tested spot sample_class_v2 chose for it in
        # THIS scene: compute the one rigid SE(2) delta that carries the
        # object's own frame-0 global position/heading onto its target
        # frame-0 global position/heading, then apply that SAME delta to
        # every frame of its stored trajectory (box AND points). This is
        # what lets the object replay its real relative motion while still
        # starting exactly where the collision test placed it.
        R0 = input_dict['l2g_r_mat']                                  # this clip's frame-0 pose (still valid; aug runs later)
        t0 = input_dict['l2g_t']
        target_ctr_g0 = boxes_l0[:, :3] @ R0.T + t0                   # source-scene coords reused as local here -> global map
        target_yaw_g0 = _heading_local_to_global(boxes_l0[:, 6], R0)  # same for heading

        objects = []
        for k in range(n_sampled): # go through all the vehicles to be pasted in
            seq = sampled_dict['sequences'][k]
            db_ctr_g0 = seq[0]['ctr_g']                               # where it REALLY was, on the map
            db_yaw_g0 = seq[0]['yaw_g']
            delta_yaw = float(target_yaw_g0[k] - db_yaw_g0)           # rotation needed to face the target way
            Rd = _rot_z(delta_yaw)                                    # rot matrix
            delta_t = target_ctr_g0[k] - db_ctr_g0 @ Rd.T             # translation that lands frame 0 exactly on target

            frames = []
            # bring the sequence box points into the target spot, preserving the object's own relative motion across its recorded frames
            for fr in seq:
                ctr_g = fr['ctr_g'] @ Rd.T + delta_t                  # same rigid delta on EVERY frame -> motion preserved
                yaw_g = fr['yaw_g'] + delta_yaw                       # Rd is pure-Z, so headings just add
                pts_tensor = fr['points'].tensor.numpy()
                pts_xyz_g = pts_tensor[:, :3] @ Rd.T + delta_t        # translate points into the target spot, saved in global coords
            
                frames.append(dict(                                    # paste in
                    dt=fr['dt'], ctr_g=ctr_g, yaw_g=yaw_g,
                    pts_xyz_g=pts_xyz_g, pts_extra=pts_tensor[:, 3:],
                    points_type=type(fr['points']),
                    points_dim=fr['points'].points_dim,
                    attribute_dims=fr['points'].attribute_dims))
            objects.append(dict(dims=boxes_l0[k, 3:6].copy(), frames=frames))

        cache = dict(
            objects=objects,
            gt_labels_3d=sampled_dict['gt_labels_3d'].copy(),
            instance_sentinels=sentinels.copy(),
            t0_train=input_dict['timestamp'],
        )
        if aug_state is not None:
            aug_state[self._cache_key] = cache
        return self._reproject_to_frame(input_dict, cache)

    @staticmethod
    def _velocity_at(seq, i, Rk):
        """vx, vy for recorded frame `seq[i]`, expressed in the frame whose
        local->global rotation is Rk.

        Taken by finite-differencing the object's own recorded trajectory --
        the motion is already there, it just has to be read off. Forward
        difference where a later frame exists, backward otherwise; zero only
        when the object has a single recorded frame and its motion is
        genuinely unknown.

        Rotated but NOT translated: a velocity is a direction, so the
        translation cancels out of the position difference. That also matches
        how nuScenes stores GT velocity in the first place -- a global-frame
        quantity re-expressed in the local frame's axes (see
        mmdetection3d/tools/data_converter/nuscenes_converter.py:234-245).
        """
        if i + 1 < len(seq):
            a, b = seq[i], seq[i + 1]
        elif i > 0:
            a, b = seq[i - 1], seq[i]
        else:
            return np.zeros(2, dtype=np.float32)

        span = b['dt'] - a['dt']
        if span <= 0:
            return np.zeros(2, dtype=np.float32)
        v_g = (b['ctr_g'] - a['ctr_g']) / span
        return ((v_g[None, :] @ Rk)[0, :2]).astype(np.float32)

    def _reproject_to_frame(self, input_dict, cache):
        Rk = input_dict['l2g_r_mat']
        tk = input_dict['l2g_t']
        training_dt = input_dict['timestamp'] - cache['t0_train']

        boxes_l = []
        pts_xyz_l_list = []
        pts_extra_list = []
        for obj in cache['objects']:
            # Nearest recorded frame to this training frame's time offset;
            # naturally clamps to the last real frame once training_dt runs
            # past the object's available history (dt's are increasing, so
            # the last frame is the closest match beyond that point) --
            # there is no more real data to draw on past there.
            seq = obj['frames']
            i = min(range(len(seq)), key=lambda j: abs(seq[j]['dt'] - training_dt)) # index of the frame closest to the current training frame's time offset
            frame = seq[i]
            ctr_l = (frame['ctr_g'][None, :] - tk) @ Rk # bring the object's global center into THIS frame's local coordinates
            yaw_l = _heading_global_to_local(np.array([frame['yaw_g']]), Rk) # bring the object's global heading into THIS frame's local coordinates
            vel_l = self._velocity_at(seq, i, Rk) # finite-difference the recorded trajectory -> vx, vy in THIS frame
            boxes_l.append(np.concatenate(
                [ctr_l[0], obj['dims'], yaw_l, vel_l]).astype(np.float32))

            pts_xyz_l_list.append(((frame['pts_xyz_g'] - tk) @ Rk).astype(np.float32))
            pts_extra_list.append(frame['pts_extra'])

        boxes_l = np.stack(boxes_l, axis=0)
        pts_xyz_l = np.concatenate(pts_xyz_l_list, axis=0)
        pts_extra = np.concatenate(pts_extra_list, axis=0)
        pts_tensor_l = np.concatenate([pts_xyz_l, pts_extra], axis=1).astype(np.float32)

        # points_type/points_dim/attribute_dims are fixed by the dataset's
        # own points_loader config, so any one frame's metadata describes
        # the whole concatenated tensor.
        ref = cache['objects'][0]['frames'][0]
        points = ref['points_type'](
            torch.from_numpy(pts_tensor_l),
            points_dim=ref['points_dim'],
            attribute_dims=ref['attribute_dims'])

        return dict(
            gt_bboxes_3d=boxes_l,
            gt_labels_3d=cache['gt_labels_3d'],
            points=points,
            # how many of `points` belong to each object, in order -- lets
            # _apply_paste filter occluded points per object and drop any
            # object left without enough evidence to be detectable.
            point_counts=np.array([len(p) for p in pts_xyz_l_list], dtype=np.int64),
            instance_sentinels=cache['instance_sentinels'])

    def _apply_paste(self, input_dict, paste):
        """_apply_paste(scene, paste):

    1. carve out the box interiors            (pre-existing, unchanged)
       real_points ← remove_points_in_boxes(real_points, paste_boxes)

    2. build a depth reference from REAL points only
       for each real point:
           bin  ← (azimuth, elevation) quantised
           keep the MINIMUM range seen in each bin
       →  "how close is real geometry along this ray"

    3. test each PASTED point against it
       occluded  ⟺  its bin has real geometry AND that geometry is nearer
       visible   ← not occluded
       ⚠ real points are only ever READ here, never removed

    4. per object (not over the flat cloud):
       survivors ← its visible points
       if len(survivors) < min_paste_points:
            drop the object entirely — box, label, sentinel and all
       else:
            keep it with only its visible points

    5. if nothing survived → leave the frame untouched
       else concatenate the survivors into the scene
        """
        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        gt_labels_3d = input_dict['gt_labels_3d']
        points = input_dict['points']

        # Carve room for the pasted boxes first, so the depth reference below
        # doesn't include real points the paste is about to replace.
        points = self.remove_points_in_boxes(points, paste['gt_bboxes_3d'])

        # Drop pasted points the sensor could not have seen, then keep only
        # the objects still carrying enough evidence to be detectable.
        # Filtering per object (rather than over the flat cloud) is what lets
        # a stripped object take its box and label with it -- a GT box with
        # no points is worse than no paste at all.
        paste_pts = paste['points'].tensor.numpy()
        visible = _visible_paste_mask(paste_pts[:, :3], points.tensor[:, :3].numpy())

        keep_obj, kept_pts, off = [], [], 0
        for k, count in enumerate(paste['point_counts']):
            mask = visible[off:off + count]
            if int(mask.sum()) >= self.min_paste_points:
                keep_obj.append(k)
                kept_pts.append(paste_pts[off:off + count][mask])
            off += count

        if not keep_obj:
            return                                     # nothing survived; leave the frame untouched

        keep_obj = np.asarray(keep_obj, dtype=np.int64)
        kept_pts = np.concatenate(kept_pts, axis=0)

        gt_labels_3d = np.concatenate(
            [gt_labels_3d, paste['gt_labels_3d'][keep_obj]], axis=0)
        gt_bboxes_3d = gt_bboxes_3d.new_box(
            np.concatenate(
                [gt_bboxes_3d.tensor.numpy(), paste['gt_bboxes_3d'][keep_obj]]))
        points = points.cat(
            [paste['points'].new_point(torch.from_numpy(kept_pts)), points])

        input_dict['gt_bboxes_3d'] = gt_bboxes_3d
        input_dict['gt_labels_3d'] = gt_labels_3d.astype(np.long)
        input_dict['points'] = points

        if 'ann_info' in input_dict:
            existing = input_dict['ann_info']['instance_inds']
            input_dict['ann_info'] = dict(input_dict['ann_info'])
            input_dict['ann_info']['instance_inds'] = np.concatenate(
                [existing,
                 paste['instance_sentinels'][keep_obj].astype(existing.dtype)])

    def __call__(self, input_dict):
        if not self.enabled:
            return input_dict

        aug_state = input_dict.get('aug_state')

        if aug_state is not None and self._cache_key in aug_state:
            cached = aug_state[self._cache_key]
            if cached is None:
                return input_dict
            paste = self._reproject_to_frame(input_dict, cached)
        else:
            paste = self._draw_and_cache(input_dict, aug_state)
            if paste is None:
                return input_dict

        self._apply_paste(input_dict, paste)
        return input_dict


@PIPELINES.register_module()
class ScaleMultiViewImage3D(object):
    """Random scale the image
    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",
    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, scale=0.75):
        self.scale = scale

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
            'img': list of imgs
            'lidar2img' (list of 4x4 array)
            'intrinsic' (list of 4x4 array)
            'extrinsic' (list of 4x4 array)
        Returns:
            dict: Updated result dict.
        """
        rand_scale = self.scale
        img_shape = results['img_shape'][0]
        y_size = int((img_shape[0] * rand_scale) // 32) * 32
        x_size = int((img_shape[1] * rand_scale) // 32) * 32
        y_scale = y_size * 1.0 / img_shape[0]
        x_scale = x_size * 1.0 / img_shape[1]
        scale_factor = np.eye(4)
        scale_factor[0, 0] *= x_scale
        scale_factor[1, 1] *= y_scale
        for key in results.get('img_fields', ['img']):
            result_img = [mmcv.imresize(img, (x_size, y_size), return_scale=False) for img in results[key]]
            results[key] = result_img
            lidar2img = [scale_factor @ l2i for l2i in results['lidar2img']]
            results['lidar2img'] = lidar2img

        results['img_shape'] = [img.shape for img in result_img]
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        return repr_str

@PIPELINES.register_module()
class ResizeMultiViewKeepRatio(object):
    """
    Resize all camera images by a single scale factor (keep aspect ratio),
    and update intrinsics/cam2img/lidar2img accordingly.

    Args:
        scale (float | tuple | list):
            - float -> uniform scale factor (e.g., 0.75)
            - tuple (w, h) -> target long/short size like MMCV img_scale
        keep_ratio (bool): keep aspect ratio (True by default)
        random_range (tuple[float,float] | None): if set, sample scale ~ U(a,b)
    """
    def __init__(self, scale=0.75, keep_ratio=True, random_range=None):
        self.scale = scale
        self.keep_ratio = keep_ratio
        self.random_range = random_range

    def _pick_scale(self, h, w):
        if self.random_range is not None:
            s = np.random.uniform(self.random_range[0], self.random_range[1])
            return s
        if isinstance(self.scale, (int, float)):
            return float(self.scale)
        tgt_w, tgt_h = self.scale
        if self.keep_ratio:
            return min(tgt_w / float(w), tgt_h / float(h))
        else:
            return (tgt_w / float(w), tgt_h / float(h))

    def __call__(self, results):
        assert 'img' in results, "Expect multiview images in results['img']"
        imgs = results['img']
        h, w = imgs[0].shape[:2]

        s = self._pick_scale(h, w)

        if isinstance(s, tuple):
            sx, sy = s[0], s[1]
            new_w, new_h = int(round(w * sx)), int(round(h * sy))
        else:
            sx = sy = float(s)
            new_w, new_h = int(round(w * s)), int(round(h * s))

        resized = [mmcv.imresize(im, (new_w, new_h), return_scale=False) for im in imgs]
        results['img'] = resized

        S = np.eye(4, dtype=np.float32)
        S[0, 0] = sx
        S[1, 1] = sy

        if 'lidar2img' in results:
            results['lidar2img'] = [S @ M for M in results['lidar2img']]

        for k in ['cam2img', 'camera_intrinsics', 'intrinsic']:
            if k in results:
                new_list = []
                for K in results[k]:
                    K = np.array(K, dtype=np.float32)
                    if K.shape == (3, 3):
                        K = K.copy()
                        K[0, 0] *= sx; K[1, 1] *= sy
                        K[0, 2] *= sx; K[1, 2] *= sy
                    elif K.shape == (4, 4):
                        K = S @ K
                    new_list.append(K)
                results[k] = new_list

        results['ori_shape'] = results.get('ori_shape', (h, w, 3))
        results['img_shape'] = [(new_h, new_w, 3) for _ in resized]
        results['scale_factor'] = (sx, sy, sx, sy)

        return results

    def __repr__(self):
        return (f"{self.__class__.__name__}(scale={self.scale}, "
                f"keep_ratio={self.keep_ratio}, random_range={self.random_range})")


@PIPELINES.register_module()
class LoadRadarPointsMultiSweeps(object):
    """Load radar points from multiple sweeps.

    This is usually used for nuScenes dataset to utilize previous sweeps.

    Args:
        sweeps_num (int): Number of sweeps. Defaults to 10.
        load_dim (int): Dimension number of the loaded points. Defaults to 5.
        use_dim (list[int]): Which dimension to use. Defaults to [0, 1, 2, 4].
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
        pad_empty_sweeps (bool): Whether to repeat keyframe when
            sweeps is empty. Defaults to False.
        remove_close (bool): Whether to remove close points.
            Defaults to False.
        test_mode (bool): If test_model=True used for testing, it will not
            randomly sample sweeps but select the nearest N frames.
            Defaults to False.
    """

    def __init__(self,
                 load_dim=18,
                 use_dim=[0, 1, 2, 3, 4],
                 sweeps_num=3,
                 file_client_args=dict(backend='disk'),
                 max_num=300,
                 pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                 test_mode=False):
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.sweeps_num = sweeps_num
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.max_num = max_num
        self.test_mode = test_mode
        self.pc_range = pc_range

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
            [N, 18]
        """
        radar_obj = RadarPointCloud.from_file(pts_filename)

        # [18, N]
        points = radar_obj.points

        return points.transpose().astype(np.float32)

    def _pad_or_drop(self, points):
        '''
        points: [N, 18]
        '''

        num_points = points.shape[0]

        if num_points == self.max_num:
            masks = np.ones((num_points, 1),
                            dtype=points.dtype)

            return points, masks

        if num_points > self.max_num:
            points = np.random.permutation(points)[:self.max_num, :]
            masks = np.ones((self.max_num, 1),
                            dtype=points.dtype)

            return points, masks

        if num_points < self.max_num:
            zeros = np.zeros((self.max_num - num_points, points.shape[1]),
                             dtype=points.dtype)
            masks = np.ones((num_points, 1),
                            dtype=points.dtype)

            points = np.concatenate((points, zeros), axis=0)
            masks = np.concatenate((masks, zeros.copy()[:, [0]]), axis=0)

            return points, masks

    def __call__(self, results):
        """Call function to load multi-sweep point clouds from files.

        Args:
            results (dict): Result dict containing multi-sweep point cloud \
                filenames.

        Returns:
            dict: The result dict containing the multi-sweep points data. \
                Added key and value are described below.

                - points (np.ndarray | :obj:`BasePoints`): Multi-sweep point \
                    cloud arrays.
        """
        radars_dict = results['radar']

        points_sweep_list = []
        for key, sweeps in radars_dict.items():
            if len(sweeps) < self.sweeps_num:
                idxes = list(range(len(sweeps)))
            else:
                idxes = list(range(self.sweeps_num))

            ts = sweeps[0]['timestamp'] * 1e-6
            for idx in idxes:
                sweep = sweeps[idx]

                points_sweep = self._load_points(sweep['data_path'])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)

                timestamp = sweep['timestamp'] * 1e-6
                time_diff = ts - timestamp
                time_diff = np.ones((points_sweep.shape[0], 1)) * time_diff

                # velocity compensated by the ego motion in sensor frame
                velo_comp = points_sweep[:, 8:10]
                velo_comp = np.concatenate(
                    (velo_comp, np.zeros((velo_comp.shape[0], 1))), 1)
                velo_comp = velo_comp @ sweep['sensor2lidar_rotation'].T
                velo_comp = velo_comp[:, :2]

                # velocity in sensor frame
                velo = points_sweep[:, 6:8]
                velo = np.concatenate(
                    (velo, np.zeros((velo.shape[0], 1))), 1)
                velo = velo @ sweep['sensor2lidar_rotation'].T
                velo = velo[:, :2]

                points_sweep[:, :3] = points_sweep[:, :3] @ sweep[
                    'sensor2lidar_rotation'].T
                points_sweep[:, :3] += sweep['sensor2lidar_translation']

                points_sweep_ = np.concatenate(
                    [points_sweep[:, :6], velo,
                     velo_comp, points_sweep[:, 10:],
                     time_diff], axis=1)
                points_sweep_list.append(points_sweep_)

        points = np.concatenate(points_sweep_list, axis=0)

        points = points[:, self.use_dim]

        points[:, 0:1] = (points[:, 0:1] - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
        points[:, 1:2] = (points[:, 1:2] - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])
        points[:, 2:3] = (points[:, 2:3] - self.pc_range[2]) / (self.pc_range[5] - self.pc_range[2])

        if self.max_num > 0:
            points, mask = self._pad_or_drop(points)
        else:
            mask = np.ones((points.shape[0], points.shape[1]),
                           dtype=points.dtype)

        points = np.concatenate((points, mask), axis=-1)

        results['radar'] = points.astype(np.float32)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return f'{self.__class__.__name__}(sweeps_num={self.sweeps_num})'


@PIPELINES.register_module()
class PadMultiViewImage(object):
    """Pad the multi-view image.
    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",
    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        if self.size is not None:
            padded_img = [mmcv.impad(
                img, shape=self.size, pad_val=self.pad_val) for img in results['img']]
        elif self.size_divisor is not None:
            padded_img = [mmcv.impad_to_multiple(
                img, self.size_divisor, pad_val=self.pad_val) for img in results['img']]
        results['img_shape'] = [img.shape for img in results['img']]
        results['img'] = padded_img
        results['pad_shape'] = [img.shape for img in padded_img]
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str


@PIPELINES.register_module()
class NormalizeMultiviewImage(object):
    """Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        results['img'] = [mmcv.imnormalize(
            img, self.mean, self.std, self.to_rgb) for img in results['img']]
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str
