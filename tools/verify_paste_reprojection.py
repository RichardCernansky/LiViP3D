"""Standalone geometry verifier for TrackConsistentObjectSample's temporal
replay (LIFT-style GT-paste: a pasted object replays its own real recorded
multi-frame trajectory instead of a frozen or fabricated one).

Run before training any config that uses TrackConsistentObjectSample with a
TemporalDataBaseSampler:
    python tools/verify_paste_reprojection.py

Checks, in order:
  A0. Re-anchoring lands frame 0 exactly on the collision-tested target
      placement (position AND yaw), regardless of where the object was
      actually recorded in its source scene.
  A1. Real motion is preserved after re-anchoring: an object moving at a
      known rate along its own heading in the database keeps moving at
      that same rate and along its (re-anchored) heading after being
      pasted -- checked with the training clip's ego held static, so local
      coordinates equal global ones and displacement is easy to verify
      directly, both in magnitude and direction.
  A2. Ego turns 90 degrees between training frames (rotation only, R != I,
      R != R.T -- catches a transposed-R bug in the final reprojection
      step, the same class of bug a straight-line-only test would miss;
      see tools/create_temporal_gt_database.py's docstring and the
      A2/A4 checks this replaces for the earlier world-static design).
  A3. Negative check: the A2 result does NOT match what a transposed-R
      bug would produce.
  A4. Graceful degradation: once the training clip's time offset runs
      past the object's last recorded frame, its position holds at that
      last frame instead of extrapolating.
  A5. Full __call__ path (draw -> cache -> replay across training frames,
      including a rotation) preserves instance ids and matches the
      low-level _reproject_to_frame result exactly.
  A6. Empty db_sampler.sample_all() result is handled without crashing.
  A7. Occlusion filter, via the real _apply_paste path. The filter is
      ONE-DIRECTIONAL (LIFT Sec 3.3 "filter out the occluded point" of the
      pasted pattern): a pasted point behind real geometry on the same ray
      is dropped, real points are NEVER removed whatever the geometry, and
      an object stripped below min_paste_points loses its box and label
      too rather than being supervised with no evidence.

Exits non-zero on any failure.
"""
import os
import sys

import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, 'mmdetection3d'))

import plugin.vip3d.pipeline as pipeline_mod  # noqa: E402
from mmdet3d.core.bbox import LiDARInstance3DBoxes  # noqa: E402
from mmdet3d.core.points import LiDARPoints  # noqa: E402

TCOS = pipeline_mod.TrackConsistentObjectSample


def rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)


def make_lidar_points(xyz):
    xyz = np.asarray(xyz, dtype=np.float32)
    extra = np.zeros((len(xyz), 2), dtype=np.float32)  # intensity, ring
    return LiDARPoints(torch.from_numpy(np.concatenate([xyz, extra], axis=1)),
                        points_dim=5, attribute_dims=dict(intensity=3, ring=4))


def make_sequence(ctr_g0, yaw_g0, speed_mps, dts):
    """A synthetic object moving at `speed_mps` along its OWN heading
    (yaw_g0) in DB-global coordinates, with one point at its own center and
    one point 1m ahead along its heading (to also exercise point rotation)."""
    heading = np.array([np.cos(yaw_g0), np.sin(yaw_g0), 0.0], dtype=np.float32)
    frames = []
    for dt in dts:
        ctr_g = np.asarray(ctr_g0, dtype=np.float32) + heading * speed_mps * dt
        pts = np.stack([ctr_g, ctr_g + heading], axis=0)
        frames.append(dict(dt=dt, ctr_g=ctr_g, yaw_g=yaw_g0, points=make_lidar_points(pts)))
    return frames


class FakeTemporalDBSampler:
    """Mimics TemporalDataBaseSampler.sample_all's return shape without
    touching disk or a real database."""

    def __init__(self, box3d_lidar, label, sequence):
        self.box3d_lidar = box3d_lidar
        self.label = label
        self.sequence = sequence

    def sample_all(self, gt_bboxes, gt_labels, img=None):
        return dict(
            gt_labels_3d=np.array([self.label]),
            gt_bboxes_3d=self.box3d_lidar[np.newaxis, :].copy(),
            sequences=[self.sequence])


class EmptyDBSampler:
    def sample_all(self, gt_bboxes, gt_labels, img=None):
        return None


def make_obj(db_sampler, min_paste_points=0):
    obj = TCOS.__new__(TCOS)          # bypasses __init__, so set what it would have
    obj.enabled = True
    obj.db_sampler = db_sampler
    obj.remove_points_in_boxes = lambda p, b: p
    # 0 by default: the geometry checks use 1-2 point objects and are not
    # about the evidence guard. A7 sets a real threshold to exercise it.
    obj.min_paste_points = min_paste_points
    return obj


def fail(msg):
    print(f'FAIL: {msg}')
    sys.exit(1)


def check(cond, msg):
    if not cond:
        fail(msg)
    print(f'  ok: {msg}')


def main():
    yaw_target = np.deg2rad(30)
    box3d_lidar = np.array([10.0, 5.0, -1.0, 4.0, 1.8, 1.5, yaw_target, 0.0, 0.0], dtype=np.float32)

    db_ctr_g0 = np.array([1000.0, 500.0, 0.0], dtype=np.float32)
    db_yaw_g0 = 0.0
    speed = 2.0  # m/s along the object's own heading, in DB-global coords
    dts = [0.0, 0.5, 1.0, 1.5]
    sequence = make_sequence(db_ctr_g0, db_yaw_g0, speed, dts)

    obj = make_obj(FakeTemporalDBSampler(box3d_lidar, label=0, sequence=sequence))
    identity_frame = dict(
        gt_bboxes_3d=LiDARInstance3DBoxes(torch.zeros((0, 9)), box_dim=9),
        gt_labels_3d=np.zeros((0,), dtype=np.int64),
        l2g_r_mat=rot_z(0.0), l2g_t=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        timestamp=100.0,
    )
    aug_state = {}
    paste0 = obj._draw_and_cache(identity_frame, aug_state)
    cache = aug_state['object_sample']

    print('A0. Re-anchoring lands frame 0 exactly on the target placement')
    check(np.allclose(paste0['gt_bboxes_3d'][0, :3], box3d_lidar[:3], atol=1e-4),
          'frame-0 center matches the collision-tested target exactly')
    check(abs(paste0['gt_bboxes_3d'][0, 6] - yaw_target) < 1e-4,
          'frame-0 yaw matches the collision-tested target exactly')

    print('A1. Real motion preserved (magnitude + direction) with ego held static')
    heading_reanchored = np.array([np.cos(yaw_target), np.sin(yaw_target), 0.0])
    for dt in [0.5, 1.0, 1.5]:
        frame_dict = dict(l2g_r_mat=rot_z(0.0), l2g_t=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
                           timestamp=100.0 + dt)
        paste_t = obj._reproject_to_frame(frame_dict, cache)
        disp = paste_t['gt_bboxes_3d'][0, :3] - paste0['gt_bboxes_3d'][0, :3]
        expected = heading_reanchored * speed * dt
        check(np.allclose(disp, expected, atol=1e-3),
              f'dt={dt}: displacement {disp.round(3)} matches expected {expected.round(3)}')

    print('A2. Ego turns 90deg between training frames (rotation only)')
    d2 = dict(l2g_r_mat=rot_z(np.pi / 2), l2g_t=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
              timestamp=100.0 + 1.0)
    paste2 = obj._reproject_to_frame(d2, cache)
    # ctr_g at dt=1.0 (independently recomputed, not reusing the implementation's formula):
    Rd = np.array([[np.cos(yaw_target), -np.sin(yaw_target), 0],
                   [np.sin(yaw_target), np.cos(yaw_target), 0], [0, 0, 1]])
    delta_t = box3d_lidar[:3] - db_ctr_g0 @ Rd.T
    ctr_g_at_1s = (db_ctr_g0 + np.array([np.cos(db_yaw_g0), np.sin(db_yaw_g0), 0]) * speed * 1.0) @ Rd.T + delta_t
    expect_ctr = ctr_g_at_1s @ rot_z(np.pi / 2)
    check(np.allclose(paste2['gt_bboxes_3d'][0, :3], expect_ctr, atol=1e-3),
          f'local center is {expect_ctr.round(3)} after a 90deg ego turn')

    print('A3. Negative check: result must NOT match a transposed-R bug')
    wrong_ctr = ctr_g_at_1s @ rot_z(np.pi / 2).T
    check(not np.allclose(paste2['gt_bboxes_3d'][0, :3], wrong_ctr, atol=1e-3),
          'A2 result differs from what a transposed-R bug would produce')

    print('A4. Graceful degradation past the last recorded frame')
    d_far = dict(l2g_r_mat=rot_z(0.0), l2g_t=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
                 timestamp=100.0 + 10.0)  # far past dts[-1]=1.5
    paste_far = obj._reproject_to_frame(d_far, cache)
    d_last = dict(l2g_r_mat=rot_z(0.0), l2g_t=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
                  timestamp=100.0 + 1.5)
    paste_last = obj._reproject_to_frame(d_last, cache)
    check(np.allclose(paste_far['gt_bboxes_3d'][0, :3], paste_last['gt_bboxes_3d'][0, :3], atol=1e-6),
          'position at training_dt=10.0 holds at the last real frame (dt=1.5), not extrapolated')

    print('A5. Full __call__ path across frames, including a rotation')
    obj2 = make_obj(FakeTemporalDBSampler(box3d_lidar, label=0, sequence=sequence))
    aug_state2 = {}
    real_pts = LiDARPoints(torch.zeros((0, 5)), points_dim=5, attribute_dims=dict(intensity=3, ring=4))

    def frame_dict(R, t, ts):
        return dict(
            gt_bboxes_3d=LiDARInstance3DBoxes(torch.zeros((0, 9)), box_dim=9),
            gt_labels_3d=np.zeros((0,), dtype=np.int64),
            points=real_pts.clone(),
            l2g_r_mat=R, l2g_t=t, timestamp=ts,
            aug_state=aug_state2,
            ann_info=dict(instance_inds=np.zeros((0,), dtype=np.int64)),
        )

    out_a = obj2(frame_dict(rot_z(0.0), np.array([[0.0, 0.0, 0.0]], dtype=np.float32), 100.0))
    out_b = obj2(frame_dict(rot_z(np.pi / 2), np.array([[0.0, 0.0, 0.0]], dtype=np.float32), 101.0))
    check(out_a['ann_info']['instance_inds'][0] == out_b['ann_info']['instance_inds'][0],
          'same pasted instance id replayed across frames via __call__')
    check(np.allclose(out_b['gt_bboxes_3d'].tensor[0, :3].numpy(), expect_ctr, atol=1e-3),
          '__call__ path matches the low-level _reproject_to_frame result under rotation')

    print('A7b. An object stripped of evidence loses its box too')
    # min_paste_points=1 with a fully-occluded single-point object: the box
    # must not survive without points, or the model is taught to predict a
    # detection with nothing under it.
    box_far = np.array([20.0, 0.0, 0.0, 4.0, 1.8, 1.5, 0.0, 0.0, 0.0], dtype=np.float32)
    seq_far = [dict(dt=0.0, ctr_g=np.array([1000.0, 0.0, 0.0], dtype=np.float32), yaw_g=0.0,
                     points=make_lidar_points([[1000.0, 0.0, 0.0]]))]
    og = make_obj(FakeTemporalDBSampler(box_far, label=0, sequence=seq_far), min_paste_points=1)
    dg = dict(
        gt_bboxes_3d=LiDARInstance3DBoxes(torch.zeros((0, 9)), box_dim=9),
        gt_labels_3d=np.zeros((0,), dtype=np.int64),
        points=make_lidar_points([[10.0, 0.0, 0.0]]),   # real occluder, nearer, same ray
        l2g_r_mat=rot_z(0.0), l2g_t=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        timestamp=100.0, aug_state={},
        ann_info=dict(instance_inds=np.zeros((0,), dtype=np.int64)),
    )
    outg = og(dg)
    check(outg['gt_bboxes_3d'].tensor.shape[0] == 0,
          'fully-occluded object contributes no GT box')
    check(len(outg['ann_info']['instance_inds']) == 0,
          'and no instance_ind either (arrays stay aligned)')
    check(outg['points'].tensor.shape[0] == 1,
          'the real occluder itself is still there')

    print('A6. Empty db_sampler.sample_all() result')
    obj3 = make_obj(EmptyDBSampler())
    aug_state3 = {}
    d_empty = frame_dict(rot_z(0.0), np.array([[0.0, 0.0, 0.0]], dtype=np.float32), 100.0)
    d_empty['aug_state'] = aug_state3
    out_empty = obj3(d_empty)
    check(aug_state3['object_sample'] is None, 'None sample cached for later frames')
    check(out_empty['gt_bboxes_3d'].tensor.shape[0] == 0, 'no boxes added when nothing was sampled')

    print('A7. Occlusion filter, via the real _apply_paste path')
    # Flat (z=0, y=0) single-point objects placed with identity ego
    # throughout, so a shared x-axis means exactly the same ray (azimuth=0,
    # elevation=0) regardless of range -- makes "same ray, different range"
    # trivial to construct and predict exactly, reusing the already-proven
    # re-anchoring/reprojection path (A0) for where the pasted point lands.
    def one_point_sequence(ctr_g0):
        pts = np.array([ctr_g0], dtype=np.float32)
        return [dict(dt=0.0, ctr_g=np.asarray(ctr_g0, dtype=np.float32), yaw_g=0.0,
                     points=make_lidar_points(pts))]

    def run_paste(target_x, real_points_xyz):
        box = np.array([target_x, 0.0, 0.0, 4.0, 1.8, 1.5, 0.0, 0.0, 0.0], dtype=np.float32)
        seq = one_point_sequence([1000.0, 0.0, 0.0])
        o = make_obj(FakeTemporalDBSampler(box, label=0, sequence=seq))
        real_pts = make_lidar_points(real_points_xyz)
        d = dict(
            gt_bboxes_3d=LiDARInstance3DBoxes(torch.zeros((0, 9)), box_dim=9),
            gt_labels_3d=np.zeros((0,), dtype=np.int64),
            points=real_pts,
            l2g_r_mat=rot_z(0.0), l2g_t=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            timestamp=100.0, aug_state={},
        )
        return o(d)['points'].tensor[:, :3].numpy()

    unrelated = [0.0, 10.0, 0.0]  # far off-axis: azimuth ~90deg away, must survive regardless

    out = run_paste(target_x=20.0, real_points_xyz=[[10.0, 0.0, 0.0], unrelated])
    check(any(np.allclose(p, [10.0, 0.0, 0.0], atol=1e-3) for p in out),
          'closer real point (10,0,0) survives against a farther pasted point on the same ray')
    check(not any(np.allclose(p, [20.0, 0.0, 0.0], atol=1e-3) for p in out),
          'farther pasted point (20,0,0) is shadowed by the closer real occluder')
    check(any(np.allclose(p, unrelated, atol=1e-3) for p in out),
          'real point on an unrelated ray is left untouched')

    out = run_paste(target_x=10.0, real_points_xyz=[[20.0, 0.0, 0.0], unrelated])
    check(any(np.allclose(p, [10.0, 0.0, 0.0], atol=1e-3) for p in out),
          'closer pasted point (10,0,0) survives against a farther real point on the same ray')
    check(any(np.allclose(p, [20.0, 0.0, 0.0], atol=1e-3) for p in out),
          'farther REAL point (20,0,0) is KEPT -- the filter never removes real data')
    check(any(np.allclose(p, unrelated, atol=1e-3) for p in out),
          'real point on an unrelated ray is left untouched')

    print('\nALL CHECKS PASSED')


if __name__ == '__main__':
    main()
