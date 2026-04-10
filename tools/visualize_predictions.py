"""
BEV visualization of ViP3D detections and predicted trajectories.
Usage:
    python tools/visualize_predictions.py \
        --results work_dirs/vip3d_resnet50_3cam/results_nusc.json \
        --dataroot /mnt/hdd/cernanskyr/nuscenes \
        --out work_dirs/vip3d_resnet50_3cam/vis \
        --num_samples 20 \
        --score_thr 0.3
"""
import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pyquaternion import Quaternion

CLASS_COLORS = {
    'car':          '#4c8be0',
    'truck':        '#e07c4c',
    'bus':          '#e0c14c',
    'trailer':      '#9b59b6',
    'motorcycle':   '#e74c3c',
    'bicycle':      '#2ecc71',
    'pedestrian':   '#1abc9c',
    'construction_vehicle': '#95a5a6',
    'traffic_cone': '#f39c12',
    'barrier':      '#bdc3c7',
}
PRED_COLOR = '#ff4444'
GT_COLOR   = '#44ff44'
RANGE = 50  # metres around ego


def quat_to_yaw(q):
    """quaternion [w,x,y,z] → yaw (rad)"""
    q = Quaternion(q)
    return q.yaw_pitch_roll[0]


def draw_box_bev(ax, cx, cy, w, l, yaw, color, alpha=0.8, linewidth=1.5):
    """Draw a rotated rectangle (BEV box)."""
    corners = np.array([
        [ l/2,  w/2],
        [-l/2,  w/2],
        [-l/2, -w/2],
        [ l/2, -w/2],
    ])
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    corners = (R @ corners.T).T + np.array([cx, cy])
    poly = plt.Polygon(corners, fill=False, edgecolor=color,
                       linewidth=linewidth, alpha=alpha)
    ax.add_patch(poly)
    # heading tick
    front = (R @ np.array([l/2, 0])) + np.array([cx, cy])
    ax.plot([cx, front[0]], [cy, front[1]], color=color,
            linewidth=linewidth, alpha=alpha)


EGO_COLOR  = '#ffffff'
EGO_LENGTH = 4.5   # metres
EGO_WIDTH  = 2.0


def draw_ego(ax, yaw):
    """Draw ego vehicle as a filled arrow pointing in its heading direction."""
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    # Box corners (front of car = +x in local frame)
    corners = np.array([
        [ EGO_LENGTH/2,  EGO_WIDTH/2],
        [-EGO_LENGTH/2,  EGO_WIDTH/2],
        [-EGO_LENGTH/2, -EGO_WIDTH/2],
        [ EGO_LENGTH/2, -EGO_WIDTH/2],
    ])
    corners = (R @ corners.T).T
    poly = plt.Polygon(corners, closed=True, facecolor=EGO_COLOR,
                       edgecolor=EGO_COLOR, alpha=0.9, zorder=5)
    ax.add_patch(poly)
    # Heading arrow from centre to front
    front = R @ np.array([EGO_LENGTH/2 + 1.5, 0.0])
    ax.annotate('', xy=(front[0], front[1]), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='#ffdd00',
                                lw=2.0, mutation_scale=15),
                zorder=6)


def visualize_sample(ax, ego_xy, ego_yaw, pred_boxes, gt_boxes, score_thr):
    ax.set_facecolor('#1a1a2e')
    ax.set_xlim(-RANGE, RANGE)
    ax.set_ylim(-RANGE, RANGE)
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')

    # ego vehicle
    draw_ego(ax, ego_yaw)

    # GT boxes
    for box in gt_boxes:
        tx, ty = box['translation'][0] - ego_xy[0], box['translation'][1] - ego_xy[1]
        if abs(tx) > RANGE or abs(ty) > RANGE:
            continue
        sz = box['size']  # [w, l, h]
        yaw = quat_to_yaw(box['rotation'])
        draw_box_bev(ax, tx, ty, sz[0], sz[1], yaw,
                     color=GT_COLOR, alpha=0.6, linewidth=1.0)

    # Predicted boxes + trajectories
    for box in pred_boxes:
        if box.get('tracking_score', 1.0) < score_thr:
            continue
        tx = box['translation'][0] - ego_xy[0]
        ty = box['translation'][1] - ego_xy[1]
        if abs(tx) > RANGE or abs(ty) > RANGE:
            continue

        cls = box.get('tracking_name', 'car')
        color = CLASS_COLORS.get(cls, '#ffffff')
        sz = box['size']
        yaw = quat_to_yaw(box['rotation'])
        draw_box_bev(ax, tx, ty, sz[0], sz[1], yaw, color=color)

        # Predicted trajectories (pred_outputs): shape (K, T, 2) absolute coords
        if 'pred_outputs' in box and box['pred_outputs']:
            trajs = np.array(box['pred_outputs'])  # (K, T, 2)
            probs = np.array(box.get('pred_probs', [1.0/len(trajs)]*len(trajs)))
            best_k = np.argmax(probs)
            for k, traj in enumerate(trajs):
                traj_local = traj - np.array(ego_xy)
                alpha = 0.8 if k == best_k else 0.25
                lw = 1.5 if k == best_k else 0.7
                ax.plot(traj_local[:, 0], traj_local[:, 1],
                        color=PRED_COLOR, alpha=alpha, linewidth=lw)
                # dot at endpoint
                if k == best_k:
                    ax.plot(traj_local[-1, 0], traj_local[-1, 1],
                            'o', color=PRED_COLOR, markersize=3, alpha=0.9)

    # Legend
    handles = [
        mpatches.Patch(color=EGO_COLOR,  label='Ego vehicle'),
        mpatches.Patch(color=GT_COLOR,   label='GT boxes'),
        mpatches.Patch(color=PRED_COLOR, label='Pred trajectories'),
    ]
    for cls, col in CLASS_COLORS.items():
        handles.append(mpatches.Patch(color=col, label=cls))
    ax.legend(handles=handles, loc='upper right', fontsize=6,
              framealpha=0.4, facecolor='#1a1a2e', labelcolor='white')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', required=True)
    parser.add_argument('--dataroot', default='/mnt/hdd/cernanskyr/nuscenes')
    parser.add_argument('--out', default='work_dirs/vis')
    parser.add_argument('--num_samples', type=int, default=20)
    parser.add_argument('--score_thr', type=float, default=0.3)
    parser.add_argument('--version', default='v1.0-trainval')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print('Loading results...')
    with open(args.results) as f:
        data = json.load(f)
    results = data['results']

    print('Loading NuScenes...')
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    # Build GT lookup: sample_token → list of annotation dicts
    print('Building GT lookup...')
    gt_lookup = {}
    for ann in nusc.sample_annotation:
        tok = ann['sample_token']
        if tok not in gt_lookup:
            gt_lookup[tok] = []
        sample = nusc.get('sample', tok)
        sd = nusc.get('sample_data', sample['data']['CAM_FRONT'])
        ego_pose = nusc.get('ego_pose', sd['ego_pose_token'])
        gt_lookup[tok].append({
            'translation': ann['translation'],
            'size': ann['size'],
            'rotation': ann['rotation'],
            'category': ann['category_name'],
        })

    tokens = list(results.keys())[:args.num_samples]

    for i, token in enumerate(tokens):
        pred_boxes = results[token]
        gt_boxes = gt_lookup.get(token, [])

        # Get ego pose for this sample
        sample = nusc.get('sample', token)
        sd = nusc.get('sample_data', sample['data']['CAM_FRONT'])
        ego_pose = nusc.get('ego_pose', sd['ego_pose_token'])
        ego_xy  = ego_pose['translation'][:2]
        ego_yaw = quat_to_yaw(ego_pose['rotation'])

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        fig.patch.set_facecolor('#1a1a2e')
        visualize_sample(ax, ego_xy, ego_yaw, pred_boxes, gt_boxes, args.score_thr)
        ax.set_title(f'Sample {i+1}: {token[:16]}...', color='white', fontsize=9)

        out_path = os.path.join(args.out, f'sample_{i+1:04d}.png')
        plt.tight_layout()
        plt.savefig(out_path, dpi=120, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.close()
        print(f'[{i+1}/{len(tokens)}] Saved {out_path}')

    print(f'\nDone. {len(tokens)} images saved to {args.out}/')


if __name__ == '__main__':
    main()
