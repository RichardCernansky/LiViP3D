"""
BEV feature and heatmap TensorBoard visualizer.
Call visualize_bev() from _forward_single; it logs every VIS_INTERVAL steps (rank-0 only).
Launch viewer:  tensorboard --logdir runs/bev_vis
"""

import torch.distributed as dist

# ENABLED       = True
# DEBUG_PRINTS  = False
# VIS_INTERVAL  = 20
_CLASS_NAMES = ['car', 'truck', 'bus', 'trailer', 'motorcycle', 'bicycle', 'pedestrian']

_writer        = None
_step          = 0
_gt_bev        = None   # (N,2) numpy array of GT box centres in normalised [0,1] BEV coords
_heatmap       = None   # numpy [K, H, W] sigmoid heatmap (max-class used for overlay)
_top_q_ref     = None   # numpy (K,2) xy [0,1] of top-K heatmap queries
_top_q_idx     = None   # 1-D int64 tensor: indices into query tensor for top-K queries
_smca_attn     = None   # list[num_cams] of numpy (K, H0*W0) — per-camera attention for top-K
_smca_gauss    = None   # list[num_cams] of numpy (K, H0*W0) — raw Gaussian mask for top-K
_smca_feat_hw  = None   # (H0, W0) feature map size used by SMCA


def set_gt_bev(xy_norm):
    """Store GT box centres so visualize_lidar_bev_attn() can overlay them.
    xy_norm: numpy/tensor (N,2), x and y already normalised to [0,1] in BEV space.
    """
    global _gt_bev
    import numpy as np
    _gt_bev = np.asarray(xy_norm) if xy_norm is not None else None


def set_heatmap(hm):
    """Store heatmap so visualize_lidar_bev_attn() can show it as a second panel.
    hm: Tensor [B, K, H, W] pre-sigmoid logits or None.
    """
    global _heatmap
    if hm is None:
        _heatmap = None
    else:
        _heatmap = hm[0].detach().float().sigmoid().cpu().numpy()  # [K, H, W]


def set_top_queries(ref_pts, indices):
    """Store top-K heatmap query positions and their indices in the query tensor.
    ref_pts : Tensor [K, 3]  normalised [0,1] BEV coords
    indices : Tensor [K]     integer indices into the full query tensor
    """
    global _top_q_ref, _top_q_idx
    import numpy as np
    _top_q_ref = ref_pts[:, :2].detach().cpu().numpy()   # [K, 2]
    _top_q_idx = indices.detach().cpu()                   # [K] int tensor


def store_smca_attn(attn_per_cam, H0, W0):
    """Called from SMCACrossAtten after the camera loop.
    attn_per_cam : list[num_cams] of Tensor [K, H0*W0]  (top-K queries only)
    H0, W0       : feature-map spatial dimensions
    """
    global _smca_attn, _smca_feat_hw
    import numpy as np
    _smca_attn    = [a.detach().cpu().numpy() for a in attn_per_cam]
    _smca_feat_hw = (H0, W0)


def store_smca_gauss(gauss_per_cam, H0, W0):
    """Called from SMCACrossAtten before the camera loop.
    gauss_per_cam : list[num_cams] of Tensor [K, H0*W0]  raw Gaussian mask for top-K queries
    H0, W0        : feature-map spatial dimensions
    """
    global _smca_gauss, _smca_feat_hw
    _smca_gauss   = [g.detach().cpu().numpy() for g in gauss_per_cam]
    _smca_feat_hw = (H0, W0)


def visualize_train_smca(heatmap, img):
    """New TensorBoard panel bev/train_smca.
    Left panel : BEV heatmap (max class) + top-K query dots.
    Right panels: per-camera image with SMCA attention heatmap overlaid.

    heatmap : Tensor [B, K_cls, H, W]  pre-sigmoid logits
    img     : Tensor [B, num_cam, 3, H_img, W_img]  normalised BGR (mean-only)
    """
    if not ENABLED:
        return
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    if _step % VIS_INTERVAL != 0:
        return
    if _top_q_ref is None or _smca_attn is None:
        return

    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import cv2

    writer   = _get_writer()
    step     = _step
    num_cams = len(_smca_attn)
    H0, W0   = _smca_feat_hw

    num_cams_vis = min(num_cams, img[0].shape[0])  # pre-compute for figure layout
    fig, axes = plt.subplots(1, num_cams_vis + 1, figsize=(6 * (num_cams_vis + 1), 6))

    # ── Panel 0: BEV heatmap + top-K query dots ─────────────────────────────
    hm = heatmap[0].detach().float().sigmoid().cpu().numpy()  # [K_cls, H, W]
    bg = hm.max(axis=0)
    bg = bg / (bg.max() + 1e-6)
    axes[0].imshow(bg, cmap='jet', origin='lower',
                   extent=[0, 1, 0, 1], aspect='auto', vmin=0, vmax=1)
    axes[0].scatter(_top_q_ref[:, 0], _top_q_ref[:, 1],
                    c='cyan', s=30, zorder=3, edgecolors='black', linewidths=0.3,
                    label=f'top-{len(_top_q_ref)} queries')
    axes[0].set_title(f'Heatmap + top-{len(_top_q_ref)} queries  step={step}')
    axes[0].set_xlabel('x (norm BEV)'); axes[0].set_ylabel('y (norm BEV)')
    axes[0].legend(fontsize=7)

    # ── BGR image denorm: mean=[103.53, 116.28, 123.675], std=1, to_rgb=False
    img_np = img[0].detach().cpu().float().numpy()  # [num_cam_img, 3, H_img, W_img]
    mean   = np.array([103.530, 116.280, 123.675], dtype=np.float32).reshape(3, 1, 1)
    # img backbone may see fewer cameras than SMCA (e.g. front-3 backbone, 6-cam attention)
    num_cams_vis = min(num_cams, img_np.shape[0])

    for cam_i in range(num_cams_vis):
        ax = axes[cam_i + 1]

        # Denorm + convert BGR→RGB for display
        cam_bgr = (img_np[cam_i] + mean).clip(0, 255).astype(np.uint8)
        cam_rgb = cam_bgr[::-1].transpose(1, 2, 0)          # [H_img, W_img, 3]
        H_img, W_img = cam_rgb.shape[:2]

        # Attention: max over top-K queries → [H0*W0] → [H0, W0] → upsample
        attn = _smca_attn[cam_i]                            # [K, H0*W0]
        attn_map = attn.max(axis=0).reshape(H0, W0)        # [H0, W0]
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() + 1e-6)
        attn_up  = cv2.resize(attn_map, (W_img, H_img),
                              interpolation=cv2.INTER_LINEAR)

        ax.imshow(cam_rgb)
        ax.imshow(attn_up, cmap='hot', alpha=0.5, vmin=0, vmax=1)
        ax.set_title(f'Cam {cam_i}  step={step}')
        ax.axis('off')

    fig.suptitle(f'SMCA attention for top-{len(_top_q_ref)} heatmap queries')
    fig.tight_layout()
    writer.add_figure('bev/train_smca', fig, global_step=step)
    plt.close(fig)
    writer.flush()


def visualize_smca_gauss(img):
    """TensorBoard panel bev/smca_gauss — shows raw Gaussian mask overlaid on camera images.
    With sigma=4M the mask is essentially flat (all white).
    With sigma=5px it is a tight circular blob at the projected 3D box centre.

    img : Tensor [B, num_cam, 3, H_img, W_img]
    """
    if not ENABLED:
        return
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    if _step % VIS_INTERVAL != 0:
        return
    if _smca_gauss is None or _top_q_ref is None:
        return

    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import cv2

    writer   = _get_writer()
    step     = _step
    H0, W0   = _smca_feat_hw
    num_cams = len(_smca_gauss)

    img_np = img[0].detach().cpu().float().numpy()        # [num_cam_img, 3, H_img, W_img]
    mean   = np.array([103.530, 116.280, 123.675], dtype=np.float32).reshape(3, 1, 1)
    num_cams_vis = min(num_cams, img_np.shape[0])

    fig, axes = plt.subplots(1, num_cams_vis, figsize=(6 * num_cams_vis, 6))
    if num_cams_vis == 1:
        axes = [axes]

    for cam_i in range(num_cams_vis):
        ax = axes[cam_i]
        cam_bgr = (img_np[cam_i] + mean).clip(0, 255).astype(np.uint8)
        cam_rgb = cam_bgr[::-1].transpose(1, 2, 0)
        H_img, W_img = cam_rgb.shape[:2]

        # max over top-K queries → [H0, W0] → upsample
        gmap = _smca_gauss[cam_i]              # [K, H0*W0]
        gmap_2d = gmap.max(axis=0).reshape(H0, W0)
        # normalise so flat=all-white and peaked=circle visible
        gmap_2d = (gmap_2d - gmap_2d.min()) / (gmap_2d.max() - gmap_2d.min() + 1e-9)
        gmap_up = cv2.resize(gmap_2d, (W_img, H_img), interpolation=cv2.INTER_LINEAR)

        ax.imshow(cam_rgb)
        ax.imshow(gmap_up, cmap='hot', alpha=0.6, vmin=0, vmax=1)
        sigma_mean = _smca_gauss[cam_i].mean()
        ax.set_title(f'Gaussian mask  Cam {cam_i}  σ_raw≈{sigma_mean:.1e}  step={step}')
        ax.axis('off')

    fig.suptitle(f'Raw SMCA Gaussian mask (sigma=4M→flat  sigma=5px→circle)')
    fig.tight_layout()
    writer.add_figure('bev/smca_gauss', fig, global_step=step)
    plt.close(fig)
    writer.flush()


def _get_writer():
    global _writer
    if _writer is None:
        # torch.utils.tensorboard breaks on setuptools >= 60 (distutils.version removed)
        import distutils, setuptools._distutils.version as _dv  # type: ignore[import]
        distutils.version = _dv
        from torch.utils.tensorboard import SummaryWriter
        # Pillow >= 10 removed ANTIALIAS; tensorboard still calls it internally
        import PIL.Image
        if not hasattr(PIL.Image, 'ANTIALIAS'):
            PIL.Image.ANTIALIAS = PIL.Image.LANCZOS
        _writer = SummaryWriter('runs/bev_vis')
    return _writer


def visualize_bev(bev_feat, heatmap, gt_hm=None):
    """
    Args:
        bev_feat  : Tensor [B, C, H, W]  pts_neck output (raw features)
        heatmap   : Tensor [B, K, H, W]  heatmap_head output (pre-sigmoid logits)
        gt_hm     : Tensor [K, H, W]     GT Gaussian heatmap, or None during val/test
    """
    global _step
    _step += 1

    if not ENABLED:
        return
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    if _step % VIS_INTERVAL != 0:
        return

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    writer = _get_writer()
    step   = _step

    # ── BEV features: mean across channels ──────────────────────────────────
    feat = bev_feat[0].detach().float().cpu()          # [C, H, W]
    feat_mean = feat.mean(0).numpy()                   # [H, W]
    feat_mean = (feat_mean - feat_mean.min()) / (feat_mean.ptp() + 1e-6)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    im = ax.imshow(feat_mean, cmap='jet', origin='lower', vmin=0, vmax=1)
    ax.set_title(f'BEV feat mean  step={step}')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    writer.add_figure('bev/features', fig, global_step=step)
    plt.close(fig)

    # ── Predicted heatmap: max + per-class ──────────────────────────────────
    hm_pred = heatmap[0].detach().float().sigmoid().cpu().numpy()  # [K, H, W]
    # pred: auto-scale so structure is visible even when all values are small
    _log_heatmap_grid(writer, hm_pred, f'Pred heatmap  step={step}', 'bev/heatmap_pred', step, fixed_scale=False)

    # ── GT heatmap (training only) ───────────────────────────────────────────
    if gt_hm is not None:
        gt = gt_hm.detach().float().cpu().numpy()      # [K, H, W]
        _log_heatmap_grid(writer, gt, f'GT heatmap  step={step}', 'bev/heatmap_gt', step, fixed_scale=True)

    writer.flush()


def visualize_lidar_bev_attn(bev_feat, reference_points, sample_xy=None, attn_w=None):
    """Visualize deformable BEV attention: query positions, sampling points, GT overlay.

    Args:
        bev_feat         : Tensor [B, C, H, W]    raw BEV feature map
        reference_points : Tensor [B, N, 3]        query ref pts in normalised [0,1] BEV
        sample_xy        : Tensor [B, N, P, 2]     deformable sampling locations [0,1]
        attn_w           : Tensor [B, N, P]        softmax attention weights per sampling pt
    """
    if not ENABLED:
        return
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    if _step % VIS_INTERVAL != 0:
        return

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    writer = _get_writer()
    step   = _step

    ref = reference_points[0, :, :2].detach().cpu().numpy()    # [N, 2]

    if _heatmap is not None:
        bg = _heatmap.max(axis=0)                               # [H, W] max over classes
        bg = bg / (bg.max() + 1e-6)                            # normalise to [0,1] like training vis
        cmap = 'jet'
        title_bg = 'Pred heatmap'
    else:
        bg = bev_feat[0].detach().float().cpu().mean(0).numpy()
        bg = (bg - bg.min()) / (bg.ptp() + 1e-6)
        cmap = 'gray'
        title_bg = 'BEV feat'

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))

    def _show_hm(ax):
        ax.imshow(bg, cmap=cmap, origin='lower',
                  extent=[0, 1, 0, 1], aspect='auto', vmin=0, vmax=1)
        ax.set_xlabel('x (normalised BEV)')
        ax.set_ylabel('y (normalised BEV)')

    # Panel 1: GT heatmap only
    _show_hm(ax1)
    ax1.set_title(f'{title_bg}  step={step}')

    # Panel 2: GT heatmap + queries
    _show_hm(ax2)
    ax2.scatter(ref[:, 0], ref[:, 1], c='cyan', s=8, alpha=0.8,
                edgecolors='none', label=f'queries (N={len(ref)})', zorder=3)
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
    ax2.set_title(f'{title_bg} + queries  step={step}')
    ax2.legend(loc='upper right', fontsize=7, markerscale=1.5)

    # Panel 3: GT heatmap + queries + GT stars
    _show_hm(ax3)
    ax3.scatter(ref[:, 0], ref[:, 1], c='cyan', s=8, alpha=0.8,
                edgecolors='none', label=f'queries (N={len(ref)})', zorder=3)
    if _gt_bev is not None and len(_gt_bev) > 0:
        ax3.scatter(_gt_bev[:, 0], _gt_bev[:, 1],
                    marker='*', c='lime', s=150, zorder=5,
                    edgecolors='black', linewidths=0.4,
                    label=f'GT boxes (n={len(_gt_bev)})')
    ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
    ax3.set_title(f'{title_bg} + queries + GT  step={step}')
    ax3.legend(loc='upper right', fontsize=7, markerscale=1.5)
    fig.tight_layout()
    writer.add_figure('bev/lidar_attn', fig, global_step=step)
    plt.close(fig)


def _log_heatmap_grid(writer, hm, title, tag, step, fixed_scale=True):
    """hm: numpy [K, H, W]. fixed_scale=True uses vmin=0,vmax=1; False auto-scales per map."""
    import matplotlib.pyplot as plt

    K      = hm.shape[0]
    ncols  = 4
    nrows  = (K + 1 + ncols - 1) // ncols   # +1 for the max map

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    axes = axes.flatten()

    hm_max = hm.max(axis=0)
    vmax0 = 1.0 if fixed_scale else max(hm_max.max(), 1e-6)
    axes[0].imshow(hm_max, cmap='jet', origin='lower', vmin=0, vmax=vmax0)
    axes[0].set_title(f'max  (peak={hm_max.max():.3f})')
    axes[0].axis('off')

    names = _CLASS_NAMES[:K]
    for i, name in enumerate(names):
        vmax = 1.0 if fixed_scale else max(hm[i].max(), 1e-6)
        axes[i + 1].imshow(hm[i], cmap='jet', origin='lower', vmin=0, vmax=vmax)
        axes[i + 1].set_title(name)
        axes[i + 1].axis('off')

    for j in range(K + 1, len(axes)):
        axes[j].axis('off')

    fig.suptitle(title)
    fig.tight_layout()
    writer.add_figure(tag, fig, global_step=step)
    plt.close(fig)
