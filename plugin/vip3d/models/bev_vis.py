"""
BEV feature and heatmap TensorBoard visualizer.
Call visualize_bev() from _forward_single; it logs every VIS_INTERVAL steps (rank-0 only).
Launch viewer:  tensorboard --logdir runs/bev_vis
"""

import torch
import torch.distributed as dist

ENABLED      = False # set False to skip all visualization
VIS_INTERVAL = 20   # _forward_single calls between log events
_CLASS_NAMES = ['car', 'truck', 'bus', 'trailer', 'motorcycle', 'bicycle', 'pedestrian']

_writer = None
_step   = 0


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
