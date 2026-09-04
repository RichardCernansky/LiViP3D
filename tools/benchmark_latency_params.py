"""Benchmark params (M) and per-frame latency (ms) for the best-epoch
checkpoints of each LiViP3D/ViP3D training run.

Usage:
    PYTHONPATH=. python tools/benchmark_latency_params.py
    PYTHONPATH=. python tools/benchmark_latency_params.py --only "S1 (LiDAR only)" "S2 (+img-guided)"
    PYTHONPATH=. python tools/benchmark_latency_params.py --num-warmup 10 --num-timed 50
"""
import argparse
import gc
import importlib
import json
import os
import time

import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint

from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model

MODELS = [
    dict(
        name='ViP3D (baseline)',
        config='work_dirs/non-augmented/vip3d_6cam/vip3d_resnet50_6cam.py',
        checkpoint='work_dirs/non-augmented/vip3d_6cam/epoch_16.pth',
    ),
    dict(
        name='S1 (LiDAR only)',
        config='work_dirs/non-augmented/s1-livip3d_lidar_only/livip3d_resnet50_lidar_only.py',
        checkpoint='work_dirs/non-augmented/s1-livip3d_lidar_only/epoch_12_epa14.pth',
    ),
    dict(
        name='S2 (+img-guided)',
        config='work_dirs/non-augmented/s2-livip3d_lidar_img_guided/livip3d_resnet50_lidar_img_guided.py',
        checkpoint='work_dirs/non-augmented/s2-livip3d_lidar_img_guided/epoch_10.pth',
    ),
    dict(
        name='S3 (+SMCA, 3ep)',
        config='work_dirs/non-augmented/s3-livip3d_lidar_img_guided_smca_3ep/livip3d_resnet50_lidar_img_guided_smca.py',
        checkpoint='work_dirs/non-augmented/s3-livip3d_lidar_img_guided_smca_3ep/epoch_3.pth',
    ),
    dict(
        name='S3 (+SMCA, 6ep)',
        config='work_dirs/non-augmented/s3-livip3d_lidar_img_guided_smca_6ep/livip3d_resnet50_lidar_img_guided_smca.py',
        checkpoint='work_dirs/non-augmented/s3-livip3d_lidar_img_guided_smca_6ep/epoch_5.pth',
    ),
    dict(
        name='S1 augmented',
        config='work_dirs/augmented/s1-lidar_only/livip3d_resnet50_lidar_only.py',
        checkpoint='work_dirs/augmented/s1-lidar_only/epoch_20.pth',
    ),
]


def import_plugin(cfg, config_path):
    if not cfg.get('plugin', False):
        return
    if hasattr(cfg, 'plugin_dir'):
        plugin_dir = cfg.plugin_dir
        _module_dir = os.path.dirname(plugin_dir)
    else:
        _module_dir = os.path.dirname(config_path)
    _module_dir = _module_dir.split('/')
    _module_path = _module_dir[0]
    for m in _module_dir[1:]:
        _module_path = _module_path + '.' + m
    importlib.import_module(_module_path)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def benchmark_one(entry, num_warmup, num_timed):
    cfg = Config.fromfile(entry['config'])
    import_plugin(cfg, entry['config'])

    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.data.test.test_mode = True
    # Disable in-forward debug prints / TensorBoard BEV visualization (bev_vis.py):
    # these run extra tensor ops + matplotlib/TensorBoard rendering every
    # vis_interval steps and are not part of real inference cost.
    cfg.model.debug = False
    cfg.model.bev_vis = False

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, entry['checkpoint'], map_location='cpu')
    n_params = count_params(model)

    model = MMDataParallel(model.cuda(), device_ids=[0])
    model.eval()

    total = num_warmup + num_timed
    pure_inf_time = 0.0
    timed_iters = 0
    for i, data in enumerate(data_loader):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model(return_loss=False, rescale=True, **data)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        if i >= num_warmup:
            pure_inf_time += elapsed
            timed_iters += 1
            if timed_iters % 50 == 0:
                print(f'  [{timed_iters}/{num_timed}] '
                      f'running avg {pure_inf_time / timed_iters * 1000:.1f} ms/frame')

        if i + 1 >= total:
            break

    if timed_iters == 0:
        raise RuntimeError(
            f'val set for {entry["name"]} yielded fewer than '
            f'{num_warmup + 1} frames; lower --num-warmup/--num-timed')

    latency_ms = pure_inf_time / timed_iters * 1000
    fps = 1000.0 / latency_ms

    del model, data_loader, dataset
    gc.collect()
    torch.cuda.empty_cache()

    return dict(
        params_m=n_params / 1e6,
        latency_ms=latency_ms,
        fps=fps,
        timed_iters=timed_iters,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--num-warmup', type=int, default=30)
    ap.add_argument('--num-timed', type=int, default=200)
    ap.add_argument(
        '--only', nargs='*', default=None,
        help='subset of model names (as in MODELS[i]["name"]) to run')
    ap.add_argument(
        '--out', default='work_dirs/latency_params_results.json',
        help='where to dump raw results as JSON')
    args = ap.parse_args()

    results = []
    for entry in MODELS:
        if args.only and entry['name'] not in args.only:
            continue
        print(f'\n=== Benchmarking {entry["name"]} ===')
        print(f'  config:     {entry["config"]}')
        print(f'  checkpoint: {entry["checkpoint"]}')
        r = benchmark_one(entry, args.num_warmup, args.num_timed)
        r['name'] = entry['name']
        r['config'] = entry['config']
        r['checkpoint'] = entry['checkpoint']
        results.append(r)
        print(f'  -> params={r["params_m"]:.2f}M  '
              f'latency={r["latency_ms"]:.1f}ms  fps={r["fps"]:.2f}  '
              f'(over {r["timed_iters"]} timed frames)')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nRaw results written to {args.out}')

    print('\n| Experiment | Params (M) | Latency (ms) | FPS |')
    print('|---|---|---|---|')
    for r in results:
        print(f'| {r["name"]} | {r["params_m"]:.2f} | '
              f'{r["latency_ms"]:.1f} | {r["fps"]:.2f} |')


if __name__ == '__main__':
    main()
