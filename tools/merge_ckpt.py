import torch

cam = torch.load('ckpt_init/detr3d_resnet50.pth', map_location='cpu')
cam_state = cam['state_dict'] if 'state_dict' in cam else cam

pts = torch.load('ckpt_init/pp_checkpoint_epoch_20.pth', map_location='cpu')
pts_state = pts['model_state'] if 'model_state' in pts else pts

print("=== LiDAR checkpoint keys (unique prefixes) ===")
prefixes = sorted(set(k.split('.')[0] for k in pts_state.keys()))
for p in prefixes:
    print(p)
print()

merged = dict(cam_state)

# backbone_2d.blocks.* → pts_backbone (SECOND conv blocks)
# backbone_2d.deblocks.* → pts_neck (SECONDFPN upsample layers; same weights, different module name in mmdet3d)
# vfe.* → pts_voxel_encoder (PillarFeatureNet)
copied = 0
for k, v in pts_state.items():
    if k.startswith('vfe.'):
        merged['pts_voxel_encoder' + k[len('vfe'):]] = v
        copied += 1
    elif k.startswith('backbone_2d.blocks.'):
        merged['pts_backbone' + k[len('backbone_2d'):]] = v
        copied += 1
    elif k.startswith('backbone_2d.deblocks.'):
        merged['pts_neck' + k[len('backbone_2d'):]] = v
        copied += 1

print(f"Copied {copied} tensors from LiDAR checkpoint")

# ── heatmap_head: shared_conv (384→64, 3×3 + BN) ────────────────────────────────────
for k, v in pts_state.items():
    if k.startswith('dense_head.shared_conv.'):
        merged['heatmap_head' + k[len('dense_head.shared_conv'):]] = v
        print(f'  {k} -> heatmap_head{k[len("dense_head.shared_conv"):]}  {list(v.shape)}')

# ── per-task heatmap heads: loaded as-is from CenterPoint, no merging ────────────────
# task layout: 0=car(1), 1=truck+cveh(2), 2=bus+trailer(2), 4=moto+bike(2), 5=ped+cone(2)
# hm.0.* maps to Sequential[0,1] (Conv+BN), hm.1 maps to Sequential[3] (final Conv)
task_map = {0: 'hm_task0', 1: 'hm_task1', 2: 'hm_task2', 4: 'hm_task4', 5: 'hm_task5'}
for task_id, dst in task_map.items():
    for k, v in pts_state.items():
        prefix = f'dense_head.heads_list.{task_id}.hm.'
        if not k.startswith(prefix):
            continue
        rest = k[len(prefix):]          # e.g. "0.0.weight", "0.1.bias", "1.weight"
        parts = rest.split('.')
        if parts[0] == '0':             # intermediate: hm.0.X.Y → Sequential[X].Y
            new_k = f'{dst}.{parts[1]}.{".".join(parts[2:])}'
        else:                           # final conv: hm.1.Y → Sequential[3].Y
            new_k = f'{dst}.3.{".".join(parts[1:])}'
        merged[new_k] = v
        print(f'  {k} -> {new_k}  {list(v.shape)}')
print(f'  Loaded per-task heads: {list(task_map.values())}')

print("=== Merged keys (unique prefixes) ===")
merged_prefixes = sorted(set(k.split('.')[0] for k in merged.keys()))
for p in merged_prefixes:
    print(p)
print()

torch.save({'state_dict': merged}, 'ckpt_init/livip3d_init.pth')
print("done")
