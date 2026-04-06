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

key_map = {
    'vfe': 'pts_voxel_encoder',
    'backbone_2d': 'pts_backbone',
    'dense_head': 'pts_neck',
}

copied = 0
for k, v in pts_state.items():
    prefix = k.split('.')[0]
    if prefix in key_map:
        new_k = key_map[prefix] + k[len(prefix):]
        merged[new_k] = v
        copied += 1

print(f"Copied {copied} tensors from LiDAR checkpoint")
print("=== Merged keys (unique prefixes) ===")
merged_prefixes = sorted(set(k.split('.')[0] for k in merged.keys()))
for p in merged_prefixes:
    print(p)
print()

torch.save({'state_dict': merged}, 'ckpt_init/livip3d_init.pth')
print("done")
