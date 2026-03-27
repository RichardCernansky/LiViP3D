import torch

cam = torch.load('ckpt_init/detr3d_resnet50.pth', map_location='cpu')
cam_state = cam['state_dict'] if 'state_dict' in cam else cam

pts = torch.load('ckpt_init/checkpoint_epoch_20.pth', map_location='cpu')
pts_state = pts['state_dict'] if 'state_dict' in pts else pts

print("=== LiDAR checkpoint keys (unique prefixes) ===")
prefixes = sorted(set(
    (k[len('model.'):] if k.startswith('model.') else k).split('.')[0]
    for k in pts_state.keys()
))
for p in prefixes:
    print(p)
print()

merged = dict(cam_state)

lidar_prefixes = ('pts_voxel_encoder', 'pts_middle_encoder', 'pts_backbone', 'pts_neck')
for k, v in pts_state.items():
    clean_k = k[len('model.'):] if k.startswith('model.') else k
    if clean_k.startswith(lidar_prefixes):
        merged[clean_k] = v

torch.save({'state_dict': merged}, 'ckpts/livip3d_init.pth')
print("done")
