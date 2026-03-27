import torch

cam = torch.load('ckpts/detr3d_resnet50.pth', map_location='cpu')
cam_state = cam['state_dict'] if 'state_dict' in cam else cam

pts = torch.load('path/to/your_centerpoint.pth', map_location='cpu')
pts_state = pts['state_dict'] if 'state_dict' in pts else pts

merged = dict(cam_state)

lidar_prefixes = ('pts_voxel_encoder', 'pts_middle_encoder', 'pts_backbone', 'pts_neck')
for k, v in pts_state.items():
    clean_k = k[len('model.'):] if k.startswith('model.') else k
    if clean_k.startswith(lidar_prefixes):
        merged[clean_k] = v

torch.save({'state_dict': merged}, 'ckpts/livip3d_init.pth')
print("done")
