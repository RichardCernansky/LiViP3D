import json, numpy as np, sys
from collections import defaultdict
from nuscenes.nuscenes import NuScenes

results_path = sys.argv[1] if len(sys.argv) > 1 else 'work_dirs/livip3d_resnet50_3cam/results_nusc.json'
nusc = NuScenes(version='v1.0-trainval', dataroot='/mnt/ssd/cernanskyr/', verbose=False)
results = json.load(open(results_path))

CLASSES = {'car', 'truck', 'bus', 'trailer', 'motorcycle', 'bicycle', 'pedestrian'}
gt_lookup = defaultdict(list)
for sample in nusc.sample:
    for ann_tok in sample['anns']:
        ann = nusc.get('sample_annotation', ann_tok)
        for c in CLASSES:
            if c in ann['category_name']:
                gt_lookup[sample['token']].append((ann['translation'][0], ann['translation'][1], c))
                break

tp, fp = [], []
per_cls = defaultdict(lambda: {'tp': [], 'fp': []})

for tok, dets in results['results'].items():
    gts = list(gt_lookup[tok])
    used = set()
    for d in sorted(dets, key=lambda x: -x['tracking_score']):
        s = d['tracking_score']
        cx, cy = d['translation'][:2]
        cn = d['tracking_name']
        bi, bd = -1, 1e9
        for i, (gx, gy, gc) in enumerate(gts):
            if i in used or gc != cn:
                continue
            dist = ((cx - gx) ** 2 + (cy - gy) ** 2) ** 0.5
            if dist < bd:
                bd, bi = dist, i
        if bi >= 0 and bd < 2.0:
            tp.append(s); per_cls[cn]['tp'].append(s); used.add(bi)
        else:
            fp.append(s); per_cls[cn]['fp'].append(s)

tp = np.array(tp)
fp = np.array(fp)
print(f'\nOverall:')
print(f'  TP: n={len(tp):5d}  mean={tp.mean():.4f}  median={np.median(tp):.4f}  std={tp.std():.4f}')
print(f'  FP: n={len(fp):5d}  mean={fp.mean():.4f}  median={np.median(fp):.4f}  std={fp.std():.4f}')
print(f'  FP rate: {100 * len(fp) / (len(tp) + len(fp)):.1f}%')
print(f'\nPer class:')
for c in ['car', 'truck', 'bus', 'trailer', 'motorcycle', 'bicycle', 'pedestrian']:
    t = np.array(per_cls[c]['tp'])
    f = np.array(per_cls[c]['fp'])
    if len(t) + len(f) == 0:
        continue
    ts = f'{t.mean():.3f}' if len(t) else 'n/a'
    fs = f'{f.mean():.3f}' if len(f) else 'n/a'
    print(f'  {c:12s}  TP={len(t):4d} ({ts})  FP={len(f):4d} ({fs})')
