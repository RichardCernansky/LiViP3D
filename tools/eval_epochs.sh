#!/usr/bin/env bash
set -e

CONFIG=plugin/vip3d/configs/livip3d_resnet50_lidar_only.py
WORK_DIR=work_dirs/livip3d_lidar_only

for EP in 14 15 16 17; do
    CKPT=${WORK_DIR}/epoch_${EP}.pth
    OUT_DIR=${WORK_DIR}/ep${EP}

    echo "========================================"
    echo "Evaluating epoch ${EP}"
    echo "========================================"

    mkdir -p ${OUT_DIR}

    PYTHONPATH=. python tools/test.py \
        ${CONFIG} \
        ${CKPT} \
        --eval bbox \
        --out ${OUT_DIR}/results.pkl \
        --output_dir ${OUT_DIR}

    echo "Done epoch ${EP} → ${OUT_DIR}"
done

echo "All done."
