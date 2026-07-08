#!/usr/bin/env bash
set -e

CONFIG=plugin/vip3d/configs/vip3d_resnet50_6cam.py
WORK_DIR=work_dirs/vip3d_6cam

for EP in $(seq 10 24); do
    CKPT=${WORK_DIR}/epoch_${EP}.pth

    if [ ! -f ${CKPT} ]; then
        echo "Skipping epoch ${EP} — checkpoint not found"
        continue
    fi

    OUT_DIR=${WORK_DIR}/ep${EP}

    echo "========================================"
    echo "Evaluating epoch ${EP}"
    echo "========================================"

    mkdir -p ${OUT_DIR}

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tools/test.py \
        ${CONFIG} \
        ${CKPT} \
        --eval bbox \
        --out ${OUT_DIR}/results_ep${EP}.pkl \
        --output_dir ${OUT_DIR}

    python tools/prediction_eval.py \
        --result_path ${OUT_DIR}/results_nusc.json

    echo "Done epoch ${EP} → ${OUT_DIR}"
done

echo "All done."
