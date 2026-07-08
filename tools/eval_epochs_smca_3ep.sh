#!/usr/bin/env bash
set -e

CONFIG=plugin/vip3d/configs/livip3d_resnet50_lidar_img_guided_smca.py
WORK_DIR=work_dirs/livip3d_lidar_img_guided_smca_3ep

for CKPT_NAME in epoch_3; do
    CKPT=${WORK_DIR}/${CKPT_NAME}.pth
    OUT_DIR=${WORK_DIR}/${CKPT_NAME}_eval

    echo "========================================"
    echo "Evaluating ${CKPT_NAME}"
    echo "========================================"

    mkdir -p ${OUT_DIR}

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tools/test.py \
        ${CONFIG} \
        ${CKPT} \
        --eval bbox \
        --out ${OUT_DIR}/results_${CKPT_NAME}.pkl \
        --output_dir ${OUT_DIR}

    python tools/prediction_eval.py \
        --result_path ${OUT_DIR}/results_nusc.json

    echo "Done ${CKPT_NAME} → ${OUT_DIR}"
done

echo "All done."
