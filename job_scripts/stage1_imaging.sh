#!/bin/bash

# =============================================================================
# CALM Stage-1: pretrain the per-ROI imaging encoder E_I (5-fold).
# Paper (Sec. 2.4): modality encoders + classifiers are pretrained for 50 epochs.
# Produces per-fold checkpoints consumed by stage2_alignment.sh
# (main.py --pretrained_imaging_checkpoint).
# =============================================================================

cd "$(dirname "$0")/../code"

# CHANGE: absolute dir where Stage-1 imaging checkpoints are written
# (must match IMG_CKPT_TMPL in stage2_alignment.sh).
STAGE1_OUT="/path/to/checkpoints/imaging"

for FOLD in 0 1 2 3 4; do
    python3 train_stage1.py \
        --modality=imaging \
        --test_fold="${FOLD}" \
        --pretrain_epochs=50 \
        --output_features=6 \
        --normalization=layer \
        --batch_size=64 \
        --learning_rate=0.001 \
        --imaging_feature_noise=0.15 \
        --stage1_output_dir="${STAGE1_OUT}"
done
