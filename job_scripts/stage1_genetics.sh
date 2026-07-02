#!/bin/bash

# =============================================================================
# CALM Stage-1: pretrain the per-pathway genetics encoder E_G (5-fold).
# Paper (Sec. 2.4): modality encoders + classifiers are pretrained for 50 epochs.
# Genetics input: 177 KEGG pathways x 6 GWAS traits (--genetics_input_channels=6).
# Produces per-fold checkpoints consumed by stage2_alignment.sh
# (main.py --pretrained_genetics_checkpoint).
# =============================================================================

cd "$(dirname "$0")/../code"

# CHANGE: absolute dir where Stage-1 genetics checkpoints are written
# (must match GEN_CKPT_TMPL in stage2_alignment.sh).
STAGE1_OUT="/path/to/checkpoints/genetics"

for FOLD in 0 1 2 3 4; do
    python3 train_stage1.py \
        --modality=genetics \
        --test_fold="${FOLD}" \
        --pretrain_epochs=50 \
        --output_features=6 \
        --normalization=layer \
        --batch_size=64 \
        --genetics_input_channels=6 \
        --genetics_learning_rate=0.001 \
        --stage1_output_dir="${STAGE1_OUT}"
done
