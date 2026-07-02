#!/bin/bash

# =============================================================================
# CALM: Class-conditional Alignment with Linear Maps
# Stage-2 cross-modal alignment, 5-fold CV. All hyperparameters are taken
# verbatim from the paper (CALM, Sec. 2.4 "Implementation Details" + Sec. 3).
#
#   Two-stage procedure:
#     Stage 1  pretrain modality encoders + classifiers ............ 50 epochs
#     Stage 2  freeze classifiers, train linear projections W_I,W_G . 30 epochs   <- THIS SCRIPT
#   Loss weights (Eq. 2):  lambda_cmmd = 5, lambda_con = 5, lambda_orth = 0.01
#   Contrastive temperature ....................................... tau = 0.07
#   Projector learning rate ....................................... 1e-3
#   Encoder fine-tuning learning rate ............................. 1e-5
#   Normalization ................................................. LayerNorm
#   Batch size .................................................... 64
#   Gaussian feature noise on imaging modality .................... sigma = 0.15
#   Shared latent dimension ....................................... d = 6
#   Cross-validation .............................................. 5-fold
#   Inputs z-score normalized using per-feature train-set statistics.
#
#   Data (Sec. 3):
#     Imaging  : ABIDE I+II, FreeSurfer Brainnetome 246 ROIs x 4 features (+ComBat)
#     Genetics : SSC, 177 KEGG pathways x 6 GWAS traits
#     Test     : ACE (paired MRI + genetics)
#
#   NOTE: Stage 1 (50-epoch encoder/classifier pretraining) is run separately;
#   set the resulting per-fold checkpoints in PRETRAINED_* below.
# =============================================================================

cd "$(dirname "$0")/../code"

# ---- Paper hyperparameters (Sec. 2.4) ----
LAMBDA_CMMD=5            # --alignment_coral_weight (CMMD/MMD weight, with --use_mmd_alignment)
LAMBDA_CON=5            # --alignment_contrastive_weight
LAMBDA_ORTH=0.01        # --alignment_orthogonality_weight
TAU=0.07               # --alignment_tau
D_LATENT=6             # --output_features (shared latent dim d)
BATCH_SIZE=64          # --batch_size
PROJECTOR_LR=0.001     # --genetics_learning_rate (trains projector + genetics encoder)
ENCODER_FT_LR=0.00001  # --encoder_finetune_lr (imaging encoder fine-tune, 1e-5)
PRETRAIN_EPOCHS=50     # --pretrain_epochs (Stage 1, done separately)
ALIGN_EPOCHS=30        # --align_epochs == Stage-2 epochs (= --n_epochs here)
FEATURE_NOISE_SIGMA=0.15  # --imaging_feature_noise

# ---- Stage-1 pretrained checkpoints (absolute paths; CHANGE to yours; see README) ----
# Imaging template must match stage1_imaging.sh's STAGE1_OUT + filename.
IMG_CKPT_TMPL="/path/to/checkpoints/imaging/stage1_imaging_fold%d.pth"
GEN_CKPT_TMPL="/path/to/checkpoints/genetics/stage1_genetics_fold%d.pth"

for FOLD in 0 1 2 3 4; do
    python3 main.py \
        --experiment_name="calm_camera_ready_fold${FOLD}" \
        --dataset="ACE" \
        --test_fold="${FOLD}" \
        --n_epochs="${ALIGN_EPOCHS}" \
        --pretrain_epochs="${PRETRAIN_EPOCHS}" \
        --align_epochs="${ALIGN_EPOCHS}" \
        --normalization="layer" \
        --output_features="${D_LATENT}" \
        --batch_size="${BATCH_SIZE}" \
        --learning_rate="${PROJECTOR_LR}" \
        --genetics_learning_rate="${PROJECTOR_LR}" \
        --encoder_finetune_lr="${ENCODER_FT_LR}" \
        --use_mmd_alignment \
        --alignment_coral_weight="${LAMBDA_CMMD}" \
        --alignment_contrastive_weight="${LAMBDA_CON}" \
        --alignment_orthogonality_weight="${LAMBDA_ORTH}" \
        --alignment_tau="${TAU}" \
        --imaging_feature_noise="${FEATURE_NOISE_SIGMA}" \
        --pretrained_imaging_checkpoint="$(printf "$IMG_CKPT_TMPL" "$FOLD")" \
        --pretrained_genetics_checkpoint="$(printf "$GEN_CKPT_TMPL" "$FOLD")"
done
