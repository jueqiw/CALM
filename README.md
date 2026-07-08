<div align="center">

## CALM: Interpretable Cross-Modal Alignment for Biomarker Discovery from Unpaired Data


[![MICCAI](https://img.shields.io/badge/MICCAI-2026-lightblue)](https://arxiv.org/pdf/2607.01656)
</div>

## Overview
<p align="center">
  <img src="fig/calm_model_explainer.gif" alt="CALM model overview" width="70%">
</p>

## Getting Started

### Environment

```bash
pip install torch numpy pandas scikit-learn monai nibabel nilearn matplotlib seaborn scipy
```

### Quick smoke test (no data required)

The pipeline ships with a synthetic-data mode so you can verify it runs end-to-end without any
real data or cluster paths:

```bash
cd code
python3 main.py --use_synthetic_data --not_write_tensorboard \
    --experiment_name=smoke --n_epochs=2 --batch_size=16 \
    --output_features=6 --normalization=layer --use_mmd_alignment \
    --alignment_coral_weight=5 --alignment_contrastive_weight=5 \
    --alignment_orthogonality_weight=0.01 --alignment_tau=0.07 \
    --genetics_learning_rate=0.001 --encoder_finetune_lr=1e-5 --imaging_feature_noise=0.15
```

This generates random pseudo data of the correct shapes, trains for 2 epochs, and prints the
loss / ACC / AUC per epoch.

### Running on real data — paths to change

All dataset paths are cluster-specific placeholders. **Point them to your own data before a real
run** (the smoke test above bypasses all of them):

| What | Where to edit |
|---|---|
| All dataset/checkpoint roots (ABIDE/ACE/SSC CSVs, FreeSurfer dirs, TensorBoard, results) | `code/utils/const.py` |
| SSC + ACE genetics CSVs | `load_genetics_data()` in `code/main.py` |
| Stage-1 imaging checkpoint output dir | `STAGE1_OUT` in `job_scripts/stage1_imaging.sh` |
| Pretrained Stage-1 checkpoints (`--pretrained_imaging_checkpoint`, `--pretrained_genetics_checkpoint`) | `IMG_CKPT_TMPL` / `GEN_CKPT_TMPL` in `job_scripts/stage2_alignment.sh` (currently `/path/to/...`) |

Expected on-disk format:
- **Imaging:** per-subject Brainnetome ROI features as a flat `n_ROI × 4` vector (ROI-major),
  loaded by `get_ABIDE_I_subject` / `get_ABIDE_II_subject` / `get_ACE_subjects` into each
  sample's `image_features`. (If your CSV is feature-major, flip the reshape in
  `_ROIFeatureDataset` in `main.py`.)
- **Genetics:** pathway tensors via `PathwayDataset` (177 pathways × 6 GWAS traits).

### Full 5-fold run (paper hyperparameters, two stages)

All job scripts live in `job_scripts/`. The paper's procedure (Sec. 2.4) is two-stage:
encoders + classifiers are pretrained (50 epochs), then the linear projections are trained (30
epochs). Both stages use `λcmmd=5, λcon=5, λorth=0.01, τ=0.07, d=6, batch=64`, projector lr
`1e-3`, encoder finetune lr `1e-5`, imaging feature noise `σ=0.15`.

The scripts are plain bash (run each on your own machine/cluster; add your scheduler's
directives if submitting to a queue):

```bash
# Stage 1 — pretrain the modality encoders (5-fold). Set each STAGE1_OUT first.
bash job_scripts/stage1_imaging.sh     # per-ROI imaging encoder E_I
bash job_scripts/stage1_genetics.sh    # per-pathway genetics encoder E_G

# Stage 2 — cross-modal alignment (5-fold). Set IMG_CKPT_TMPL / GEN_CKPT_TMPL to the
#           Stage-1 checkpoints (absolute paths), then:
bash job_scripts/stage2_alignment.sh
```

### Repo layout

- `code/main.py` — CALM alignment training (the entry point).
- `code/models/` — `genetics_encoder.py` (per-entity `PathwayEncoder`, reused as the
  per-ROI imaging encoder `E_I`), `alignment_model.py` (`SharedLatentProjector`), `losses.py`.
- `code/train_stage1.py` — Stage-1 encoder pretraining; pick the modality with
  `--modality {imaging,genetics}` (per-ROI imaging `E_I` or per-pathway genetics `E_G`).
- `code/utils/` — `const.py` (paths), `add_argument.py` (CLI flags), `utils.py` (loaders).
- `job_scripts/` — `stage1_imaging.sh`, `stage1_genetics.sh`, `stage2_alignment.sh` (5-fold launchers).

## Method
<p align="center">
  <img src="fig/model_architecture.png" alt="CALM model architecture" width="100%">
</p>

## Imaging-Genetics Associations
<p align="center">
  <img src="fig/trait_bar_ab_bottom.png" alt="Imaging-genetics pathway–ROI associations" width="100%">
</p>

## Results
<p align="center">
  <img src="fig/scatter_and_snr_combined.png" alt="Association stability: unpaired CALM vs paired baseline" width="100%">
</p>

## Citation
If any of the results in this paper or code are useful for your research, please cite the corresponding paper:

```
@inproceedings{wang2026calm,
  title={CALM: Interpretable Cross-Modal Alignment for Biomarker Discovery from Unpaired Data},
  author={Wang, Jueqi and Jacokes, Zachary and Van Horn, John Darrell and Pelphrey, Kevin A. and Schatz, Michael C. and Venkataraman, Archana},
  booktitle={Medical Image Computing and Computer-Assisted Intervention -- MICCAI 2026},
  year={2026},
  publisher={Springer}
}
```
