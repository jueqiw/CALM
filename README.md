# CALM: Class-conditional Alignment with Linear Maps

<p align="center">
  <img src="fig/calm_model_explainer.gif" alt="CALM model overview" width="100%">
</p>

CALM learns **interpretable associations between brain ROIs and genetic pathways from
completely unpaired data** — i.e. imaging and genetics collected from disjoint populations.
Pretrained per-entity encoders project each modality into a shared latent space via learned
linear maps `W_I`, `W_G`; the association matrix `A = W_Iᵀ W_G ∈ R^{n_ROI × n_pathway}` is the
interpretable readout. Training matches class-conditional latent distributions (CMMD) and
separates diagnostic groups (supervised contrastive), with an orthogonality regularizer.

- **Imaging input:** FreeSurfer Brainnetome features — `n_ROI = 246` ROIs × `4` morphological
  features (volume, surface area, mean cortical thickness, std cortical thickness).
- **Genetics input:** `n_pathway = 177` KEGG pathways × `6` GWAS traits.
- **Train / test:** ABIDE I+II (imaging) + SSC (genetics) → tested on the paired ACE cohort.

## Environment

```bash
module load miniconda
conda activate /projectnb/ace-genetics/jueqiw/software/venvs/conda_envs/torch2_conda
# or: pip install torch numpy pandas scikit-learn monai nibabel nilearn matplotlib seaborn scipy
# (cvxopt is optional — only needed for the MK-MMD baseline)
```

## Quick smoke test (no data required)

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

## Dummy dataset — exercise the *real* loaders

`--use_synthetic_data` builds tensors in memory and skips the CSV/phenotype parsing. To run the
actual data path (`get_ABIDE_*_subject`, `preprocess_df_*`, `PathwayDataset`, FastSurfer
normalization) on throwaway data, generate a tiny on-disk dummy dataset and point the loaders at
it with the `CALM_DUMMY_DATA` env var:

```bash
python3 make_dummy_data.py                 # writes ./dummy_data (≈0.7 MB)

cd code
# Stage 2 (alignment) end-to-end on the real loaders:
CALM_DUMMY_DATA=../dummy_data python3 main.py --dataset ACE --n_epochs 2 \
    --batch_size 8 --output_features 6 --normalization layer \
    --use_mmd_alignment --genetics_input_channels 6 --not_write_tensorboard

# Stage 1 (either encoder):
CALM_DUMMY_DATA=../dummy_data python3 train_stage1.py --modality imaging \
    --dataset ACE --pretrain_epochs 2 --batch_size 8 --output_features 6 \
    --normalization layer --imaging_feature_noise 0.15 --stage1_output_dir /tmp/ck
```

`make_dummy_data.py` writes the CSVs + placeholder NIfTIs in the exact layout the loaders expect
(246 ROIs × 4 features, 177 pathways). When `CALM_DUMMY_DATA` is unset, the SCC paths in
`const.py` are used unchanged. Split sizes are multiples of `--batch_size 8` on purpose (the
alignment MMD pairs an imaging and a genetics batch each step and needs equal batch counts).

## Running on real data — paths to change

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

## Full 5-fold run (paper hyperparameters, two stages)

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

## Repo layout

- `code/main.py` — CALM alignment training (the entry point).
- `code/models/` — `genetics_encoder.py` (per-entity `PathwayEncoder`, reused as the
  per-ROI imaging encoder `E_I`), `alignment_model.py` (`SharedLatentProjector`), `losses.py`.
- `code/train_stage1.py` — Stage-1 encoder pretraining; pick the modality with
  `--modality {imaging,genetics}` (per-ROI imaging `E_I` or per-pathway genetics `E_G`).
- `code/utils/` — `const.py` (paths), `add_argument.py` (CLI flags), `utils.py` (loaders).
- `job_scripts/` — `stage1_imaging.sh`, `stage1_genetics.sh`, `stage2_alignment.sh` (5-fold launchers).

## Related work

- Learning *Unseen* Modality Interaction (NeurIPS '23); Everything at Once — multi-modal fusion
  transformer (CVPR '22); text-video embedding from incomplete data (arXiv '18); MMD-based
  Multiple Kernel Learning for incomplete-modality neuroimaging; DecAlign — hierarchical
  cross-modal alignment (arXiv '24).
