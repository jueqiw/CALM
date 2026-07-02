from typing import Tuple, Optional, Dict
from argparse import ArgumentParser, Namespace
from pathlib import Path
import os
import sys
import pickle
import random
import gc

# Fix for PyTorch 1.9.0 + Python 3.8+ compatibility
try:
    import distutils.version
except AttributeError:
    import distutils
    from packaging import version as packaging_version

    distutils.version = packaging_version

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedKFold
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

try:
    from torch.cuda.amp import autocast, GradScaler

    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False
    print("Warning: Mixed precision not available, falling back to FP32")
import matplotlib.pyplot as plt
import seaborn as sns
from monai.data import DataLoader
from monai.utils import set_determinism
from models.losses import contrastive_loss, mix_rbf_mmd2_and_ratio
from models.genetics_encoder import PathwayEncoder
from models.alignment_model import SharedLatentProjector
from utils.utils import (
    seed_everything,
    normalize_data,
    normalize_fastsurfer_features,
    add_log,
    save_result_dataframe,
    preprocess_df_ACE,
    preprocess_df_AD,
    preprocess_df_SSC,
    PathwayDataset,
    get_ABIDE_I_subject,
    get_ABIDE_II_subject,
    get_ACE_subjects,
    visualize_3d_mri,
    process_mri_patches_with_encoder,
    visualize_patch_structure,
)
from utils.add_argument import add_argument
from utils.scheduler import setup_training_components
from utils.const import (
    RESULT_FOLDER,
    ACE_FILE,
    ACE_FILE_with_relatedness,
    ADNI_FILE,
    CROSS_VAL_INDEX_ACE,
    CROSS_VAL_INDEX_ADNI,
    TENSORBOARD_CROSS_MODALITY,
    ACE_PHENOTYPE,
    SSC_FILE,
)

torch.multiprocessing.set_sharing_strategy("file_system")

# Optional UMAP for visualization
try:
    import umap.umap_ as umap
except ImportError:
    umap = None


def compute_covariance_matrix(features: torch.Tensor) -> Optional[torch.Tensor]:
    """
    features: [N, D]
    """
    n_samples = features.size(0)
    if n_samples <= 1:
        return None
    centered = features - features.mean(dim=0, keepdim=True)
    cov = torch.matmul(centered.t(), centered) / (n_samples - 1)
    return cov


def class_conditional_coral_loss(
    z_img: torch.Tensor,
    labels_img: torch.Tensor,
    z_gen: torch.Tensor,
    labels_gen: torch.Tensor,
):
    """
    Match class-wise covariance statistics between modalities.
    Returns: (total_loss, coral_asd, coral_control)
    """
    device = z_img.device
    losses = []
    coral_asd = torch.tensor(0.0, device=device)
    coral_control = torch.tensor(0.0, device=device)

    classes_img = torch.unique(labels_img)
    classes_gen = torch.unique(labels_gen)
    common_classes = torch.tensor(
        sorted(set(classes_img.tolist()) & set(classes_gen.tolist())),
        device=device,
        dtype=labels_img.dtype,
    )

    if common_classes.numel() == 0:
        return torch.tensor(0.0, device=device), coral_asd, coral_control

    for cls in common_classes:
        img_mask = labels_img == cls
        gen_mask = labels_gen == cls
        if img_mask.sum() <= 1 or gen_mask.sum() <= 1:
            continue
        cov_img = compute_covariance_matrix(z_img[img_mask])
        cov_gen = compute_covariance_matrix(z_gen[gen_mask])
        if cov_img is None or cov_gen is None:
            continue
        cls_loss = F.mse_loss(cov_img, cov_gen)
        losses.append(cls_loss)

        # Track per-class CORAL loss (0=Control, 1=ASD)
        if int(cls.item()) == 0:
            coral_control = cls_loss
        elif int(cls.item()) == 1:
            coral_asd = cls_loss

    if not losses:
        return torch.tensor(0.0, device=device), coral_asd, coral_control

    total_loss = torch.stack(losses).mean()
    return total_loss, coral_asd, coral_control


def compute_class_prototypes(
    z: torch.Tensor, labels: torch.Tensor
) -> Dict[int, torch.Tensor]:
    """
    Compute class prototypes by averaging embeddings per label.
    """
    prototypes: Dict[int, torch.Tensor] = {}
    for cls in torch.unique(labels):
        mask = labels == cls
        if mask.sum() == 0:
            continue
        prototypes[int(cls.item())] = z[mask].mean(dim=0)
    return prototypes


def orthogonality_regularizer(W: torch.Tensor) -> torch.Tensor:
    """
    Encourage W^T W ≈ I.
    """
    WWt = torch.matmul(W, W.t())
    I = torch.eye(WWt.size(0), device=W.device, dtype=W.dtype)
    return F.mse_loss(WWt, I)


def get_next_batch(loader_iter, loader):
    try:
        batch = next(loader_iter)
    except StopIteration:
        loader_iter = iter(loader)
        batch = next(loader_iter)
    return batch, loader_iter


def init_alignment_metrics():
    return {
        "total": 0.0,
        "img_cls": 0.0,
        "gen_cls": 0.0,
        "coral": 0.0,
        "mmd": 0.0,
        "coral_asd": 0.0,
        "coral_control": 0.0,
        "contrast": 0.0,
        "orth": 0.0,
        "gen_recon": 0.0,
        "steps": 0,
    }


def accumulate_metrics(metrics: dict, losses: dict):
    for key in [
        "total",
        "img_cls",
        "gen_cls",
        "coral",
        "coral_asd",
        "coral_control",
        "contrast",
        "orth",
        "gen_recon",
    ]:
        metrics[key] += losses[key]
    metrics["steps"] += 1


def finalize_metrics(metrics: dict) -> dict:
    steps = max(metrics["steps"], 1)
    return {
        key: (value / steps if key != "steps" else steps)
        for key, value in metrics.items()
    }


def unwrap_data_parallel(model: nn.Module) -> nn.Module:
    """
    Access the underlying module when wrapped with DataParallel.
    """
    return model.module if isinstance(model, nn.DataParallel) else model


def _project_imaging_tokens(roi_latent, projector):
    # roi_latent: [B, d, n_rois] from the per-ROI encoder -> [B, n_rois, d]
    roi_tokens = roi_latent.transpose(1, 2).contiguous()
    projector_module = unwrap_data_parallel(projector)
    z_img_mats = projector_module.project_image_tokens(roi_tokens)
    return z_img_mats.view(z_img_mats.size(0), -1)


def _project_genetics_tokens(pathway_latent, projector):
    projector_module = unwrap_data_parallel(projector)
    z_gen_mats = projector_module.project_genetics_tokens(pathway_latent)
    return z_gen_mats.view(z_gen_mats.size(0), -1)


def forward_genetics_encoder(
    genetics_encoder: nn.Module, pathways: torch.Tensor, return_reconstruction: bool
):
    """
    Safe forward for DataParallel: fall back to single-GPU path when batch < num_devices.
    """
    if isinstance(genetics_encoder, nn.DataParallel):
        device_count = len(genetics_encoder.device_ids)
        if pathways.size(0) < device_count:
            base = genetics_encoder.module
            return base(modality=pathways, return_reconstruction=return_reconstruction)
    return genetics_encoder(
        modality=pathways, return_reconstruction=return_reconstruction
    )


# Number of FreeSurfer morphological features per ROI
# (volume, surface area, mean cortical thickness, std cortical thickness).
N_IMG_FEATURES = 4


def forward_imaging_encoder(
    image_encoder: nn.Module, img_feat: torch.Tensor, sigma: float, training: bool
):
    """
    Forward the per-ROI imaging encoder (a PathwayEncoder over ROI features).
    img_feat: [B, n_rois, N_IMG_FEATURES]. Adds Gaussian feature noise (std=sigma)
    on the imaging modality during training only (paper Sec. 2.4).
    Returns (latent [B, d, n_rois], logits). DataParallel-safe like the genetics path.
    """
    if training and sigma and sigma > 0:
        img_feat = img_feat + torch.randn_like(img_feat) * sigma
    if isinstance(image_encoder, nn.DataParallel):
        device_count = len(image_encoder.device_ids)
        if img_feat.size(0) < device_count:
            return image_encoder.module(
                modality=img_feat, return_reconstruction=False
            )
    return image_encoder(modality=img_feat, return_reconstruction=False)


def collect_latent_embeddings(
    image_encoder,
    image_classifier,
    genetics_encoder,
    projector,
    img_loaders,
    gen_loaders,
    device,
):
    """
    Gather latent embeddings for imaging and genetics across train/val/test loaders.
    Returns numpy arrays: embeddings, modalities, labels, splits
    """
    if img_loaders is None or gen_loaders is None:
        return None

    image_encoder.eval()
    image_classifier.eval()
    genetics_encoder.eval()
    projector.eval()

    zs = []
    modalities = []
    cohorts = []
    splits = []

    with torch.no_grad():
        # Imaging split-wise
        for split_name, loader in [
            ("train", img_loaders[0]),
            ("val", img_loaders[1]),
            ("test", img_loaders[2]),
        ]:
            if loader is None:
                continue
            for batch in loader:
                img_feat = batch["img_feat"].to(device).float()
                labels = batch["label"].detach().cpu().view(-1).numpy()
                img_latent, _ = forward_imaging_encoder(
                    image_encoder, img_feat, 0.0, training=False
                )
                z = _project_imaging_tokens(img_latent, projector).cpu().numpy()
                zs.append(z)
                n = z.shape[0]
                modalities.extend(["imaging"] * n)
                cohorts.extend(labels.tolist())
                splits.extend([split_name] * n)

        # Genetics split-wise
        for split_name, loader in [
            ("train", gen_loaders[0]),
            ("val", gen_loaders[1]),
            ("test", gen_loaders[2]),
        ]:
            if loader is None:
                continue
            for batch in loader:
                pathways = batch["pathway"].to(device).float()
                labels = batch["label"].detach().cpu().view(-1).numpy()
                pathway_latent, _ = forward_genetics_encoder(
                    genetics_encoder, pathways, return_reconstruction=False
                )
                z = _project_genetics_tokens(pathway_latent, projector).cpu().numpy()
                zs.append(z)
                n = z.shape[0]
                modalities.extend(["genetics"] * n)
                cohorts.extend(labels.tolist())
                splits.extend([split_name] * n)

    if not zs:
        return None

    embeddings = np.concatenate(zs, axis=0)
    modalities = np.array(modalities)
    cohorts = np.array(cohorts).reshape(-1)
    splits = np.array(splits)
    return embeddings, modalities, cohorts, splits


def log_latent_stats(embeddings, modalities, writer=None, epoch=-1):
    """
    Compute and optionally log basic stats (mean variance and L2 norm) per modality.
    """
    stats = {}
    for modality in ["imaging", "genetics"]:
        mask = modalities == modality
        if not np.any(mask):
            continue
        emb = embeddings[mask]
        stats[modality] = {
            "var_mean": float(emb.var(axis=0).mean()),
            "mean_abs": float(np.abs(emb).mean()),
            "l2_mean": float(np.linalg.norm(emb, axis=1).mean()),
        }
        print(
            f"[Latent stats][{modality}][epoch {epoch}] "
            f"var_mean={stats[modality]['var_mean']:.6f}, "
            f"mean_abs={stats[modality]['mean_abs']:.6f}, "
            f"l2_mean={stats[modality]['l2_mean']:.6f}"
        )
        if writer is not None:
            writer.add_scalar(
                f"latents/{modality}_var_mean", stats[modality]["var_mean"], epoch
            )
            writer.add_scalar(
                f"latents/{modality}_mean_abs", stats[modality]["mean_abs"], epoch
            )
            writer.add_scalar(
                f"latents/{modality}_l2_mean", stats[modality]["l2_mean"], epoch
            )
    return stats


def visualize_latent_distributions(
    embeddings, modalities, epoch, experiment_name, writer=None
):
    """
    Plot L2 norm distributions (per sample) and per-dimension variance for each modality.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=200)
    colors = {"imaging": "tab:blue", "genetics": "tab:orange"}

    # Shared bins for fair comparison
    all_l2 = np.linalg.norm(embeddings, axis=1)
    l2_bins = np.linspace(all_l2.min(), all_l2.max(), 40)
    var_dict = {}
    for modality in ["imaging", "genetics"]:
        mask = modalities == modality
        if not np.any(mask):
            continue
        emb = embeddings[mask]
        var_dict[modality] = emb.var(axis=0)
    if var_dict:
        concat_vars = np.concatenate(list(var_dict.values()))
        vmin, vmax = np.percentile(concat_vars, [0.5, 99.5])
        var_bins = np.linspace(vmin, vmax, 40)
    else:
        var_bins = np.linspace(0, 1, 40)

    for modality in ["imaging", "genetics"]:
        mask = modalities == modality
        if not np.any(mask):
            continue
        emb = embeddings[mask]
        l2_norms = np.linalg.norm(emb, axis=1)
        var_per_dim = emb.var(axis=0)

        axes[0].hist(
            l2_norms, bins=l2_bins, alpha=0.6, color=colors[modality], label=modality
        )
        axes[1].hist(
            var_per_dim,
            bins=var_bins,
            alpha=0.6,
            color=colors[modality],
            label=modality,
        )

    axes[0].set_title("Latent L2 norms (per sample)")
    axes[0].set_xlabel("L2 norm")
    axes[0].set_ylabel("Count")
    axes[0].legend(frameon=False)

    axes[1].set_title("Per-dim variance")
    axes[1].set_xlabel("Variance")
    axes[1].set_ylabel("Count")
    axes[1].legend(frameon=False)
    fig.tight_layout()

    out_dir = Path(RESULT_FOLDER) / experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"latent_stats_epoch_{epoch+1}.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    print(f"Saved latent distribution plot: {out_path}")

    if writer is not None:
        fig.canvas.draw()
        img_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_data = img_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        writer.add_image("alignment/latent_stats", img_data, epoch, dataformats="HWC")
    plt.close(fig)


def visualize_umap(
    embeddings,
    modalities,
    cohorts,
    splits,
    epoch,
    experiment_name,
    writer=None,
):
    """
    Fit UMAP on concatenated embeddings and save/optionally log the figure.
    Colors: cohort (ASD/CON); markers: modality (imaging/genetics).
    """
    if umap is None:
        print("UMAP not installed; skipping UMAP visualization.")
        return

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embed_2d = reducer.fit_transform(embeddings)

    cohort_labels = np.where(cohorts == 1, "ASD", "CON")
    # Distinct color/marker per modality+cohort for better separation
    combo_styles = {
        ("imaging", "CON"): {"color": "#1b9e77", "marker": "o"},
        ("imaging", "ASD"): {"color": "#d95f02", "marker": "^"},
        ("genetics", "CON"): {"color": "#7570b3", "marker": "s"},
        ("genetics", "ASD"): {"color": "#e7298a", "marker": "D"},
    }

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    for modality in np.unique(modalities):
        mask_mod = modalities == modality
        for cohort in ["CON", "ASD"]:
            mask_cohort = cohort_labels == cohort
            mask = mask_mod & mask_cohort
            if mask.sum() == 0:
                continue
            style = combo_styles.get(
                (modality, cohort), {"color": "gray", "marker": "o"}
            )
            ax.scatter(
                embed_2d[mask, 0],
                embed_2d[mask, 1],
                s=10,
                c=style["color"],
                marker=style["marker"],
                alpha=0.6,
                label=f"{modality}-{cohort}",
                edgecolors="none",
            )

    ax.set_title(f"UMAP (epoch {epoch+1})")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="best", fontsize=8, frameon=False)
    fig.tight_layout()

    # Save to results folder
    out_dir = Path(RESULT_FOLDER) / experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"umap_epoch_{epoch+1}.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    print(f"Saved UMAP visualization: {out_path}")

    if writer is not None:
        fig.canvas.draw()
        img_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_data = img_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        writer.add_image("alignment/umap", img_data, epoch, dataformats="HWC")
    plt.close(fig)


def compute_alignment_losses(
    img_batch: dict,
    gen_batch: dict,
    image_encoder: nn.Module,
    image_classifier: nn.Module,
    genetics_encoder: nn.Module,
    projector: SharedLatentProjector,
    criterion_img: nn.Module,
    criterion_gen: nn.Module,
    hparams: Namespace,
    device: torch.device,
    mmd_loss_fn=mix_rbf_mmd2_and_ratio,
):
    img_feat = img_batch["img_feat"].to(device).float()
    img_labels = img_batch["label"].to(device).view(-1)
    img_latent, img_logits = forward_imaging_encoder(
        image_encoder,
        img_feat,
        getattr(hparams, "imaging_feature_noise", 0.0),
        training=unwrap_data_parallel(image_encoder).training,
    )
    img_logits = img_logits.squeeze(-1)
    img_loss = criterion_img(img_logits, img_labels.float())

    pathways = gen_batch["pathway"].to(device).float()
    pathway_target = (
        pathways[..., 0] if pathways.dim() == 3 else pathways.squeeze(-1)
    )
    gen_labels = gen_batch["label"].to(device).view(-1)

    # Update to get reconstructed output
    genetics_encoder_module = unwrap_data_parallel(genetics_encoder)
    if genetics_encoder_module.use_reconstruction:
        pathway_latent, gen_logits, reconstructed = forward_genetics_encoder(
            genetics_encoder, pathways, return_reconstruction=True
        )
        recon_loss = genetics_encoder_module.reconstruction_loss_fn(
            reconstructed, pathway_target
        )
    else:
        pathway_latent, gen_logits = forward_genetics_encoder(
            genetics_encoder, pathways, return_reconstruction=False
        )
        recon_loss = torch.tensor(0.0, device=device)

    gen_logits = gen_logits.squeeze(-1)
    gen_loss = criterion_gen(gen_logits, gen_labels.float())

    # Tokens: per-ROI latent [B, d, n_rois] -> [B, n_rois, d]
    projector_module = unwrap_data_parallel(projector)
    roi_tokens = img_latent.transpose(1, 2).contiguous()
    z_img_mats = projector_module.project_image_tokens(roi_tokens)
    z_gen_mats = projector_module.project_genetics_tokens(pathway_latent)

    z_img_flat = z_img_mats.view(z_img_mats.size(0), -1)
    z_gen_flat = z_gen_mats.view(z_gen_mats.size(0), -1)
    img_labels_long = img_labels.long()
    gen_labels_long = gen_labels.long()

    # Ensure matched batch sizes for alignment losses (MMD/contrastive/CORAL)
    min_bs = min(z_img_flat.size(0), z_gen_flat.size(0))
    z_img_flat = z_img_flat[:min_bs]
    img_labels_long = img_labels_long[:min_bs]

    if getattr(hparams, "use_mmd_alignment", False):
        # When using MMD, treat alignment_coral_weight as the MMD weight; set coral-related terms to 0 for logging clarity
        sigma_list = [1, 2, 5, 10, 20, 40, 80]
        mmd_value, _, _ = mmd_loss_fn(z_img_flat, z_gen_flat, sigma_list, biased=True)
        coral = torch.tensor(0.0, device=device)
        coral_asd = torch.tensor(0.0, device=device)
        coral_control = torch.tensor(0.0, device=device)
    else:
        coral, coral_asd, coral_control = class_conditional_coral_loss(
            z_img_flat, img_labels_long, z_gen_flat, gen_labels_long
        )
        mmd_value = torch.tensor(0.0, device=device)
    if getattr(hparams, "align_projector_only", False):
        contrast = torch.tensor(0.0, device=device)
    else:
        contrast = contrastive_loss(
            z_img_flat,
            img_labels_long,
            z_gen_flat,
            gen_labels_long,
            lambda_param=hparams.alignment_lambda,
            tau=hparams.alignment_tau,
        )
    orth = orthogonality_regularizer(projector_module.W_I) + orthogonality_regularizer(
        projector_module.W_G
    )

    # alignment_coral_weight doubles as the MMD weight when use_mmd_alignment is enabled
    coral_like_weight = hparams.alignment_coral_weight
    total = (
        hparams.imaging_cls_weight * img_loss
        + hparams.genetics_cls_weight * gen_loss
        + coral_like_weight
        * (mmd_value if getattr(hparams, "use_mmd_alignment", False) else coral)
        + hparams.alignment_contrastive_weight * contrast
        + hparams.alignment_orthogonality_weight * orth
        + hparams.reconstruction_weight * recon_loss
    )

    return {
        "total": total,
        "img_cls": img_loss,
        "gen_cls": gen_loss,
        "coral": coral,
        "mmd": mmd_value,
        "coral_asd": coral_asd,
        "coral_control": coral_control,
        "contrast": contrast,
        "orth": orth,
        "gen_recon": recon_loss,
    }


def evaluate_alignment(
    image_encoder,
    image_classifier,
    genetics_encoder,
    projector,
    img_loader,
    gen_loader,
    criterion_img,
    criterion_gen,
    hparams,
    device,
    collect_ids: bool = False,
    mmd_loss_fn=mix_rbf_mmd2_and_ratio,
):
    image_encoder.eval()
    image_classifier.eval()
    genetics_encoder.eval()
    projector.eval()

    metrics = init_alignment_metrics()
    if img_loader is None or gen_loader is None:
        return finalize_metrics(metrics)

    # Collect predictions and labels for accuracy/AUC calculation
    img_preds_list = []
    img_labels_list = []
    gen_preds_list = []
    gen_labels_list = []

    # NEW: Collect ALL subject IDs with their correctness status
    img_results = {} if collect_ids else None  # {subject_id: is_correct}
    gen_results = {} if collect_ids else None  # {subject_id: is_correct}

    total_steps = max(len(img_loader), len(gen_loader))
    img_iter = iter(img_loader)
    gen_iter = iter(gen_loader)

    with torch.no_grad():
        for _ in range(total_steps):
            img_batch, img_iter = get_next_batch(img_iter, img_loader)
            gen_batch, gen_iter = get_next_batch(gen_iter, gen_loader)

            # Compute losses
            losses = compute_alignment_losses(
                img_batch,
                gen_batch,
                image_encoder,
                image_classifier,
                genetics_encoder,
                projector,
                criterion_img,
                criterion_gen,
                hparams,
                device,
                mmd_loss_fn=mmd_loss_fn,
            )
            loss_values = {k: losses[k].item() for k in losses}
            accumulate_metrics(metrics, loss_values)

            # Collect predictions for imaging
            img_feat = img_batch["img_feat"].to(device).float()
            img_labels = img_batch["label"].to(device).view(-1)
            img_labels_long = img_labels.long()
            _, img_logits = forward_imaging_encoder(
                image_encoder, img_feat, 0.0, training=False
            )
            img_logits = img_logits.squeeze(-1)
            img_probs = torch.sigmoid(img_logits)
            img_preds_list.append(img_probs.cpu())
            img_labels_list.append(img_labels.cpu())

            if collect_ids:
                img_pred_labels = (img_probs > 0.5).long()
                ids = img_batch.get(
                    "New ID", img_batch.get("subject_id", img_batch.get("ids", None))
                )
                if ids is not None:
                    if isinstance(ids, (list, tuple)):
                        ids_list = list(ids)
                    elif isinstance(ids, np.ndarray):
                        ids_list = ids.tolist()
                    else:
                        ids_list = [ids]

                    correctness = (img_pred_labels == img_labels_long).cpu().tolist()
                    for idx_local, is_correct in enumerate(correctness):
                        subject_id = str(ids_list[idx_local])
                        # Store the result (overwrite if duplicate, taking latest)
                        img_results[subject_id] = is_correct

            # Collect predictions for genetics
            pathways = gen_batch["pathway"].to(device).float()
            gen_labels = gen_batch["label"].to(device).view(-1)
            gen_labels_long = gen_labels.long()
            _, gen_logits = forward_genetics_encoder(
                genetics_encoder, pathways, return_reconstruction=False
            )
            gen_logits = gen_logits.squeeze(-1)
            gen_probs = torch.sigmoid(gen_logits)
            gen_preds_list.append(gen_probs.cpu())
            gen_labels_list.append(gen_labels.cpu())

            if collect_ids:
                gen_pred_labels = (gen_probs > 0.5).long()
                ids = gen_batch.get("ids", gen_batch.get("subject_id", None))
                if ids is not None:
                    if isinstance(ids, (list, tuple)):
                        ids_list = list(ids)
                    elif isinstance(ids, np.ndarray):
                        ids_list = ids.tolist()
                    else:
                        ids_list = [ids]

                    correctness = (gen_pred_labels == gen_labels_long).cpu().tolist()
                    for idx_local, is_correct in enumerate(correctness):
                        subject_id = str(ids_list[idx_local])
                        # Store the result (overwrite if duplicate, taking latest)
                        gen_results[subject_id] = is_correct

    # Compute accuracy and AUC
    from sklearn.metrics import roc_auc_score, accuracy_score

    img_preds = torch.cat(img_preds_list).numpy()
    img_labels = torch.cat(img_labels_list).numpy()
    gen_preds = torch.cat(gen_preds_list).numpy()
    gen_labels = torch.cat(gen_labels_list).numpy()

    metrics_final = finalize_metrics(metrics)

    # NEW: Compute cross-modality correctness statistics
    if collect_ids and img_results is not None and gen_results is not None:
        # Get sets of correct subject IDs
        img_correct_set = {sid for sid, correct in img_results.items() if correct}
        gen_correct_set = {sid for sid, correct in gen_results.items() if correct}

        # Compute overlaps
        both_correct = img_correct_set & gen_correct_set
        img_only_correct = img_correct_set - gen_correct_set
        gen_only_correct = gen_correct_set - img_correct_set

        # Store in metrics
        metrics_final["img_correct"] = len(img_correct_set)
        metrics_final["img_total"] = len(img_results)
        metrics_final["gen_correct"] = len(gen_correct_set)
        metrics_final["gen_total"] = len(gen_results)
        metrics_final["both_correct_subjects"] = len(both_correct)
        metrics_final["img_only_correct_subjects"] = len(img_only_correct)
        metrics_final["gen_only_correct_subjects"] = len(gen_only_correct)

        # Also store the ID lists for debugging
        metrics_final["img_correct_ids"] = list(img_correct_set)
        metrics_final["gen_correct_ids"] = list(gen_correct_set)
        metrics_final["both_correct_ids"] = list(both_correct)

    # Imaging metrics
    try:
        metrics_final["img_auc"] = roc_auc_score(img_labels, img_preds)
        metrics_final["img_acc"] = accuracy_score(
            img_labels, (img_preds > 0.5).astype(int)
        )
    except:
        metrics_final["img_auc"] = 0.0
        metrics_final["img_acc"] = 0.0

    # Genetics metrics
    try:
        metrics_final["gen_auc"] = roc_auc_score(gen_labels, gen_preds)
        metrics_final["gen_acc"] = accuracy_score(
            gen_labels, (gen_preds > 0.5).astype(int)
        )
    except:
        metrics_final["gen_auc"] = 0.0
        metrics_final["gen_acc"] = 0.0

    return metrics_final


def create_folder(hparams: Namespace) -> Path:
    result_fold_path = RESULT_FOLDER / f"{hparams.experiment_name}"
    if not os.path.exists(result_fold_path):
        os.makedirs(result_fold_path)
        print(f"Create folder: {result_fold_path}!")

    return result_fold_path


def load_synthetic_data(
    hparams,
    n_rois: int = 246,
    n_pathways: int = 177,
    n_train: int = 128,
    n_val: int = 32,
    n_test: int = 32,
):
    """Pseudo data for an end-to-end smoke test (no real data / SCC paths needed).

    Yields the same batch schema as the real loaders:
      imaging  -> {"img_feat": [n_rois, N_IMG_FEATURES], "label"}
      genetics -> {"pathway":  [n_pathways, C_gen],       "label"}
    Returns the two loader-triples plus (n_rois, n_pathways).
    """
    set_determinism(seed=42)
    c_gen = getattr(hparams, "genetics_input_channels", 1)
    print("\n*** USING SYNTHETIC PSEUDO DATA (smoke test) ***")

    def img_split(n):
        return [
            {
                "img_feat": torch.randn(n_rois, N_IMG_FEATURES),
                "label": torch.tensor(int(i % 2), dtype=torch.long),
            }
            for i in range(n)
        ]

    def gen_split(n):
        return [
            {
                "pathway": torch.randn(n_pathways, c_gen),
                "label": torch.tensor(int(i % 2), dtype=torch.long),
            }
            for i in range(n)
        ]

    bs = hparams.batch_size
    img_loaders = (
        DataLoader(img_split(n_train), batch_size=bs, shuffle=True),
        DataLoader(img_split(n_val), batch_size=bs, shuffle=False),
        DataLoader(img_split(n_test), batch_size=bs, shuffle=False),
    )
    gen_loaders = (
        DataLoader(gen_split(n_train), batch_size=bs, shuffle=True),
        DataLoader(gen_split(n_val), batch_size=bs, shuffle=False),
        DataLoader(gen_split(n_test), batch_size=bs, shuffle=False),
    )
    return img_loaders, gen_loaders, n_rois, n_pathways


class _ROIFeatureDataset(torch.utils.data.Dataset):
    """Per-subject ROI morphological features for the CALM imaging encoder.

    Each item: {"img_feat": FloatTensor[n_rois, N_IMG_FEATURES], "label": LongTensor}.
    Reads the (already z-scored) flat "image_features" vector carried on each sample.
    """

    def __init__(self, samples, n_rois):
        self.samples = samples
        self.n_rois = n_rois

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        feat = np.asarray(sample["image_features"], dtype=np.float32).reshape(
            self.n_rois, N_IMG_FEATURES
        )
        return {
            "img_feat": torch.from_numpy(feat),
            "label": torch.tensor(int(sample["label"]), dtype=torch.long),
        }


def load_imaging_data(hparams: Namespace):
    """
    Load imaging data with same splits as encoder_images.py.
    Imaging input is per-ROI FreeSurfer morphological features ([n_rois, 4]),
    not 3D volumes. Returns train/val/test dataloaders and n_rois.
    """
    # Get all imaging data samples
    samples_ABIDE_I = get_ABIDE_I_subject()
    samples_ABIDE_II = get_ABIDE_II_subject()
    ACE_img_samples = get_ACE_subjects()

    # Combine ABIDE I and II for train/val split
    abide_samples = samples_ABIDE_I + samples_ABIDE_II

    print(f"\n{'='*70}")
    print("IMAGING DATA LOADING")
    print(f"{'='*70}")
    print(f"Number of ABIDE I samples: {len(samples_ABIDE_I)}")
    print(f"Number of ABIDE II samples: {len(samples_ABIDE_II)}")
    print(f"Number of ACE imaging samples: {len(ACE_img_samples)}")

    # Set determinism for reproducible results
    set_determinism(seed=42)
    abide_shuffled = abide_samples.copy()
    random.shuffle(abide_shuffled)

    # Split ABIDE samples into train/val (0.9 for training, 0.1 for validation)
    split_idx = int(0.9 * len(abide_shuffled))
    train_img_samples = abide_shuffled[:split_idx]
    val_img_samples = abide_shuffled[split_idx:]
    test_img_samples = ACE_img_samples

    # Normalize FastSurfer features using training set statistics
    train_img_samples, val_img_samples, test_img_samples, feat_mean, feat_std = (
        normalize_fastsurfer_features(
            train_img_samples, val_img_samples, test_img_samples
        )
    )

    # Test mode: reduce dataset size for quick testing
    if hasattr(hparams, "test_mode") and hparams.test_mode:
        original_train_size = len(train_img_samples)
        original_val_size = len(val_img_samples)
        original_test_size = len(test_img_samples)

        test_ratio = (
            hparams.test_mode_ratio if hasattr(hparams, "test_mode_ratio") else 0.1
        )
        train_img_samples = train_img_samples[
            : int(len(train_img_samples) * test_ratio)
        ]
        val_img_samples = val_img_samples[: int(len(val_img_samples) * test_ratio)]
        test_img_samples = test_img_samples[: int(len(test_img_samples) * test_ratio)]

        print(f"\nTEST MODE ENABLED (ratio: {test_ratio})")
        print(
            f"  Imaging Training: {original_train_size} → {len(train_img_samples)} samples"
        )
        print(
            f"  Imaging Validation: {original_val_size} → {len(val_img_samples)} samples"
        )
        print(f"  Imaging Test: {original_test_size} → {len(test_img_samples)} samples")
        print("=" * 60)

    # Print imaging label distributions
    train_labels_0 = sum(1 for sample in train_img_samples if sample["label"] == 0)
    train_labels_1 = sum(1 for sample in train_img_samples if sample["label"] == 1)
    val_labels_0 = sum(1 for sample in val_img_samples if sample["label"] == 0)
    val_labels_1 = sum(1 for sample in val_img_samples if sample["label"] == 1)
    test_labels_0 = sum(1 for sample in test_img_samples if sample["label"] == 0)
    test_labels_1 = sum(1 for sample in test_img_samples if sample["label"] == 1)

    print(f"\nImaging Training set: {train_labels_0} class-0, {train_labels_1} class-1")
    print(f"Imaging Validation set: {val_labels_0} class-0, {val_labels_1} class-1")
    print(f"Imaging Test set: {test_labels_0} class-0, {test_labels_1} class-1")
    print(f"{'='*70}\n")

    # Build per-ROI feature tensors from the (already z-scored) flat FastSurfer
    # vector carried on each sample. n_rois is derived from the data.
    feat_len = len(train_img_samples[0]["image_features"])
    assert feat_len % N_IMG_FEATURES == 0, (
        f"image_features length {feat_len} not divisible by N_IMG_FEATURES={N_IMG_FEATURES}"
    )
    n_rois = feat_len // N_IMG_FEATURES
    # NOTE: flat vector reshaped ROI-major -> [n_rois, N_IMG_FEATURES] (all 4
    # morphological features of ROI 0, then ROI 1, ...). This assumes the
    # Brainnetome group-stats CSV is column-ordered ROI-major; if it is
    # feature-major, change _ROIFeatureDataset.__getitem__ to
    # .reshape(N_IMG_FEATURES, n_rois).T. Confirm against the CSV header on SCC.
    print(f"Imaging ROI feature tensor per subject: [{n_rois}, {N_IMG_FEATURES}]")

    train_img_loader = DataLoader(
        _ROIFeatureDataset(train_img_samples, n_rois),
        batch_size=hparams.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_img_loader = DataLoader(
        _ROIFeatureDataset(val_img_samples, n_rois),
        batch_size=min(hparams.batch_size, 16),
        shuffle=False,
        num_workers=0,
    )
    test_img_loader = DataLoader(
        _ROIFeatureDataset(test_img_samples, n_rois),
        batch_size=min(hparams.batch_size, 16),
        shuffle=False,
        num_workers=0,
    )

    return train_img_loader, val_img_loader, test_img_loader, n_rois


def load_pretrained_imaging_encoder(hparams: Namespace, n_rois: int, device):
    """
    Build the per-ROI imaging encoder E_I (a PathwayEncoder over ROI morphological
    features) and optionally load a Stage-1 checkpoint. Mirrors the genetics loader.
    Returns: image_encoder, image_classifier (Identity — the encoder has its own head).
    """
    print(f"\n{'='*70}")
    print("BUILDING IMAGING ENCODER (per-ROI features)")
    print(f"{'='*70}")

    img_pretrain_folder = Path(
        "/projectnb/ace-genetics/jueqiw/experiment/CrossModalityLearning/model_weight/img"
    )

    # E_I: per-ROI MLP over [n_rois, N_IMG_FEATURES] morphological features.
    # Same per-entity architecture as the genetics PathwayEncoder.
    image_encoder = PathwayEncoder(
        n_pathway=n_rois,
        classifier_latent_dim=getattr(hparams, "classifier_latent_dim", 32),
        normalization=getattr(hparams, "normalization", "layer"),
        relu_at_coattention=getattr(hparams, "relu_at_coattention", False),
        input_channels=N_IMG_FEATURES,
        encoder_hidden_dim_1=getattr(hparams, "encoder_hidden_dim_1", 32),
        encoder_hidden_dim_2=getattr(hparams, "encoder_hidden_dim_2", 16),
        encoder_hidden_dim_3=getattr(hparams, "encoder_hidden_dim_3", None),
        output_features=hparams.output_features,
        use_reconstruction=False,
        pathway_dropout=getattr(hparams, "pathway_dropout", 0.0),
        encoder_dropout=getattr(hparams, "encoder_dropout", 0.0),
        classifier_dropout=getattr(hparams, "classifier_drop_out", 0.5),
    ).to(device)

    # The encoder carries its own classifier head; keep a no-op classifier so the
    # downstream (encoder, classifier) call signatures stay unchanged.
    image_classifier = nn.Identity()

    if (
        hasattr(hparams, "pretrained_imaging_checkpoint")
        and hparams.pretrained_imaging_checkpoint
    ):
        ckpt_path = img_pretrain_folder / hparams.pretrained_imaging_checkpoint
        print(f"Loading pretrained imaging encoder from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        state = checkpoint.get("model_state_dict", checkpoint)
        missing_keys, unexpected_keys = image_encoder.load_state_dict(
            state, strict=False
        )
        if missing_keys:
            print(f"⚠️  Missing keys (random init): {missing_keys}")
        if unexpected_keys:
            print(f"⚠️  Unexpected keys (ignored): {unexpected_keys}")
        print("✓ Successfully loaded pretrained imaging encoder")

        if getattr(hparams, "freeze_encoder_decoder", False):
            print(f"\n{'='*70}")
            print("STAGE 2: Freezing imaging encoder (classifier head included)")
            for param in image_encoder.parameters():
                param.requires_grad = False
            print(f"{'='*70}\n")
        else:
            print("Note: --freeze_encoder_decoder not set, encoder remains trainable")

    else:
        print("⚠️  No pretrained checkpoint specified. Training from scratch.")

    print(f"{'='*70}\n")
    return image_encoder, image_classifier


def load_pretrained_genetics_encoder_classifier(
    hparams: Namespace, n_pathways: int, device
):
    """
    Load pretrained genetics encoder from checkpoint
    Returns: genetics_encoder
    """
    print(f"\n{'='*70}")
    print("LOADING PRETRAINED GENETICS ENCODER")
    print(f"{'='*70}")

    genetics_pretrain_folder = Path(
        "/projectnb/ace-genetics/jueqiw/experiment/CrossModalityLearning/model_weight/genetics"
    )

    checkpoint = None
    checkpoint_path = None
    checkpoint_input_channels = getattr(hparams, "genetics_input_channels", 1)
    if (
        hasattr(hparams, "pretrained_genetics_checkpoint")
        and hparams.pretrained_genetics_checkpoint
    ):
        checkpoint_path = (
            genetics_pretrain_folder / hparams.pretrained_genetics_checkpoint
        )
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "input_channels" in checkpoint:
            if checkpoint_input_channels != checkpoint["input_channels"]:
                print(
                    f"Adjusting genetics_input_channels from {checkpoint_input_channels} to "
                    f"{checkpoint['input_channels']} to match checkpoint."
                )
            checkpoint_input_channels = checkpoint["input_channels"]

    hparams.genetics_input_channels = checkpoint_input_channels

    # Create genetics encoder (same architecture as encoder_genetics.py)
    genetics_encoder_classifier = PathwayEncoder(
        n_pathway=n_pathways,  # 177
        classifier_latent_dim=(
            hparams.classifier_latent_dim
            if hasattr(hparams, "classifier_latent_dim")
            else 32
        ),
        normalization=(
            hparams.normalization if hasattr(hparams, "normalization") else "layer"
        ),
        relu_at_coattention=(
            hparams.relu_at_coattention
            if hasattr(hparams, "relu_at_coattention")
            else False
        ),
        input_channels=checkpoint_input_channels,
        encoder_hidden_dim_1=(
            hparams.encoder_hidden_dim_1
            if hasattr(hparams, "encoder_hidden_dim_1")
            else 8
        ),
        encoder_hidden_dim_2=(
            hparams.encoder_hidden_dim_2
            if hasattr(hparams, "encoder_hidden_dim_2")
            else 4
        ),
        encoder_hidden_dim_3=(
            hparams.encoder_hidden_dim_3
            if hasattr(hparams, "encoder_hidden_dim_3")
            else None
        ),
        output_features=hparams.output_features,  # Should be 8
        use_reconstruction=hparams.use_reconstruction,
        pathway_dropout=hparams.pathway_dropout,
        encoder_dropout=hparams.encoder_dropout,
        classifier_dropout=hparams.classifier_drop_out,
    ).to(device)

    # Load pretrained weights if specified
    if checkpoint is not None:
        print(f"Loading pretrained weights from: {hparams.pretrained_genetics_checkpoint}")

        # The genetics encoder checkpoint structure (from encoder_genetics.py):
        # checkpoint = {'model_state_dict': model.state_dict(), 'epoch': ..., 'n_pathway': ..., 'output_features': ...}
        # Load with strict=False to handle architecture changes
        missing_keys, unexpected_keys = genetics_encoder_classifier.load_state_dict(
            checkpoint["model_state_dict"], strict=False
        )

        if missing_keys:
            print(
                f"⚠️  Missing keys in checkpoint (will use random initialization): {missing_keys}"
            )
        if unexpected_keys:
            print(
                f"⚠️  Unexpected keys in checkpoint (will be ignored): {unexpected_keys}"
            )

        print(
            f"Successfully loaded pretrained genetics encoder from epoch {checkpoint.get('epoch', 'unknown')}"
        )

        # Freeze genetics encoder if specified (similar to imaging encoder)
        if (
            hasattr(hparams, "freeze_encoder_decoder")
            and hparams.freeze_encoder_decoder
        ):
            print(f"\n{'='*70}")
            print(f"STAGE 2: Freezing genetics encoder")
            for param in genetics_encoder_classifier.parameters():
                param.requires_grad = False
            print(f"  All genetics encoder parameters frozen")
            print(f"{'='*70}\n")
        else:
            print(
                f"Note: --freeze_encoder_decoder not set, genetics encoder remains trainable"
            )

    else:
        print("⚠️  No pretrained checkpoint specified. Training from scratch.")

    print(f"{'='*70}\n")
    return genetics_encoder_classifier


def load_genetics_data(hparams: Namespace):
    """
    Load genetics data with same splits as encoder_genetics.py
    SSC for train/val, ACE for test
    Returns train/val/test dataloaders for genetics
    """
    # Load SSC pathway data for training
    SSC_pathway_data = pd.read_csv(SSC_FILE)
    SSC_pathway_data = SSC_pathway_data.drop(columns=SSC_pathway_data.columns[0])
    SSC_pathway, SSC_label = preprocess_df_SSC(SSC_pathway_data)

    # Load ACE pathway data for testing
    ACE_pathway_data = pd.read_csv(ACE_FILE_with_relatedness)
    ACE_pathway_data = ACE_pathway_data.drop(columns=ACE_pathway_data.columns[0])
    ACE_img, ACE_pathway, ACE_label, ACE_father_site_ids, ACE_new_ids = (
        preprocess_df_ACE(ACE_pathway_data)
    )

    print(f"\n{'='*70}")
    print("GENETICS DATA LOADING")
    print(f"{'='*70}")
    print(f"SSC pathway shape: {SSC_pathway.shape} (samples × pathways)")
    print(f"ACE pathway shape: {ACE_pathway.shape} (samples × pathways)")

    # Set determinism for reproducible results
    set_determinism(seed=42)

    train_indices, val_indices = train_test_split(
        range(len(SSC_pathway)), test_size=0.1, stratify=SSC_label, random_state=42
    )

    X_train_pathway = SSC_pathway.iloc[train_indices]
    y_train = SSC_label.iloc[train_indices]
    X_val_pathway = SSC_pathway.iloc[val_indices]
    y_val = SSC_label.iloc[val_indices]
    X_test_pathway = ACE_pathway
    y_test = ACE_label

    print(
        f"\nGenetics Training set: {(y_train==0).sum()} class-0, {(y_train==1).sum()} class-1"
    )
    print(
        f"Genetics Validation set: {(y_val==0).sum()} class-0, {(y_val==1).sum()} class-1"
    )
    print(
        f"Genetics Test set: {(y_test==0).sum()} class-0, {(y_test==1).sum()} class-1"
    )
    print(f"{'='*70}\n")

    # Create genetics dataloaders
    train_gen_dataset = PathwayDataset(X_train_pathway, y_train, hparams=hparams)
    val_gen_dataset = PathwayDataset(X_val_pathway, y_val, hparams=hparams)
    test_gen_dataset = PathwayDataset(
        X_test_pathway, y_test, hparams=hparams, ids=ACE_new_ids
    )

    train_gen_loader = DataLoader(
        train_gen_dataset, batch_size=hparams.batch_size, shuffle=True
    )
    val_gen_loader = DataLoader(
        val_gen_dataset, batch_size=hparams.batch_size, shuffle=False
    )
    test_gen_loader = DataLoader(
        test_gen_dataset, batch_size=hparams.batch_size, shuffle=False
    )

    # Return number of pathways for model initialization
    n_pathways = X_train_pathway.shape[1]
    return train_gen_loader, val_gen_loader, test_gen_loader, n_pathways


def main(hparams: Namespace, writer: SummaryWriter):
    """
    Main training function for cross-modality alignment
    """
    print("\n" + "=" * 80)
    print("CROSS-MODALITY ALIGNMENT TRAINING")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if getattr(hparams, "use_synthetic_data", False):
        (train_img_loader, val_img_loader, test_img_loader), (
            train_gen_loader,
            val_gen_loader,
            test_gen_loader,
        ), n_rois, n_pathways = load_synthetic_data(hparams)
    else:
        train_img_loader, val_img_loader, test_img_loader, n_rois = load_imaging_data(
            hparams
        )
        train_gen_loader, val_gen_loader, test_gen_loader, n_pathways = (
            load_genetics_data(hparams)
        )

    image_encoder, image_classifier = load_pretrained_imaging_encoder(
        hparams, n_rois, device
    )
    image_encoder.train()

    # Imaging encoder fine-tuning optimizer (paper Sec. 2.4: encoder finetune lr 1e-5).
    imaging_optimizer = None
    imaging_scheduler = None
    if getattr(hparams, "align_projector_only", False):
        print(
            "align_projector_only enabled: freezing imaging encoder; no imaging optimizer."
        )
    else:
        img_params = [p for p in image_encoder.parameters() if p.requires_grad]
        if img_params:
            imaging_optimizer = torch.optim.AdamW(
                img_params,
                lr=getattr(hparams, "encoder_finetune_lr", 1e-5),
                weight_decay=getattr(hparams, "genetics_weight_decay", 1e-4),
            )

    genetics_encoder = load_pretrained_genetics_encoder_classifier(
        hparams, n_pathways, device
    )
    if not getattr(hparams, "freeze_encoder_decoder", False):
        for param in genetics_encoder.parameters():
            param.requires_grad = True
    freeze_gen_layers = getattr(hparams, "freeze_genetics_layers", 0)
    if freeze_gen_layers > 0:
        layer_names = [
            "encoder_linear_1",
            "encoder_linear_2",
            "encoder_linear_3",
            "encoder_linear_out",
        ]
        actual_encoder = genetics_encoder
        if isinstance(genetics_encoder, nn.DataParallel):
            actual_encoder = genetics_encoder.module
        for idx, name in enumerate(layer_names):
            if freeze_gen_layers >= idx + 1 and hasattr(actual_encoder, name):
                param_obj = getattr(actual_encoder, name)
                if isinstance(param_obj, torch.nn.Parameter):
                    param_obj.requires_grad = False
                elif hasattr(param_obj, "requires_grad"):
                    param_obj.requires_grad = False
        print(
            f"freeze_genetics_layers={freeze_gen_layers}: froze first {freeze_gen_layers} encoder weight matrices"
        )

    # Optional: freeze encoders/classifiers and train projector only
    if getattr(hparams, "align_projector_only", False):
        hparams.imaging_cls_weight = 0.0
        hparams.genetics_cls_weight = 0.0
        hparams.reconstruction_weight = 0.0
        print(
            "Freezing imaging/genetics encoders and classifiers; zeroing cls/recon weights."
        )
        for p in image_encoder.parameters():
            p.requires_grad = False
        for p in image_classifier.parameters():
            p.requires_grad = False
        for p in genetics_encoder.parameters():
            p.requires_grad = False

    projector = SharedLatentProjector(
        latent_dim=hparams.output_features, n_rois=n_rois, n_pathways=n_pathways
    ).to(device)

    # Wrap models with DataParallel for multi-GPU training
    if getattr(hparams, "multi_gpu", False) and torch.cuda.device_count() > 1:
        print(f"\n{'='*70}")
        print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
        print(f"{'='*70}")
        image_encoder = nn.DataParallel(image_encoder)
        image_classifier = nn.DataParallel(image_classifier)
        genetics_encoder = nn.DataParallel(genetics_encoder)
        projector = nn.DataParallel(projector)
        print(f"✓ All models wrapped with DataParallel")
    else:
        print(f"\nRunning on single GPU: {device}")

    # Print parameter counts (same style as encoder_images.py)
    print("\n=== Checking which parameters are frozen/unfrozen ===")

    # Imaging encoder
    frozen_count_img = 0
    unfrozen_count_img = 0
    for name, param in image_encoder.named_parameters():
        status = "UNFROZEN" if param.requires_grad else "FROZEN"
        print(f"  {name}: {status}")
        if param.requires_grad:
            unfrozen_count_img += 1
        else:
            frozen_count_img += 1

    print(
        f"\nSummary: {frozen_count_img} frozen, {unfrozen_count_img} unfrozen image encoder parameters"
    )

    # Imaging classifier
    frozen_count_clf = 0
    unfrozen_count_clf = 0
    for name, param in image_classifier.named_parameters():
        status = "UNFROZEN" if param.requires_grad else "FROZEN"
        print(f"  {name}: {status}")
        if param.requires_grad:
            unfrozen_count_clf += 1
        else:
            frozen_count_clf += 1

    print(
        f"\nSummary: {frozen_count_clf} frozen, {unfrozen_count_clf} unfrozen image classifier parameters"
    )

    # Genetics encoder
    frozen_count_gen = 0
    unfrozen_count_gen = 0
    for name, param in genetics_encoder.named_parameters():
        status = "UNFROZEN" if param.requires_grad else "FROZEN"
        print(f"  {name}: {status}")
        if param.requires_grad:
            unfrozen_count_gen += 1
        else:
            frozen_count_gen += 1

    print(
        f"\nSummary: {frozen_count_gen} frozen, {unfrozen_count_gen} unfrozen genetics encoder parameters"
    )
    print("=" * 60)

    total_encoder_params = sum(p.numel() for p in image_encoder.parameters())
    trainable_encoder_params = sum(
        p.numel() for p in image_encoder.parameters() if p.requires_grad
    )
    total_classifier_params = sum(p.numel() for p in image_classifier.parameters())
    trainable_classifier_params = sum(
        p.numel() for p in image_classifier.parameters() if p.requires_grad
    )
    total_genetics_params = sum(p.numel() for p in genetics_encoder.parameters())
    trainable_genetics_params = sum(
        p.numel() for p in genetics_encoder.parameters() if p.requires_grad
    )

    print(
        f"Encoder: {trainable_encoder_params}/{total_encoder_params} trainable params"
    )
    print(
        f"Classifier: {trainable_classifier_params}/{total_classifier_params} trainable params"
    )
    print(
        f"Genetics Encoder: {trainable_genetics_params}/{total_genetics_params} trainable params"
    )

    criterion_img = nn.BCEWithLogitsLoss()
    criterion_gen = nn.BCEWithLogitsLoss()
    if getattr(hparams, "align_projector_only", False):
        genetics_optimizer = torch.optim.AdamW(
            projector.parameters(),
            lr=hparams.genetics_learning_rate,
            weight_decay=hparams.genetics_weight_decay,
        )
    else:
        genetics_optimizer = torch.optim.AdamW(
            list(genetics_encoder.parameters()) + list(projector.parameters()),
            lr=hparams.genetics_learning_rate,
            weight_decay=hparams.genetics_weight_decay,
        )

    print("✓ Data loading and model initialization complete!")

    # Reset to training mode
    image_encoder.train()
    genetics_encoder.train()

    print("Starting alignment training...\n")

    global_step = 0

    # UMAP at epoch 0 (before training)
    if umap is not None and writer is not None:
        loaders_tuple_img = (train_img_loader, val_img_loader, test_img_loader)
        loaders_tuple_gen = (train_gen_loader, val_gen_loader, test_gen_loader)
        latent_pack = collect_latent_embeddings(
            image_encoder,
            image_classifier,
            genetics_encoder,
            projector,
            loaders_tuple_img,
            loaders_tuple_gen,
            device,
        )
        if latent_pack is not None:
            visualize_umap(
                *latent_pack,
                epoch=-1,  # label as pretrain/epoch0
                experiment_name=hparams.experiment_name,
                writer=writer,
            )
        else:
            print("Skipping UMAP at epoch 0: no embeddings collected.")

    for epoch in range(hparams.n_epochs):
        image_encoder.train()
        image_classifier.train()
        genetics_encoder.train()
        projector.train()

        train_metrics = init_alignment_metrics()
        train_img_preds_list = []
        train_img_labels_list = []
        train_gen_preds_list = []
        train_gen_labels_list = []
        steps = max(len(train_img_loader), len(train_gen_loader))
        img_iter = iter(train_img_loader)
        gen_iter = iter(train_gen_loader)

        for step in range(steps):
            img_batch, img_iter = get_next_batch(img_iter, train_img_loader)
            gen_batch, gen_iter = get_next_batch(gen_iter, train_gen_loader)

            losses = compute_alignment_losses(
                img_batch,
                gen_batch,
                image_encoder,
                image_classifier,
                genetics_encoder,
                projector,
                criterion_img,
                criterion_gen,
                hparams,
                device,
            )

            if imaging_optimizer is not None:
                imaging_optimizer.zero_grad()
            genetics_optimizer.zero_grad()
            losses["total"].backward()
            if imaging_optimizer is not None:
                imaging_optimizer.step()
            genetics_optimizer.step()
            if imaging_scheduler is not None:
                imaging_scheduler.step()

            loss_values = {k: losses[k].item() for k in losses}
            accumulate_metrics(train_metrics, loss_values)

            # Collect predictions for train accuracy/AUC
            with torch.no_grad():
                img_feat = img_batch["img_feat"].to(device).float()
                img_labels = img_batch["label"].to(device).view(-1)
                _, img_logits = forward_imaging_encoder(
                    image_encoder, img_feat, 0.0, training=False
                )
                img_logits = img_logits.squeeze(-1)
                train_img_preds_list.append(torch.sigmoid(img_logits).cpu())
                train_img_labels_list.append(img_labels.cpu())

                pathways = gen_batch["pathway"].to(device).float()
                gen_labels = gen_batch["label"].to(device).view(-1)
                _, gen_logits = forward_genetics_encoder(
                    genetics_encoder, pathways, return_reconstruction=False
                )
                gen_logits = gen_logits.squeeze(-1)
                train_gen_preds_list.append(torch.sigmoid(gen_logits).cpu())
                train_gen_labels_list.append(gen_labels.cpu())

            global_step += 1

        train_metrics = finalize_metrics(train_metrics)

        train_img_preds = torch.cat(train_img_preds_list).numpy()
        train_img_labels = torch.cat(train_img_labels_list).numpy()
        train_gen_preds = torch.cat(train_gen_preds_list).numpy()
        train_gen_labels = torch.cat(train_gen_labels_list).numpy()

        try:
            train_metrics["img_auc"] = roc_auc_score(train_img_labels, train_img_preds)
            train_metrics["img_acc"] = accuracy_score(
                train_img_labels, (train_img_preds > 0.5).astype(int)
            )
        except:
            train_metrics["img_auc"] = 0.0
            train_metrics["img_acc"] = 0.0

        try:
            train_metrics["gen_auc"] = roc_auc_score(train_gen_labels, train_gen_preds)
            train_metrics["gen_acc"] = accuracy_score(
                train_gen_labels, (train_gen_preds > 0.5).astype(int)
            )
        except:
            train_metrics["gen_auc"] = 0.0
            train_metrics["gen_acc"] = 0.0

        val_metrics = evaluate_alignment(
            image_encoder,
            image_classifier,
            genetics_encoder,
            projector,
            val_img_loader,
            val_gen_loader,
            criterion_img,
            criterion_gen,
            hparams,
            device,
            collect_ids=False,
        )
        test_metrics = evaluate_alignment(
            image_encoder,
            image_classifier,
            genetics_encoder,
            projector,
            test_img_loader,
            test_gen_loader,
            criterion_img,
            criterion_gen,
            hparams,
            device,
            collect_ids=True,
        )
        # Cross-modality stats are now computed inside evaluate_alignment()

        print(
            f"Epoch {epoch+1}/{hparams.n_epochs} "
            f"- Train Total: {train_metrics['total']:.4f}, "
            f"Val Total: {val_metrics['total']:.4f}, "
            f"Test Total: {test_metrics['total']:.4f}",
            flush=True,
        )
        print(
            f"  Loss -> img_cls: {train_metrics['img_cls']:.4f}, "
            f"gen_cls: {train_metrics['gen_cls']:.4f}, coral: {train_metrics['coral']:.4f} "
            f"(ASD: {train_metrics['coral_asd']:.4f}, CTL: {train_metrics['coral_control']:.4f}), "
            f"contrast: {train_metrics['contrast']:.4f}, orth: {train_metrics['orth']:.4f}, "
            f"gen_recon: {train_metrics['gen_recon']:.4f} | "
            f"Val gen_recon: {val_metrics['gen_recon']:.4f} | "
            f"Test gen_recon: {test_metrics['gen_recon']:.4f}",
            flush=True,
        )
        print(
            f"  ACC/AUC -> Train: Img {train_metrics.get('img_acc', 0.0):.3f}/{train_metrics.get('img_auc', 0.0):.3f} "
            f"Gen {train_metrics.get('gen_acc', 0.0):.3f}/{train_metrics.get('gen_auc', 0.0):.3f} | "
            f"Val: Img {val_metrics.get('img_acc', 0.0):.3f}/{val_metrics.get('img_auc', 0.0):.3f} "
            f"Gen {val_metrics.get('gen_acc', 0.0):.3f}/{val_metrics.get('gen_auc', 0.0):.3f} | "
            f"Test: Img {test_metrics.get('img_acc', 0.0):.3f}/{test_metrics.get('img_auc', 0.0):.3f} "
            f"Gen {test_metrics.get('gen_acc', 0.0):.3f}/{test_metrics.get('gen_auc', 0.0):.3f}",
            flush=True,
        )
        if "img_correct" in test_metrics:
            print(
                f"  Test correct counts -> Img: {test_metrics.get('img_correct', 0)}/{test_metrics.get('img_total', 0)}, "
                f"Gen: {test_metrics.get('gen_correct', 0)}/{test_metrics.get('gen_total', 0)}, "
                f"Both (by subject ID): {test_metrics.get('both_correct_subjects', 0)}, "
                f"Img-only: {test_metrics.get('img_only_correct_subjects', 0)}, "
                f"Gen-only: {test_metrics.get('gen_only_correct_subjects', 0)}",
                flush=True,
            )

        # UMAP visualization every 10 epochs (if available)
        if umap is not None and writer is not None and (epoch + 1) % 10 == 0:
            loaders_tuple_img = (train_img_loader, val_img_loader, test_img_loader)
            loaders_tuple_gen = (train_gen_loader, val_gen_loader, test_gen_loader)
            latent_pack = collect_latent_embeddings(
                image_encoder,
                image_classifier,
                genetics_encoder,
                projector,
                loaders_tuple_img,
                loaders_tuple_gen,
                device,
            )
            if latent_pack is not None:
                visualize_umap(
                    *latent_pack,
                    epoch=epoch,
                    experiment_name=hparams.experiment_name,
                    writer=writer,
                )
            else:
                print("Skipping UMAP: no embeddings collected.")

        # Clean up memory after each epoch to prevent accumulation
        torch.cuda.empty_cache()
        gc.collect()

        if writer is not None:
            for split_name, metrics_dict in [
                ("train_epoch", train_metrics),
                ("val", val_metrics),
                ("test", test_metrics),
            ]:
                writer.add_scalar(
                    f"{split_name}/loss_total", metrics_dict["total"], epoch
                )
                writer.add_scalar(
                    f"{split_name}/loss_imaging_cls", metrics_dict["img_cls"], epoch
                )
                writer.add_scalar(
                    f"{split_name}/loss_genetics_cls", metrics_dict["gen_cls"], epoch
                )
                writer.add_scalar(
                    f"{split_name}/loss_coral", metrics_dict["coral"], epoch
                )
                if "mmd" in metrics_dict:
                    writer.add_scalar(
                        f"{split_name}/loss_mmd", metrics_dict["mmd"], epoch
                    )
                else:
                    writer.add_scalar(
                        f"{split_name}/loss_coral_asd", metrics_dict["coral_asd"], epoch
                    )
                    writer.add_scalar(
                        f"{split_name}/loss_coral_control",
                        metrics_dict["coral_control"],
                        epoch,
                    )
                writer.add_scalar(
                    f"{split_name}/loss_contrastive", metrics_dict["contrast"], epoch
                )
                writer.add_scalar(
                    f"{split_name}/loss_orthogonality", metrics_dict["orth"], epoch
                )
                writer.add_scalar(
                    f"{split_name}/correct_img",
                    metrics_dict.get("img_correct", 0),
                    epoch,
                )
                writer.add_scalar(
                    f"{split_name}/correct_gen",
                    metrics_dict.get("gen_correct", 0),
                    epoch,
                )
                writer.add_scalar(
                    f"{split_name}/total_img", metrics_dict.get("img_total", 0), epoch
                )
                writer.add_scalar(
                    f"{split_name}/total_gen", metrics_dict.get("gen_total", 0), epoch
                )
                writer.add_scalar(
                    f"{split_name}/loss_gen_recon", metrics_dict["gen_recon"], epoch
                )
                # Log cross-modality correctness metrics (only for test split with collect_ids=True)
                if "both_correct_subjects" in metrics_dict:
                    writer.add_scalar(
                        f"{split_name}/both_correct_subjects",
                        metrics_dict["both_correct_subjects"],
                        epoch,
                    )
                    writer.add_scalar(
                        f"{split_name}/img_only_correct_subjects",
                        metrics_dict["img_only_correct_subjects"],
                        epoch,
                    )
                    writer.add_scalar(
                        f"{split_name}/gen_only_correct_subjects",
                        metrics_dict["gen_only_correct_subjects"],
                        epoch,
                    )

                # Add accuracy and AUC metrics (only available for val/test)
                if "img_auc" in metrics_dict:
                    writer.add_scalar(
                        f"{split_name}/imaging_auc", metrics_dict["img_auc"], epoch
                    )
                    writer.add_scalar(
                        f"{split_name}/imaging_acc", metrics_dict["img_acc"], epoch
                    )
                    writer.add_scalar(
                        f"{split_name}/genetics_auc", metrics_dict["gen_auc"], epoch
                    )
                    writer.add_scalar(
                        f"{split_name}/genetics_acc", metrics_dict["gen_acc"], epoch
                    )

            # Log association matrix T = W_I^T @ W_G (n_rois × n_pathways) every 10 epochs
            if (epoch + 1) % 10 == 0:
                with torch.no_grad():
                    base_projector = unwrap_data_parallel(projector)
                    association_matrix = (
                        torch.matmul(
                            base_projector.W_I.T,
                            base_projector.W_G,  # [n_rois, n_pathways]
                        )
                        .cpu()
                        .numpy()
                    )

                    # Create a heatmap figure
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(association_matrix, ax=ax, cmap="viridis")
                    ax.set_xlabel("Pathways (Genetics)")
                    ax.set_ylabel("ROIs (Imaging)")
                    ax.set_title("ROI-Pathway Association Matrix (T)")
                    fig.tight_layout()

                    # Convert plot to image tensor for TensorBoard
                    fig.canvas.draw()
                    img_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                    img_data = img_data.reshape(
                        fig.canvas.get_width_height()[::-1] + (3,)
                    )
                    plt.close(fig)

                    # Add the heatmap image to TensorBoard
                    writer.add_image(
                        "alignment/association_matrix",
                        img_data,
                        epoch,
                        dataformats="HWC",
                    )

    print("\nAlignment training finished.")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(42)
    parser = ArgumentParser(description="Trainer args", add_help=False)
    add_argument(parser)
    hparams = parser.parse_args()
    writer = (
        None
        if hparams.not_write_tensorboard
        else SummaryWriter(log_dir=TENSORBOARD_CROSS_MODALITY / hparams.experiment_name)
    )
    main(hparams, writer)
