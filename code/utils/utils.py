import os
import re
import sys
import random
import pickle
from argparse import Namespace
from typing import List, Optional, Tuple
from collections import defaultdict

import shutil
import numpy as np
import pandas as pd
import torch
from sklearn import metrics
import matplotlib.pyplot as plt
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import nibabel as nib
from matplotlib.patches import Rectangle

from utils.const import (
    RESULT_FOLDER,
    ABIDE_DATA_FOLDER_I,
    ABIDE_DATA_FOLDER_I_FREESURFER_RECON,
    ABIDE_DATA_FOLDER_II,
    ACE_FILE_with_relatedness,
    # ABIDE_I_transform,
    # ABIDE_II_transform,
    ABIDE_I_MNI,
    ABIDE_II_MNI,
    ABIDE_I_PHENOTYPE,
    ABIDE_II_PHENOTYPE,
    ABIDE_II_PHENOTYPE_Long,
    ACE_PHENOTYPE,
    ABIDE_I_IMAGE_FEATURES,
    ABIDE_II_IMAGE_FEATURES,
    ACE_IMG_FEAT,
    ACE_MRI_FOLDER,
)
from torch.utils.data import Dataset, DataLoader


class PathwayDataset(Dataset):
    def __init__(
        self,
        pathway: pd.DataFrame,
        label: pd.DataFrame,
        hparams: Namespace,
        ids: pd.DataFrame = None,
        extra_pathway: pd.DataFrame = None,
    ):
        self.pathway = pathway
        self.label = label.to_numpy()
        self.ids = ids
        self.datast = hparams.dataset
        self.extra_pathway = extra_pathway
        self.n_channels = max(1, int(getattr(hparams, "genetics_input_channels", 1)))

        if self.extra_pathway is not None:
            if len(self.extra_pathway) != len(self.pathway):
                raise ValueError(
                    "extra_pathway and pathway must have the same number of rows"
                )
            # Ensure we have at least two channels when a second pathway view is provided
            self.n_channels = max(self.n_channels, 2)

    def __len__(self):
        return len(self.pathway)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        primary = self.pathway.iloc[idx, :]
        label = np.expand_dims(self.label[idx], axis=0)

        # Build pathway tensor with optional multi-channel support
        pathway_tensor = torch.tensor(primary.to_numpy(), dtype=torch.float32)
        if self.extra_pathway is not None:
            secondary = torch.tensor(
                self.extra_pathway.iloc[idx, :].to_numpy(), dtype=torch.float32
            )
            pathway_tensor = torch.stack([pathway_tensor, secondary], dim=-1)
        else:
            pathway_tensor = pathway_tensor.unsqueeze(-1)
            if self.n_channels > 1:
                pathway_tensor = pathway_tensor.repeat(1, self.n_channels)

        if self.ids is not None:
            if isinstance(self.ids, pd.Series):
                ids = self.ids.iloc[idx]
            else:
                ids = self.ids.iloc[idx, :]
            # Some splits have missing IDs; fall back to a stable placeholder so
            # the DataLoader collate function never sees None. `pd.isna` may
            # return arrays/Series, so reduce it to a single bool.
            missing = False
            if ids is None:
                missing = True
            else:
                try:
                    missing = pd.isna(ids)
                    if isinstance(missing, (np.ndarray, pd.Series, list)):
                        missing = bool(np.all(missing))
                except Exception:
                    missing = False
            if missing:
                ids = f"{self.datast}_sample_{idx}"
            ids = str(ids)
        else:
            ids = f"{self.datast}_sample_{idx}"
        sample = {
            "pathway": pathway_tensor,
            "label": torch.tensor(label).float(),
        }
        if ids is not None:
            sample["ids"] = ids

        return sample


def create_training_samples() -> List[dict]:
    samples_ABIDE_I, n_subjects_lower_21_I = get_ABIDE_I_subject()
    samples_ABIDE_II, n_subjects_lower_21_II = get_ABIDE_II_subject()

    print(f"Number of subjects in ABIDE I: {len(samples_ABIDE_I)}")
    print(f"Number of subjects in ABIDE II: {len(samples_ABIDE_II)}")
    print(f"Number of subjects with age <= 21 in ABIDE I: {n_subjects_lower_21_I}")
    print(f"Number of subjects with age <= 21 in ABIDE II: {n_subjects_lower_21_II}")

    # merge the two dicts
    merged_samples = samples_ABIDE_I + samples_ABIDE_II
    sorted_samples = sorted(merged_samples, key=lambda x: x["age"])

    with open("samples_ABIDE_merged.pkl", "wb") as f:
        pickle.dump(merged_samples, f)

    with open("samples_ABIDE_merged.pkl", "rb") as f:
        samples = pickle.load(f)

    return samples


def get_ABIDE_II_transformed_subject() -> List[dict]:
    ABIDE_II = list(ABIDE_II_transform.glob("**/transformed_*.nii.gz"))
    ABIDE_II_phenotype_file = pd.read_csv(ABIDE_II_PHENOTYPE, encoding="cp1252")
    ABIDE_II_phenotype_file_Longi = pd.read_csv(ABIDE_II_PHENOTYPE_Long)

    longitudinal_subjects = set(
        # turn it to string
        ABIDE_II_phenotype_file_Longi[
            ABIDE_II_phenotype_file_Longi["SESSION"] == "Baseline"
        ]["SUB_ID"].values.astype(str)
    )

    sample_dicts = []
    for path in ABIDE_II:
        m = re.search(r"-(\d{5})(?=/)", str(path))
        if m:
            subject_id = m.group(1)

        if subject_id.startswith("5"):
            continue
        else:
            row = ABIDE_II_phenotype_file[
                ABIDE_II_phenotype_file["SUB_ID"] == int(subject_id)
            ]

        if row.empty:
            print(
                f"Warning: No phenotype data found for subject {subject_id}. Skipping."
            )
            continue

        img_path = f"/projectnb/ace-ig/ABIDE/ABIDE_II_BIDS/derivatives/MNI/sub-{subject_id}/anat/sub-{subject_id}_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz"
        mask_path = str(img_path).replace("preproc_T1w", "brain_mask")
        age = row["AGE_AT_SCAN "].values[0] if "AGE_AT_SCAN " in row else None
        transformed_img_path = str(path)
        digits = re.search(r"transformed_(\d+)", str(path))
        jacobian_path = str(path).replace("transformed_", "sim_BN_targetJac_")
        dispfield_path = str(path).replace("transformed_", "sim_BN_dispfield_")

        if subject_id in longitudinal_subjects:
            print(f"subject {subject_id} is in the skip list, skipping.")
            continue

        if age is None:
            continue

        sample_dicts.append(
            {
                "img": str(img_path),
                "mask": mask_path,
                "transformed_img": transformed_img_path,
                "jacobian": jacobian_path,
                "dispfield": dispfield_path,
                "label": int(row["DX_GROUP"].values[0]) - 1,
                "subject_id": subject_id,
                "site_id": row["SITE_ID"].values[0],
                "age": age,
                "dataset": "II",
                "digits": digits.group(1),
            }
        )

    return sample_dicts


def get_ABIDE_I_transformed_subject() -> List[dict]:
    ABIDE_I = list(ABIDE_I_transform.glob("**/transformed_*.nii.gz"))
    ABIDE_I_phenotype_file = pd.read_csv(ABIDE_I_PHENOTYPE)
    ABIDE_II_phenotype_file = pd.read_csv(ABIDE_II_PHENOTYPE_Long)

    longitudinal_subjects = set(
        # turn it to string
        ABIDE_II_phenotype_file[ABIDE_II_phenotype_file["SESSION"] == "Baseline"][
            "SUB_ID"
        ].values.astype(str)
    )

    sample_dicts = []
    for path in ABIDE_I:
        m = re.search(r"-(\d{5})(?=/)", str(path))
        if m:
            subject_id = m.group(1)

        row = ABIDE_I_phenotype_file[
            ABIDE_I_phenotype_file["SUB_ID"] == int(subject_id)
        ]
        if row.empty:
            print(
                f"Warning: No phenotype data found for subject {subject_id}. Skipping."
            )
            continue

        img_path = f"/projectnb/ace-ig/ABIDE/ABIDE_I_BIDS/derivatives/MNI/sub-{subject_id}/anat/sub-{subject_id}_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz"
        mask_path = str(img_path).replace("preproc_T1w", "brain_mask")
        age = row["AGE_AT_SCAN"].values[0] if "AGE_AT_SCAN" in row else None
        transformed_img_path = str(path)
        digits = re.search(r"transformed_(\d+)", str(path))
        jacobian_path = str(path).replace("transformed_", "sim_BN_targetJac_")
        dispfield_path = str(path).replace("transformed_", "sim_BN_dispfield_")

        if subject_id in longitudinal_subjects:
            print(f"subject {subject_id} is in the skip list, skipping.")
            continue

        if age is None:
            continue

        sample_dicts.append(
            {
                "img": str(img_path),
                "mask": mask_path,
                "transformed_img": transformed_img_path,
                "jacobian": jacobian_path,
                "dispfield": dispfield_path,
                "label": int(row["DX_GROUP"].values[0]) - 1,
                "subject_id": subject_id,
                "site_id": row["SITE_ID"].values[0],
                "age": age,
                "dataset": "I",
                "digits": digits.group(1),
            }
        )

    return sample_dicts


def visualize_3d_mri(
    data: np.ndarray,
    writer: Optional[SummaryWriter] = None,
    title: str = "3D_MRI",
    epoch: int = 0,
) -> None:
    """
    Visualize 3D MRI data by creating a grid of 2D slices from different planes.
    """
    _, axial_slices, coronal_slices, sagittal_slices = data.shape
    # 2D from different planes
    # turn left for 90 degrees
    axial = np.rot90(data[0, axial_slices // 2, :, :], k=1)
    coronal = np.rot90(data[0, :, coronal_slices // 2, :], k=1)
    sagittal = np.rot90(data[0, :, :, sagittal_slices // 2], k=1)
    img = np.concatenate([axial, coronal, sagittal], axis=1)
    img = np.expand_dims(img, axis=0)  # Add channel dimension for tensorboard

    if writer:
        writer.add_image(title, img, epoch)


def split_mri_into_patches(input_img: torch.Tensor, patch_size: int = 96):
    """
    Split 3D MRI into 8 non-overlapping patches (2x2x2 grid)

    Args:
        input_img: Input MRI tensor [1, D, H, W] where D,H,W >= patch_size
        patch_size: Size of each patch dimension (default: 96)

    Returns:
        patches: List of 8 patches, each [1, patch_size, patch_size, patch_size]
    """
    patches = [
        input_img[:, :patch_size, :patch_size, :patch_size],  # patch_0: front-top-left
        input_img[:, patch_size:, :patch_size, :patch_size],  # patch_1: back-top-left
        input_img[
            :, :patch_size, patch_size:, :patch_size
        ],  # patch_2: front-bottom-left
        input_img[
            :, patch_size:, patch_size:, :patch_size
        ],  # patch_3: back-bottom-left
        input_img[:, :patch_size, :patch_size, patch_size:],  # patch_4: front-top-right
        input_img[:, patch_size:, :patch_size, patch_size:],  # patch_5: back-top-right
        input_img[
            :, :patch_size, patch_size:, patch_size:
        ],  # patch_6: front-bottom-right
        input_img[
            :, patch_size:, patch_size:, patch_size:
        ],  # patch_7: back-bottom-right
    ]
    return patches


def reconstruct_features_from_patches(patch_features: torch.Tensor):
    """
    Reconstruct feature map from 8 patches back to original spatial arrangement

    Args:
        patch_features: [8, channels, 3, 3, 3] - features from 8 patches

    Returns:
        reconstructed: [channels, 6, 6, 6] - spatially arranged features
    """
    batch_size, channels, patch_d, patch_h, patch_w = patch_features.shape

    # Initialize output feature map: 2x2x2 patches -> 6x6x6 feature map
    reconstructed = torch.zeros(channels, 6, 6, 6, device=patch_features.device)

    # Define spatial positions for each patch in the 2x2x2 grid
    # Each patch maps to a 3x3x3 region in the final 6x6x6 volume
    # Match the splitting logic: [depth, height, width] = [D, H, W]
    patch_positions = [
        (0, 0, 0),  # patch_0: [:96, :96, :96] -> front-top-left -> [0:3, 0:3, 0:3]
        (1, 0, 0),  # patch_1: [96:, :96, :96] -> back-top-left -> [3:6, 0:3, 0:3]
        (0, 1, 0),  # patch_2: [:96, 96:, :96] -> front-bottom-left -> [0:3, 3:6, 0:3]
        (1, 1, 0),  # patch_3: [96:, 96:, :96] -> back-bottom-left -> [3:6, 3:6, 0:3]
        (0, 0, 1),  # patch_4: [:96, :96, 96:] -> front-top-right -> [0:3, 0:3, 3:6]
        (1, 0, 1),  # patch_5: [96:, :96, 96:] -> back-top-right -> [3:6, 0:3, 3:6]
        (0, 1, 1),  # patch_6: [:96, 96:, 96:] -> front-bottom-right -> [0:3, 3:6, 3:6]
        (1, 1, 1),  # patch_7: [96:, 96:, 96:] -> back-bottom-right -> [3:6, 3:6, 3:6]
    ]

    for patch_idx, (z_pos, y_pos, x_pos) in enumerate(patch_positions):
        # Calculate spatial coordinates in the 6x6x6 output
        z_start, z_end = z_pos * 3, (z_pos + 1) * 3
        y_start, y_end = y_pos * 3, (y_pos + 1) * 3
        x_start, x_end = x_pos * 3, (x_pos + 1) * 3

        # Place patch features in their corresponding spatial location
        reconstructed[:, z_start:z_end, y_start:y_end, x_start:x_end] = patch_features[
            patch_idx
        ]

    return reconstructed


def visualize_patch_structure(save_path: str = None):
    """
    Create a visualization showing how 3D MRI is split into 8 patches and reconstructed
    This function actually tests the patch functions with real data
    """
    print("Testing patch processing functions...")

    # Create test MRI data with identifiable patterns
    test_mri = torch.zeros(1, 192, 192, 192)
    patch_values = [i + 1 for i in range(8)]  # Values 1-8 for tracking

    # Fill each patch region with unique values
    test_mri[:, :96, :96, :96] = patch_values[0]  # patch_0
    test_mri[:, 96:, :96, :96] = patch_values[1]  # patch_1
    test_mri[:, :96, 96:, :96] = patch_values[2]  # patch_2
    test_mri[:, 96:, 96:, :96] = patch_values[3]  # patch_3
    test_mri[:, :96, :96, 96:] = patch_values[4]  # patch_4
    test_mri[:, 96:, :96, 96:] = patch_values[5]  # patch_5
    test_mri[:, :96, 96:, 96:] = patch_values[6]  # patch_6
    test_mri[:, 96:, 96:, 96:] = patch_values[7]  # patch_7

    # Test actual patch splitting
    patches = split_mri_into_patches(test_mri, patch_size=96)
    print(f"✓ Split into {len(patches)} patches, each with shape {patches[0].shape}")

    # Verify patch values
    for i, patch in enumerate(patches):
        actual_value = torch.unique(patch).item()
        expected_value = patch_values[i]
        if actual_value == expected_value:
            print(f"✓ Patch {i}: Correct value {actual_value}")
        else:
            print(f"✗ Patch {i}: Expected {expected_value}, got {actual_value}")

    # Create simulated encoded features (256 channels, 3x3x3 each)
    simulated_patch_features = []
    for i, patch in enumerate(patches):
        # Simulate encoder output: fill with patch index for tracking
        feature = torch.full((256, 3, 3, 3), patch_values[i], dtype=torch.float32)
        simulated_patch_features.append(feature)

    # Stack to [8, 256, 3, 3, 3]
    patch_features = torch.stack(simulated_patch_features, dim=0)
    print(f"✓ Created simulated patch features: {patch_features.shape}")

    # Test reconstruction
    reconstructed = reconstruct_features_from_patches(patch_features)
    print(f"✓ Reconstructed features: {reconstructed.shape}")

    # Verify reconstruction correctness
    reconstruction_correct = True
    expected_positions = [
        (slice(0, 3), slice(0, 3), slice(0, 3)),  # patch_0 -> [0:3, 0:3, 0:3]
        (slice(3, 6), slice(0, 3), slice(0, 3)),  # patch_1 -> [3:6, 0:3, 0:3]
        (slice(0, 3), slice(3, 6), slice(0, 3)),  # patch_2 -> [0:3, 3:6, 0:3]
        (slice(3, 6), slice(3, 6), slice(0, 3)),  # patch_3 -> [3:6, 3:6, 0:3]
        (slice(0, 3), slice(0, 3), slice(3, 6)),  # patch_4 -> [0:3, 0:3, 3:6]
        (slice(3, 6), slice(0, 3), slice(3, 6)),  # patch_5 -> [3:6, 0:3, 3:6]
        (slice(0, 3), slice(3, 6), slice(3, 6)),  # patch_6 -> [0:3, 3:6, 3:6]
        (slice(3, 6), slice(3, 6), slice(3, 6)),  # patch_7 -> [3:6, 3:6, 3:6]
    ]

    for patch_idx, (d_slice, h_slice, w_slice) in enumerate(expected_positions):
        region = reconstructed[:, d_slice, h_slice, w_slice]
        expected_value = patch_values[patch_idx]
        actual_value = torch.unique(region).item()

        if actual_value == expected_value:
            print(f"✓ Region {patch_idx}: Correct mapping to value {actual_value}")
        else:
            print(
                f"✗ Region {patch_idx}: Expected {expected_value}, got {actual_value}"
            )
            reconstruction_correct = False

    if reconstruction_correct:
        print("✓ RECONSTRUCTION TEST PASSED: All spatial mappings correct!")
    else:
        print("✗ RECONSTRUCTION TEST FAILED: Spatial mapping errors detected!")
        return

    fig = plt.figure(figsize=(16, 10))

    # Create 3D plots

    # Plot 1: Original 3D volume structure
    ax1 = fig.add_subplot(2, 3, 1, projection="3d")

    # Draw original volume as wireframe cube
    # Define cube vertices
    r = [0, 192]
    X, Y = np.meshgrid(r, r)

    # Draw 6 faces of the cube
    ax1.plot_surface(X, Y, np.zeros_like(X), alpha=0.1, color="blue")  # bottom
    ax1.plot_surface(X, Y, X * 0 + 192, alpha=0.1, color="blue")  # top
    ax1.plot_surface(X, np.zeros_like(X), Y, alpha=0.1, color="blue")  # front
    ax1.plot_surface(X, X * 0 + 192, Y, alpha=0.1, color="blue")  # back
    ax1.plot_surface(np.zeros_like(X), X, Y, alpha=0.1, color="blue")  # left
    ax1.plot_surface(X * 0 + 192, X, Y, alpha=0.1, color="blue")  # right

    # Add division lines at patch_size=96
    for coord in [96]:
        # X-direction planes (perpendicular to X-axis)
        y_range, z_range = np.meshgrid([0, 192], [0, 192])
        x_const = np.full_like(y_range, coord)
        ax1.plot_surface(x_const, y_range, z_range, alpha=0.3, color="red")

        # Y-direction planes (perpendicular to Y-axis)
        x_range, z_range = np.meshgrid([0, 192], [0, 192])
        y_const = np.full_like(x_range, coord)
        ax1.plot_surface(x_range, y_const, z_range, alpha=0.3, color="red")

        # Z-direction planes (perpendicular to Z-axis)
        x_range, y_range = np.meshgrid([0, 192], [0, 192])
        z_const = np.full_like(x_range, coord)
        ax1.plot_surface(x_range, y_range, z_const, alpha=0.3, color="red")

    ax1.set_title("Original MRI Volume\n(192X192X192)")
    ax1.set_xlabel("X (Right-Left)")
    ax1.set_ylabel("Y (Anterior-Posterior)")
    ax1.set_zlabel("Z (Superior-Inferior)")

    # Plot 2: 8 patches layout
    ax2 = fig.add_subplot(2, 3, 2, projection="3d")

    # Define patch colors
    colors = ["red", "green", "blue", "orange", "purple", "brown", "pink", "gray"]
    patch_names = [
        "front-top-left",
        "back-top-left",
        "front-bottom-left",
        "back-bottom-left",
        "front-top-right",
        "back-top-right",
        "front-bottom-right",
        "back-bottom-right",
    ]

    # Draw each patch as a colored cube
    patch_coords = [
        (0, 0, 0),
        (96, 0, 0),
        (0, 96, 0),
        (96, 96, 0),
        (0, 0, 96),
        (96, 0, 96),
        (0, 96, 96),
        (96, 96, 96),
    ]

    for i, ((x, y, z), color, name) in enumerate(
        zip(patch_coords, colors, patch_names)
    ):
        # Draw cube wireframe
        vertices = [
            [x, x + 96, x + 96, x, x],  # x coordinates
            [y, y, y + 96, y + 96, y],  # y coordinates
            [z, z, z, z, z],  # z coordinates (bottom face)
        ]
        ax2.plot(
            vertices[0],
            vertices[1],
            vertices[2],
            color=color,
            linewidth=2,
            label=f"Patch {i}",
        )

        # Top face
        vertices_top = [vertices[0], vertices[1], [z + 96] * 5]
        ax2.plot(
            vertices_top[0], vertices_top[1], vertices_top[2], color=color, linewidth=2
        )

        # Vertical edges
        for j in range(4):
            ax2.plot(
                [vertices[0][j], vertices[0][j]],
                [vertices[1][j], vertices[1][j]],
                [z, z + 96],
                color=color,
                linewidth=2,
            )

    ax2.set_title("8 Patches (96×96×96 each)")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")

    # Plot 3: Feature reconstruction (6×6×6)
    ax3 = fig.add_subplot(2, 3, 3, projection="3d")

    # Draw 6×6×6 feature grid
    for i in range(2):  # depth (D) dimension
        for j in range(2):  # height (H) dimension
            for k in range(2):  # width (W) dimension
                # Each 3×3×3 block from one patch
                # Map to actual patch indices based on our splitting logic
                d_start, h_start, w_start = i * 3, j * 3, k * 3

                # Correct patch indexing to match split_mri_into_patches:
                # patch_0: [:96, :96, :96] -> (i=0, j=0, k=0)
                # patch_1: [96:, :96, :96] -> (i=1, j=0, k=0)
                # patch_2: [:96, 96:, :96] -> (i=0, j=1, k=0)
                # patch_3: [96:, 96:, :96] -> (i=1, j=1, k=0)
                # patch_4: [:96, :96, 96:] -> (i=0, j=0, k=1)
                # patch_5: [96:, :96, 96:] -> (i=1, j=0, k=1)
                # patch_6: [:96, 96:, 96:] -> (i=0, j=1, k=1)
                # patch_7: [96:, 96:, 96:] -> (i=1, j=1, k=1)
                patch_idx = i + j * 2 + k * 4
                color = colors[patch_idx]

                # Draw 3×3×3 block (use d,h,w coordinates)
                d_coords = [d_start, d_start + 3, d_start + 3, d_start, d_start]
                h_coords = [h_start, h_start, h_start + 3, h_start + 3, h_start]
                w_coords = [w_start] * 5

                ax3.plot(d_coords, h_coords, w_coords, color=color, linewidth=2)
                ax3.plot(
                    d_coords, h_coords, [w_start + 3] * 5, color=color, linewidth=2
                )

                # Vertical edges
                for edge_idx in range(4):
                    ax3.plot(
                        [d_coords[edge_idx], d_coords[edge_idx]],
                        [h_coords[edge_idx], h_coords[edge_idx]],
                        [w_start, w_start + 3],
                        color=color,
                        linewidth=2,
                    )

    ax3.set_title("Reconstructed Features\n(6×6×6 spatial arrangement)")
    ax3.set_xlabel("X (3×3×3 blocks)")
    ax3.set_ylabel("Y")
    ax3.set_zlabel("Z")

    # Plot 4: 2D slice view showing patch arrangement
    ax4 = fig.add_subplot(2, 3, 4)

    # Create 2D grid showing patch indices
    patch_grid = np.array(
        [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]  # Front layer (z=0)  # Back layer (z=1)
    )

    # Show front layer
    for i in range(2):
        for j in range(2):
            patch_idx = patch_grid[0, i, j]
            color = colors[patch_idx]
            rect = plt.Rectangle(
                (j, 1 - i), 1, 1, facecolor=color, alpha=0.7, edgecolor="black"
            )
            ax4.add_patch(rect)
            ax4.text(
                j + 0.5,
                1 - i + 0.5,
                f"P{patch_idx}",
                ha="center",
                va="center",
                fontweight="bold",
            )

    ax4.set_xlim(0, 2)
    ax4.set_ylim(0, 2)
    ax4.set_title("Front Layer (Z=0)\nPatch Arrangement")
    ax4.set_xlabel("X direction")
    ax4.set_ylabel("Y direction")
    ax4.set_xticks([0.5, 1.5])
    ax4.set_xticklabels(["Left", "Right"])
    ax4.set_yticks([0.5, 1.5])
    ax4.set_yticklabels(["Bottom", "Top"])
    ax4.grid(True)

    # Plot 5: Back layer
    ax5 = fig.add_subplot(2, 3, 5)

    # Show back layer
    for i in range(2):
        for j in range(2):
            patch_idx = patch_grid[1, i, j]
            color = colors[patch_idx]
            rect = plt.Rectangle(
                (j, 1 - i), 1, 1, facecolor=color, alpha=0.7, edgecolor="black"
            )
            ax5.add_patch(rect)
            ax5.text(
                j + 0.5,
                1 - i + 0.5,
                f"P{patch_idx}",
                ha="center",
                va="center",
                fontweight="bold",
            )

    ax5.set_xlim(0, 2)
    ax5.set_ylim(0, 2)
    ax5.set_title("Back Layer (Z=1)\nPatch Arrangement")
    ax5.set_xlabel("X direction")
    ax5.set_ylabel("Y direction")
    ax5.set_xticks([0.5, 1.5])
    ax5.set_xticklabels(["Left", "Right"])
    ax5.set_yticks([0.5, 1.5])
    ax5.set_yticklabels(["Bottom", "Top"])
    ax5.grid(True)

    # Plot 6: Processing pipeline text
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis("off")

    pipeline_text = """
        PROCESSING PIPELINE:
        1. Input MRI: [1, 192, 192, 192]
        ↓
        2. Split into 8 patches: [1, 96, 96, 96] each
        ↓
        3. Encode each patch: [1, 96, 96, 96] → [256, 3, 3, 3]
        ↓
        4. Stack patches: [8, 256, 3, 3, 3]
        ↓
        5. Reconstruct spatial: [256, 6, 6, 6]

        SPATIAL MAPPING:
        • Each 96³ patch → 3³ feature region
        • 2×2×2 patches → 6×6×6 features
        • Maintains anatomical correspondence
        • 256 channels per spatial location
        """

    ax6.text(
        0.05,
        0.95,
        pipeline_text,
        transform=ax6.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Patch structure visualization saved to: {save_path}")

    plt.show()


def process_mri_patches_with_encoder(
    input_img: torch.Tensor, encoder, patch_size: int = 96
):
    """
    Process MRI through patch-based encoding and reconstruct features

    Args:
        input_img: Input MRI [1, D, H, W]
        encoder: Neural network encoder (e.g., EnhancedImageEncoder)
        patch_size: Size of each patch (default: 96)

    Returns:
        reconstructed_features: [channels, 6, 6, 6] - spatially coherent features
    """
    B, C, _, _, _ = input_img.shape
    ps = patch_size
    patches = (
        input_img.view(B, C, 2, ps, 2, ps, 2, ps)
        .permute(0, 2, 4, 6, 1, 3, 5, 7)
        .contiguous()
        .view(B * 8, C, ps, ps, ps)
    )

    patch_feats = encoder(patches)
    B8, C_out, s, _, _ = patch_feats.shape
    assert B8 == B * 8, "Unexpected encoder output batch size"

    # ---- Reconstruct back to the big volume (inverse of the split) ----
    # [B*8, C_out, s, s, s] -> [B, 2, 2, 2, C_out, s, s, s]
    # -> [B, C_out, 2, s, 2, s, 2, s] -> [B, C_out, 2*s, 2*s, 2*s]
    feats_3d = (
        patch_feats.view(B, 2, 2, 2, C_out, s, s, s)
        .permute(0, 4, 1, 5, 2, 6, 3, 7)
        .contiguous()
        .view(B, C_out, 2 * s, 2 * s, 2 * s)
    )

    return feats_3d


def get_ABIDE_I_subject() -> List[dict]:
    overlapping_site_with_ace = set(["YALE", "UCLA_1", "UCLA_2"])
    ABIDE_I = ABIDE_I_MNI.glob(
        "**/sub-*_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz"
    )
    ABIDE_I_phenotype_file = pd.read_csv(ABIDE_I_PHENOTYPE)
    ABIDE_II_phenotype_file = pd.read_csv(ABIDE_II_PHENOTYPE_Long)
    ABIDE_I_image_features_file = pd.read_csv(ABIDE_I_IMAGE_FEATURES)

    longitudinal_subjects = set(
        # turn it to string
        ABIDE_II_phenotype_file[ABIDE_II_phenotype_file["SESSION"] == "Baseline"][
            "SUB_ID"
        ].values.astype(str)
    )

    sample_dicts = []
    for path in ABIDE_I:
        m = re.search(r"-(\d{5})(?=_)", str(path))
        if m:
            subject_id = m.group(1)

        row = ABIDE_I_phenotype_file[
            ABIDE_I_phenotype_file["SUB_ID"] == int(subject_id)
        ]
        if row.empty:
            print(
                f"Warning: No phenotype data found for subject {subject_id}. Skipping."
            )
            continue

        mask_path = str(path).replace("preproc_T1w", "brain_mask")
        age = row["AGE_AT_SCAN"].values[0] if "AGE_AT_SCAN" in row else None

        features_row = ABIDE_I_image_features_file[
            ABIDE_I_image_features_file.iloc[:, 0] == f"sub-{subject_id}"
        ]

        if features_row.empty:
            print(
                f"Warning: No image features found for subject {subject_id}. Skipping."
            )
            continue

        if features_row.empty:
            print(
                f"Warning: No image features found for subject {subject_id}. Skipping."
            )
            continue

        if subject_id in longitudinal_subjects:
            print(f"subject {subject_id} is in the skip list, skipping.")
            continue

        if row["SITE_ID"].values[0] in overlapping_site_with_ace:
            continue

        # remove subjects with
        sample_dicts.append(
            {
                "img": str(path),
                "mask": mask_path,
                "label": int(row["DX_GROUP"].values[0]) - 1,
                "subject_id": subject_id,
                "site_id": row["SITE_ID"].values[0],
                "age": age,
                "dataset": "I",
                "image_features": features_row.values[0][1:],
            }
        )

    return sample_dicts


def get_ABIDE_II_subject_followup() -> List[dict]:
    ABIDE_II_Baseline = Path(
        "/projectnb/ace-ig/ABIDE/ABIDE_II_BIDS_Baseline/derivatives/MNI"
    ).resolve()
    ABIDE_II = ABIDE_II_Baseline.glob(
        "**/sub-*_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz"
    )
    ABIDE_II_long_phenotype_file = pd.read_csv(ABIDE_II_PHENOTYPE_Long)
    ABIDE_II_image_features_file = pd.read_csv(ABIDE_II_IMAGE_FEATURES)
    sample_dicts = []

    for path in ABIDE_II:
        baseline_mask_path = str(path).replace("preproc_T1w", "brain_mask")
        m = re.search(r"-(\d{5})(?=[/_])", str(path))
        if m:
            subject_id = m.group(1)
            followup_img_path = str(path).replace(
                "ABIDE_II_BIDS_Baseline", "ABIDE_II_BIDS"
            )
            followup_mask_path = str(baseline_mask_path).replace(
                "ABIDE_II_BIDS_Baseline", "ABIDE_II_BIDS"
            )
            row = ABIDE_II_long_phenotype_file[
                ABIDE_II_long_phenotype_file["SUB_ID"] == int(subject_id)
            ]

            baseline_row = row[row["SESSION"] == "Baseline"]
            followup_row = row[row["SESSION"] == "Followup_1"]
            sample_dicts.append(
                {
                    "baseline_img": str(path),
                    "baseline_mask": baseline_mask_path,
                    "followup_img": followup_img_path,
                    "followup_mask": followup_mask_path,
                    "baseline_age": baseline_row["AGE_AT_SCAN "].values[0],
                    "followup_age": followup_row["AGE_AT_SCAN "].values[0],
                    "label": int(row["DX_GROUP"].values[0]) - 1,
                    "subject_id": subject_id,
                    "site_id": row["SITE_ID"].values[0],
                }
            )

    return sample_dicts


def get_ABIDE_II_subject() -> Tuple[List[dict], int]:
    ABIDE_II = ABIDE_II_MNI.glob(
        "**/sub-*_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz"
    )

    ABIDE_II_phenotype_file = pd.read_csv(ABIDE_II_PHENOTYPE, encoding="cp1252")
    ABIDE_II_image_features_file = pd.read_csv(ABIDE_II_IMAGE_FEATURES)
    sample_dicts = []
    ignore_list = [
        "29728", "29729", "29730", "29731", "29732", "29734", "29735",
        "29736", "29737", "29738", "29739", "29740", "29741", "29742",
        "29744", "29746", "29747", "29748", "29749", "29750", "29751",
        "29752", "29753", "29755", "29756", "29757", "29758", "29759",
    ]  # ABIDEII-UCLA_1 site: T2 images mislabeled as T1w (1.5x1.5x4mm spacing)

    for path in ABIDE_II:
        mask_path = str(path).replace("preproc_T1w", "brain_mask")
        m = re.search(r"-(\d{5})(?=[/_])", str(path))
        if m:
            subject_id = m.group(1)

        if subject_id.startswith("5"):
            continue
        else:
            row = ABIDE_II_phenotype_file[
                ABIDE_II_phenotype_file["SUB_ID"] == int(subject_id)
            ]
            feature_row = ABIDE_II_image_features_file[
                ABIDE_II_image_features_file.iloc[:, 0] == f"sub-{subject_id}"
            ]

        if row.empty or subject_id in ignore_list or feature_row.empty:
            print(
                f"Warning: No phenotype or image features data found for subject {subject_id}. Skipping."
            )
            continue

        age = row["AGE_AT_SCAN "].values[0] if "AGE_AT_SCAN " in row else None

        if age is None:
            # print(f"Age data missing for subject {subject_id}. Skipping.")
            continue

        sample_dicts.append(
            {
                "img": str(path),
                "mask": mask_path,
                "label": int(row["DX_GROUP"].values[0]) - 1,
                "subject_id": subject_id,
                "site_id": row["SITE_ID"].values[0],
                "age": age,
                "dataset": "II",
                "image_features": feature_row.values[0][1:],
            }
        )

    return sample_dicts


def get_ACE_subjects() -> List[dict]:
    ACE_DATA_FOLDER = Path(ACE_MRI_FOLDER).resolve()
    ACE_T1s = ACE_DATA_FOLDER.glob(
        "**/sub-*_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz"
    )
    ACE_phenotype_file = pd.read_csv(ACE_PHENOTYPE)
    ACE_ig_file = pd.read_csv(ACE_FILE_with_relatedness)
    ACE_ig_file = ACE_ig_file.drop(columns=ACE_ig_file.columns[0])
    ACE_img = pd.read_csv(ACE_IMG_FEAT)

    # remove the lines if 'Cohort' == 'US' and nan
    ACE_ig_file = ACE_ig_file[
        ~((ACE_ig_file["Cohort"] == "US") | (ACE_ig_file["Cohort"].isna()))
    ]
    ACE_ig_file = ACE_ig_file[ACE_ig_file["modality"] == "joint"].copy()
    ACE_site_ids = ACE_ig_file["Site ID"].unique().tolist()
    ACE_site_ids = [str(site_id)[-5:] for site_id in ACE_site_ids]

    sample_dicts = []
    for path in ACE_T1s:
        pattern = re.compile(r"sub-([A-Za-z0-9]+)_")
        m = pattern.search(str(path))
        if m:
            subject_id = m.group(1)

        row = ACE_phenotype_file[ACE_phenotype_file["Site ID"] == subject_id]
        img_row = ACE_img[ACE_img["Site ID"] == subject_id]

        # Check if row is empty before accessing
        if row.empty or img_row.empty:
            print(
                f"Warning: No phenotype or image features data found for subject {subject_id}. Skipping."
            )
            continue

        subject_id = subject_id[-5:]
        if subject_id not in set(ACE_site_ids):
            continue

        # replace part of the string
        mask_path = str(path).replace("preproc_T1w", "brain_mask")
        sample_dicts.append(
            {
                "img": str(path),
                "mask": mask_path,
                "label": 0 if row["Cohort"].values[0] == "CON" else 1,
                "subject_id": subject_id,
                "site_id": row["Site ID"].values[0],
                "image_features": img_row.values[0][2:],
                "New ID": row["New ID"].values[0],
            }
        )

    return sample_dicts


def set_requires_grad(nets, requires_grad=False):
    """
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def add_log(
    model: str,
    y_label: Tensor,
    output: Tensor,
    writer: SummaryWriter,
    hparams: Namespace,
    fold: int,
    epoch: int,
    result_dict: dict = None,
) -> float:
    y_pred = (torch.sigmoid(output) > 0.5).detach().cpu().numpy().astype(int)
    # y_label = y_label.astype(int)
    f1 = metrics.f1_score(y_label, y_pred, average="macro")
    acc = metrics.accuracy_score(y_label, y_pred)
    kapper = metrics.cohen_kappa_score(y_label, y_pred)
    # compute auc
    # sort y_label and y_pred by y_pred
    output = output.squeeze().detach().cpu().numpy()
    sorted_indices = np.argsort(output)
    sorted_y_label = y_label[sorted_indices]
    sorted_y_pred = output[sorted_indices]
    auc = metrics.roc_auc_score(sorted_y_label, sorted_y_pred)
    sensitivity = metrics.recall_score(y_label, y_pred, pos_label=1)
    specificity = metrics.recall_score(y_label, y_pred, pos_label=0)

    # if model == "test":
    #     if not hparams.not_write_tensorboard:
    #         data = {
    #             "F1": [f1],
    #             "Acc": [acc],
    #             "Kappa": [kapper],
    #             "Sensitivity": [sensitivity],
    #             "Specificity": [specificity],
    #             "AUC": [auc],
    #         }
    # print("loging test image")
    # test_metric_plot = sns.pointplot(
    #     data=pd.DataFrame(data), join=False, linestyles="none"
    # )
    # writer.add_figure("test_metric", test_metric_plot.get_figure())

    if hparams.not_write_tensorboard:
        result_dict["mode"].append(model)
        result_dict["fold"].append(fold)
        result_dict["Epoch"].append(epoch)
        result_dict["Acc"].append(acc)
        result_dict["F1"].append(f1)
        result_dict["Kappa"].append(kapper)
        result_dict["Sensitivity"].append(sensitivity)
        result_dict["Specificity"].append(specificity)
        result_dict["AUC"].append(auc)
    else:
        writer.add_scalar(f"{model}_metric/F1", f1, epoch)
        writer.add_scalar(f"{model}_metric/Acc", acc, epoch)
        writer.add_scalar(f"{model}_metric/Kappa", kapper, epoch)
        writer.add_scalar(f"{model}_metric/Sensitivity", sensitivity, epoch)
        writer.add_scalar(f"{model}_metric/Specificity", specificity, epoch)
        writer.add_scalar(f"{model}_metric/AUC", auc, epoch)

    return acc


def predict(input_imgs, patch_embed_block, classifier, patch_size=96):
    """
    Unified prediction function for both training and testing.

    Args:
        input_imgs: Input MRI images [batch_size, channels, D, H, W]
        patch_embed_block: The image encoder model
        classifier: The classification model
        patch_size: Size of patches for processing (default: 96)

    Returns:
        predictions: Model predictions [batch_size, num_classes]
        reconstructed_features: 5D features [batch_size, channels, Z, Y, X]
    """
    # Process MRI patches using utility function
    reconstructed_features = process_mri_patches_with_encoder(
        input_imgs, patch_embed_block, patch_size=patch_size
    )
    B, C_out, Z, Y, X = reconstructed_features.shape

    # Convert to float32 if mixed precision was used (encoder may output float16)
    if reconstructed_features.dtype == torch.float16:
        reconstructed_features = reconstructed_features.float()

    flat_feats = reconstructed_features.view(B, C_out, Z * Y * X)  # [B, C_out, 216]
    logits = classifier(flat_feats)  # [B, num_classes]
    return logits, reconstructed_features


def log_different_t_SNE_map(
    X_test_img: Tensor,
    X_test_gene: Tensor,
    latent_ls: Tensor,
    y_true: np.ndarray,
    writer: SummaryWriter,
    hparams: Namespace,
    epoch: int,
    mode: str,
):
    # log the t-SNE plot
    if hparams.input_modality == "img" or hparams.input_modality == "joint":
        # if epoch == 0 or mode == "test":
        #     X_test_img_np = X_test_img.cpu().numpy()
        #     tsne_original = TSNE(
        #         n_components=2, random_state=42, perplexity=X_test_img_np.shape[0] // 2
        #     )
        #     X_tsne = tsne_original.fit_transform(X_test_img_np)
        #     log_t_SNE_map(
        #         X_tsne=X_tsne,
        #         y=y_true,
        #         modality=hparams.input_modality,
        #         writer=writer,
        #         mode=mode,
        #         title="original_space",
        #         epoch=epoch,
        #     )
        tsne_latent = TSNE(
            n_components=2, random_state=42, perplexity=X_test_img.shape[0] // 2
        )
        X_tsne = tsne_latent.fit_transform(latent_ls.cpu().numpy())
        log_t_SNE_map(
            X_tsne=X_tsne,
            y=y_true,
            modality=hparams.input_modality,
            writer=writer,
            mode=mode,
            title="latent_space",
            epoch=epoch,
        )
    elif hparams.input_modality == "gene":
        if epoch == 0 or mode == "test":
            X_test_gene_np = X_test_gene.cpu().numpy()
            tsne_original = TSNE(
                n_components=2, random_state=42, perplexity=X_test_gene_np.shape[0] // 2
            )
            X_tsne = tsne_original.fit_transform(X_test_gene_np)
            log_t_SNE_map(
                X_tsne=X_tsne,
                y=y_true,
                modality=hparams.input_modality,
                writer=writer,
                mode=mode,
                title="original_space",
                epoch=epoch,
            )
        tsne_latent = TSNE(
            n_components=2, random_state=42, perplexity=X_test_gene.shape[0] // 2
        )

        X_tsne = tsne_latent.fit_transform(latent_ls.cpu().numpy())
        log_t_SNE_map(
            X_tsne=X_tsne,
            y=y_true,
            modality=hparams.input_modality,
            writer=writer,
            mode=mode,
            title="latent_space",
            epoch=epoch,
        )


def log_t_SNE_map(
    X_tsne: np.ndarray,
    y: np.ndarray,
    modality: str,
    writer: SummaryWriter,
    mode: str,
    title: str,
    epoch: int,
):
    # merge the t-SNE result with the label
    df = pd.DataFrame(X_tsne, columns=["Component 1", "Component 2"])
    df["y"] = y
    plot = sns.scatterplot(
        x="Component 1",
        y="Component 2",
        hue="y",
        data=df,
    )
    writer.add_figure(
        f"Epoch: {epoch}, {mode} - {modality}_{title}",
        plot.get_figure(),
        epoch,
    )


def class_label_name(hparams: Namespace):
    if hparams.n_classes == 2:
        class_label = ["CN", "AD"]
    elif hparams.n_classes == 3:
        class_label = ["CN", "MCI", "AD"]
    elif hparams.n_classes == 4:
        class_label = ["CN", "EMCI", "LMCI", "AD"]
    return class_label


def initialize_check_losses(check_losses, s, num, test_n):
    for i in s:
        if i == "class_pred" or i == "class_true":
            check_losses[i] = -np.ones((num, test_n))
        else:
            check_losses[i] = np.zeros((num))
    return


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def create_train_val_test_on_ACE(
    X: pd.DataFrame,
    y: pd.DataFrame,
    train_index: np.ndarray,
    folders: list,
    X_unaffected_sibling: pd.DataFrame,
    test_fold_id: int,
    val_fold_id: int,
):
    X_train, y_train = X.iloc[train_index], y.iloc[train_index]
    X_val, y_val = (
        X.iloc[folders[val_fold_id]],
        y.iloc[folders[val_fold_id]],
    )
    n_control = (y_val == 1).sum()
    X_val_control = X_unaffected_sibling.sample(n_control)

    X_val = X_val.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)

    X_val_asd = X_val[y_val == 1]
    X_val = pd.concat([X_val_control, X_val_asd])
    y_val = pd.concat([pd.Series([0] * n_control), pd.Series([1] * len(X_val_asd))])
    X_test, y_test = (
        X.iloc[folders[test_fold_id]],
        y.iloc[folders[test_fold_id]],
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def normalize_data(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
):
    X_train_normalized = {}
    X_val_normalized = {}
    X_test_normalized = {}

    # # fill NaN value with mean in the training set
    # for col in X_train.columns:
    #     mean = X_train[col].mean()
    #     X_train.loc[:, col] = X_train[col].fillna(mean)
    #     X_val.loc[:, col] = X_val[col].fillna(mean)
    #     X_test.loc[:, col] = X_test[col].fillna(mean)

    for col in X_train.columns:
        mean, std = X_train[col].mean(), X_train[col].std()
        X_train_normalized[col] = (X_train[col] - mean) / std
        X_val_normalized[col] = (X_val[col] - mean) / std
        X_test_normalized[col] = (X_test[col] - mean) / std

    X_train = pd.DataFrame(X_train_normalized).values
    X_val = pd.DataFrame(X_val_normalized).values
    X_test = pd.DataFrame(X_test_normalized).values

    return X_train, X_val, X_test


def normalize_fastsurfer_features(train_samples, val_samples, test_samples):
    """
    Normalize FastSurfer image features using training set statistics.
    Similar to normalize_data() - computes mean and std for each feature individually.

    Args:
        train_samples: List of dicts with 'image_features' key
        val_samples: List of dicts with 'image_features' key
        test_samples: List of dicts with 'image_features' key

    Returns:
        train_samples, val_samples, test_samples, train_mean, train_std
    """
    print(f"\n=== FastSurfer Feature Normalization ===")
    print(f"Training samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")
    print(f"Test samples: {len(test_samples)}")

    # Get number of features
    num_features = len(train_samples[0]["image_features"])
    print(f"Number of features: {num_features}")

    # Step 1: Compute mean and std for each feature from TRAINING SET ONLY
    train_mean = []
    train_std = []

    for i in range(num_features):
        # Extract feature i from all training samples
        feature_values = [
            float(sample["image_features"][i]) for sample in train_samples
        ]

        # Compute mean and std for this feature
        mean_i = np.mean(feature_values)
        std_i = np.std(feature_values)

        train_mean.append(mean_i)
        train_std.append(std_i)

    # Convert to numpy arrays
    train_mean = np.array(train_mean, dtype=np.float64)
    train_std = np.array(train_std, dtype=np.float64)

    print(f"Train mean range: [{train_mean.min():.4f}, {train_mean.max():.4f}]")
    print(f"Train std range: [{train_std.min():.4f}, {train_std.max():.4f}]")

    # Step 2: Normalize each sample using training statistics
    # Normalize training set
    for sample in train_samples:
        normalized = []
        for i in range(num_features):
            val = float(sample["image_features"][i])
            norm_val = (val - train_mean[i]) / (train_std[i] + 1e-8)
            normalized.append(norm_val)
        sample["image_features"] = np.array(normalized, dtype=np.float32)

    # Normalize validation set
    for sample in val_samples:
        normalized = []
        for i in range(num_features):
            val = float(sample["image_features"][i])
            norm_val = (val - train_mean[i]) / (train_std[i] + 1e-8)
            normalized.append(norm_val)
        sample["image_features"] = np.array(normalized, dtype=np.float32)

    # Normalize test set
    for sample in test_samples:
        normalized = []
        for i in range(num_features):
            val = float(sample["image_features"][i])
            norm_val = (val - train_mean[i]) / (train_std[i] + 1e-8)
            normalized.append(norm_val)
        sample["image_features"] = np.array(normalized, dtype=np.float32)

    # Verify normalization on first feature
    feature_0_values = [sample["image_features"][0] for sample in train_samples]
    print(f"\nAfter normalization (feature 0):")
    print(f"  mean={np.mean(feature_0_values):.6f}, std={np.std(feature_0_values):.6f}")
    print("✓ FastSurfer features normalized\n")

    return train_samples, val_samples, test_samples, train_mean, train_std


def create_y_paired_labels(y: pd.DataFrame, hparams: Namespace):
    ys = []
    for i in range(y.shape[0]):
        if y[i][0] == y[i][1]:
            ys.append(0)
        else:
            ys.append(1)

    y = np.array(ys)
    return y


def save_result_dataframe(
    result_fold_path: Path,
    df_result: pd.DataFrame,
    hparams: Namespace,
) -> None:
    result_file = result_fold_path / f"result_n_folder_{hparams.n_folder}.csv"
    overall_result_file = RESULT_FOLDER / f"{hparams.dataset}_overall_result.csv"

    df = pd.DataFrame.from_dict(df_result)
    if result_file.exists():
        # load the existing result file
        df_previous = pd.read_csv(result_file)
        # append the new result to the existing result file
        df = pd.concat([df_previous, df], ignore_index=True)
        os.remove(result_file)
    df.to_csv(result_file, index=False)

    # check the number of fold in df
    n_fold = df["fold"].nunique()
    if n_fold == 10:
        final_overall_result = defaultdict(list)
        metrics = ["Acc", "Sensitivity", "Specificity", "AUC"]
        modes = ["test", "val"]
        for epoch in range(100, hparams.n_epochs, 100):
            final_overall_result["name"].append(hparams.experiment_name)
            final_overall_result["Epoch"].append(epoch)
            for mode in modes:
                print("mode", mode)
                for metric in metrics:
                    value = (
                        (
                            df[(df["mode"] == mode) & (df["Epoch"] == epoch)].groupby(
                                "fold"
                            )[metric]
                        )
                        .mean()
                        .mean()
                    )
                    final_overall_result[f"{mode}_{metric}"].append(value)

        overall_df = pd.DataFrame.from_dict(final_overall_result)
        if overall_result_file.exists():
            df_previous = pd.read_csv(overall_result_file)
            overall_df = pd.concat([df_previous, overall_df], ignore_index=True)
            os.remove(overall_result_file)
        overall_df.to_csv(overall_result_file, index=False)


def preprocess_df_SSC(input_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    label = input_df["PHENO1"].replace({1: 0, 2: 1}).astype(int)
    pathway = input_df.drop(columns=["PHENO1", "ID"])
    return pathway, label


def preprocess_df_ACE(input_df: pd.DataFrame):
    # only choose rows with modality == `joint`
    input_df = input_df[input_df["modality"] == "joint"].copy()
    # print out the # of subjects that are "US"
    print(
        f"Number of subjects with Cohort 'US': {len(input_df[input_df['Cohort']=='US'])}"
    )
    pathway_data_disease_control = input_df[
        (input_df["Cohort"] == "CON") | (input_df["Cohort"] == "ASD")
    ]
    img = pathway_data_disease_control.iloc[:, 181:]
    pathway = pathway_data_disease_control.iloc[:, :181]
    label = pathway["Cohort"].replace({"CON": 0, "ASD": 1}).astype(int)
    father_site_ids = img["Father Site ID"]
    new_ids = pathway["New ID"]

    img = img.drop(columns=["Gender", "Father Site ID", "Mother Site ID"])
    pathway = pathway.drop(columns=["Site ID", "New ID", "modality", "Cohort"])

    return img, pathway, label, father_site_ids, new_ids


def preprocess_df_AD(input_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    input_df = input_df.drop(
        columns=[
            "ST81CV_UCSFFSX_11_02_15_UCSFFSX51_08_01_16",
            "ST81SA_UCSFFSX_11_02_15_UCSFFSX51_08_01_16",
            "ST81TA_UCSFFSX_11_02_15_UCSFFSX51_08_01_16",
            "ST81TS_UCSFFSX_11_02_15_UCSFFSX51_08_01_16",
            "ST22CV_UCSFFSX_11_02_15_UCSFFSX51_08_01_16",
            "ST22SA_UCSFFSX_11_02_15_UCSFFSX51_08_01_16",
            "ST22TA_UCSFFSX_11_02_15_UCSFFSX51_08_01_16",
            "ST22TS_UCSFFSX_11_02_15_UCSFFSX51_08_01_16",
        ]
    )

    # drop the row with index 549, 886, 914, 915, 920, 923
    input_df = input_df.drop([549, 886, 914, 915, 920, 923])
    # for each row, print out the indx that the more than 50% of the columns are space
    pathway_data_disease_control = input_df[
        (input_df["DX_bl"] == "AD") | (input_df["DX_bl"] == "CN")
    ].copy()

    # check whether "PTID.1" in the column names
    if "PTID.1" in pathway_data_disease_control.columns:
        img = pathway_data_disease_control.iloc[:, -276:]
        pathway = pathway_data_disease_control.iloc[:, :-276]

        img.drop(
            columns=[
                "PTID.1",
                "DX_bl.1",
                "AGE.1",
                "PTGENDER.1",
            ],
            inplace=True,
        )
    else:
        img = pathway_data_disease_control.iloc[:, -272:]
        pathway = pathway_data_disease_control.iloc[:, :-272]
    label = pathway["DX_bl"].replace({"CN": 0, "AD": 1}).astype(int)

    pathway.drop(
        columns=[
            "PTID",
            "DX_bl",
            "AGE",
            "PTGENDER",
        ],
        inplace=True,
    )

    return img.astype(float), pathway, label


# before training the alignment model, get the acc from the test set
def pretrained_model_result(
    image_encoder: torch.nn.Module,
    image_classifier: torch.nn.Module,
    genetics_encoder: torch.nn.Module,
    test_img_loader: DataLoader,
    test_gen_loader: DataLoader,
    device: torch.device,
) -> float:
    image_encoder.eval()
    image_classifier.eval()
    genetics_encoder.eval()

    img_test_labels = []
    img_test_preds = []

    with torch.no_grad():
        for batch in test_img_loader:
            labels = batch["label"].to(device).view(-1)
            input_imgs = batch["img"].to(device)
            logits, _ = predict(
                input_imgs, image_encoder, image_classifier, patch_size=96
            )
            preds = (torch.sigmoid(logits.squeeze(-1)) > 0.5).long()
            img_test_labels.extend(labels.cpu().numpy())
            img_test_preds.extend(preds.cpu().numpy())

    gen_test_labels = []
    gen_test_preds = []

    with torch.no_grad():
        for batch in test_gen_loader:
            labels = batch["label"].to(device).view(-1)
            pathways = batch["pathway"].to(device).float()
            _, logits = genetics_encoder(pathways)
            preds = (torch.sigmoid(logits.squeeze(-1)) > 0.5).long()
            gen_test_labels.extend(labels.cpu().numpy())
            gen_test_preds.extend(preds.cpu().numpy())

    # Convert to numpy for easier analysis
    img_test_labels = np.array(img_test_labels)
    img_test_preds = np.array(img_test_preds)
    gen_test_labels = np.array(gen_test_labels)
    gen_test_preds = np.array(gen_test_preds)

    # Calculate correctness
    img_correct = img_test_preds == img_test_labels
    gen_correct = gen_test_preds == gen_test_labels

    img_correct_count = img_correct.sum()
    gen_correct_count = gen_correct.sum()

    print(f"\n{'='*70}")
    print("PREDICTION CORRECTNESS ANALYSIS")
    print(f"{'='*70}")
    print(
        f"Imaging: {img_correct_count}/{len(img_test_labels)} correct ({img_correct_count/len(img_test_labels)*100:.1f}%)"
    )
    print(
        f"Genetics: {gen_correct_count}/{len(gen_test_labels)} correct ({gen_correct_count/len(gen_test_labels)*100:.1f}%)"
    )
