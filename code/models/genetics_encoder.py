import sys
import math

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

from models.pathway_decoder import PathwayDecoder, PathwayReconstructionLoss

# from models.sparsemax import Sparsemax


class PathwayEncoder(nn.Module):
    def __init__(
        self,
        n_pathway,
        classifier_latent_dim,
        normalization: str,
        relu_at_coattention: bool,
        input_channels: int = 1,
        encoder_hidden_dim_1: int = 32,
        encoder_hidden_dim_2: int = 16,
        encoder_hidden_dim_3: int = None,
        output_features: int = 8,
        gamma: float = 1.0,
        soft_sign_constant: float = 0.5,
        use_reconstruction: bool = False,
        pathway_dropout: float = 0.0,
        encoder_dropout: float = 0.0,
        classifier_dropout: float = 0.5,
    ):
        super(PathwayEncoder, self).__init__()
        self.n_pathway = n_pathway
        self.output_features = output_features
        self.encoder_hidden_dim_1 = encoder_hidden_dim_1
        self.encoder_hidden_dim_2 = encoder_hidden_dim_2
        self.encoder_hidden_dim_3 = encoder_hidden_dim_3
        self.input_channels = input_channels
        self.query_d = input_channels
        self.key_value_d = 4
        self.classifier_latent_dim = classifier_latent_dim
        self.normalization = normalization
        self.relu_at_coattention = relu_at_coattention
        self.gamma = gamma
        self.variance_reduction_factor = self.gamma * (self.n_pathway**0.5 / 2)
        self.soft_sign_constant = soft_sign_constant
        self.use_reconstruction = use_reconstruction
        self.pathway_dropout = pathway_dropout
        self.encoder_dropout = encoder_dropout
        self.classifier_dropout = classifier_dropout

        # Classifier dropout (applied after combining all pathways)
        self.dropout_1 = nn.Dropout(self.classifier_dropout)
        self.relu_1 = nn.LeakyReLU(0.2)
        self.classifier_1 = nn.Linear(
            self.n_pathway * self.output_features, self.classifier_latent_dim
        )
        self.dropout_2 = nn.Dropout(self.classifier_dropout)
        self.relu_2 = nn.LeakyReLU(0.2)
        self.classifier_2 = nn.Linear(self.classifier_latent_dim, 1)
        # self.sparsemax = Sparsemax(dim=-1)

        # Pathway-specific weight banks: one weight matrix per pathway per layer
        self.encoder_linear_1 = nn.Parameter(
            torch.randn(self.n_pathway, self.query_d, self.encoder_hidden_dim_1)
        )
        self.encoder_linear_2 = nn.Parameter(
            torch.randn(
                self.n_pathway, self.encoder_hidden_dim_1, self.encoder_hidden_dim_2
            )
        )
        if self.encoder_hidden_dim_3 is not None:
            self.encoder_linear_3 = nn.Parameter(
                torch.randn(
                    self.n_pathway, self.encoder_hidden_dim_2, self.encoder_hidden_dim_3
                )
            )
            self.encoder_linear_out = nn.Parameter(
                torch.randn(
                    self.n_pathway, self.encoder_hidden_dim_3, self.output_features
                )
            )
        else:
            self.encoder_linear_3 = nn.Parameter(
                torch.randn(
                    self.n_pathway, self.encoder_hidden_dim_2, self.output_features
                )
            )
            self.encoder_linear_out = None
        # Normalization modules are built for every supported mode so that
        # `encoder()`/`forward()` can apply them unconditionally. `layer`/`None`
        # keep their original semantics; `batch`/`instance` are wired through
        # `_apply_norm`, which handles the [B, P, F] <-> channel-first plumbing.
        self.norm_1 = self._make_norm(
            self.n_pathway * self.output_features, per_pathway=False
        )
        self.norm_2 = self._make_norm(self.classifier_latent_dim, per_pathway=False)
        self.query_norm = self._make_norm(self.encoder_hidden_dim_1, per_pathway=True)
        self.query_norm_2 = self._make_norm(self.encoder_hidden_dim_2, per_pathway=True)
        if self.encoder_hidden_dim_3 is not None:
            self.query_norm_3 = self._make_norm(
                self.encoder_hidden_dim_3, per_pathway=True
            )
        else:
            self.query_norm_3 = nn.Identity()
        self.output_norm = self._make_norm(
            self.output_features, per_pathway=True
        )  # Normalization before final activation

        # Activation functions for encoder layers (to match imaging encoder architecture)
        self.encoder_activation_1 = nn.LeakyReLU(0.2)
        self.encoder_activation_2 = nn.LeakyReLU(0.2)
        self.encoder_activation_3 = nn.LeakyReLU(0.2)
        self.encoder_activation_out = nn.LeakyReLU(0.2)
        self.query_relu = nn.LeakyReLU(0.2)
        self.query_relu_2 = nn.LeakyReLU(0.2)
        self.query_relu_3 = nn.LeakyReLU(0.2)

        # initialize the weights
        nn.init.xavier_uniform_(self.classifier_1.weight)
        nn.init.zeros_(self.classifier_1.bias)
        # Initialize per-pathway encoder weights
        for i in range(self.n_pathway):
            nn.init.xavier_uniform_(self.encoder_linear_1[i])
            nn.init.xavier_uniform_(self.encoder_linear_2[i])
            nn.init.xavier_uniform_(self.encoder_linear_3[i])
            if self.encoder_linear_out is not None:
                nn.init.xavier_uniform_(self.encoder_linear_out[i])

        # Initialize decoder if using reconstruction
        if self.use_reconstruction:
            # Build encoder_dims list from parameters
            encoder_dims = [
                self.input_channels,
                self.encoder_hidden_dim_1,
                self.encoder_hidden_dim_2,
            ]
            if self.encoder_hidden_dim_3 is not None:
                encoder_dims.append(self.encoder_hidden_dim_3)
            self.decoder = PathwayDecoder(
                n_pathway=self.n_pathway,
                encoder_dims=encoder_dims,
                output_features=self.output_features,
                normalization=self.normalization,
            )
            self.reconstruction_loss_fn = PathwayReconstructionLoss(loss_type="mse")

    def _make_norm(self, num_features, per_pathway):
        """Build the normalization module for the configured mode.

        `per_pathway` marks tensors carrying a pathway axis ([B, P, F]); flat
        classifier tensors ([B, D]) have no spatial axis for instance-norm, so
        they fall back to BatchNorm in `instance` mode. Pairs with `_apply_norm`.
        """
        if self.normalization == "layer":
            return nn.LayerNorm(num_features)
        if self.normalization == "batch":
            return nn.BatchNorm1d(num_features)
        if self.normalization == "instance":
            if per_pathway:
                return nn.InstanceNorm1d(num_features, affine=True)
            return nn.BatchNorm1d(num_features)
        # "None" (or any unrecognized value) -> no-op
        return nn.Identity()

    def _apply_norm(self, norm, x):
        """Apply a module from `_make_norm` to x of shape [B, P, F] or [B, D].

        LayerNorm/Identity operate on the last dim directly. BatchNorm1d and
        InstanceNorm1d expect channels in dim 1, so reshape/transpose here.
        """
        if isinstance(norm, (nn.LayerNorm, nn.Identity)):
            return norm(x)
        if x.dim() == 3:  # [B, P, F], channels = F
            if isinstance(norm, nn.InstanceNorm1d):
                # Normalize each feature channel across the pathway axis per sample.
                return norm(x.transpose(1, 2)).transpose(1, 2)
            b, p, f = x.shape  # BatchNorm1d over flattened tokens
            return norm(x.reshape(b * p, f)).view(b, p, f)
        return norm(x)  # [B, D] -> BatchNorm1d handles 2D directly

    def encoder(self, genetics_modality):
        # First layer transformation (per-pathway weights)
        latent = torch.einsum("bph,phd->bpd", genetics_modality, self.encoder_linear_1)
        latent = self._apply_norm(self.query_norm, latent)
        if self.relu_at_coattention:
            latent = self.query_relu(latent)
        # Add activation after first layer
        latent = self.encoder_activation_1(latent)

        if self.encoder_dropout > 0:
            latent = F.dropout(latent, p=self.encoder_dropout, training=self.training)

        latent = torch.einsum("bpd,pdk->bpk", latent, self.encoder_linear_2)
        latent = self._apply_norm(self.query_norm_2, latent)
        if self.relu_at_coattention:
            latent = self.query_relu_2(latent)
        # Add activation after second layer
        latent = self.encoder_activation_2(latent)

        if self.encoder_dropout > 0:
            latent = F.dropout(latent, p=self.encoder_dropout, training=self.training)

        # Output layer transformation (this is what classifier AND reconstruction use)
        latent = torch.einsum("bpk,pkm->bpm", latent, self.encoder_linear_3)
        if self.encoder_hidden_dim_3 is not None:
            latent = self._apply_norm(self.query_norm_3, latent)
            latent = self.encoder_activation_3(latent)
            if self.encoder_dropout > 0:
                latent = F.dropout(
                    latent, p=self.encoder_dropout, training=self.training
                )
            latent = torch.einsum("bpm,pmf->bpf", latent, self.encoder_linear_out)

        # Apply normalization before activation to match imaging encoder (Conv → Norm → Activation)
        latent = self._apply_norm(self.output_norm, latent)
        # Add activation after output layer to match imaging encoder
        latent = self.encoder_activation_out(latent)
        return latent

    def forward(self, modality, return_reconstruction=False):
        # Support both [batch, n_pathway] and [batch, n_pathway, channels] / [batch, channels, n_pathway]
        if modality.dim() == 2:
            modality_input = modality.unsqueeze(-1)  # (batch_size, n_pathway, 1)
        elif modality.dim() == 3:
            if modality.shape[1] == self.n_pathway and modality.shape[2] == self.input_channels:
                modality_input = modality
            elif modality.shape[1] == self.input_channels and modality.shape[2] == self.n_pathway:
                # Accept channel-first input and transpose to [batch, n_pathway, channels]
                modality_input = modality.transpose(1, 2)
            else:
                raise ValueError(
                    f"Unexpected genetics input shape {modality.shape}; expected (batch, n_pathway, channels) "
                    f"with n_pathway={self.n_pathway}, channels={self.input_channels}."
                )
        else:
            raise ValueError(
                f"Genetics modality must be rank-2 or rank-3 tensor, got shape {modality.shape}"
            )

        # PATHWAY DROPOUT: Randomly drop entire pathways during training
        # This forces the model to not rely on any specific pathway
        if self.pathway_dropout > 0:
            modality_input = F.dropout(
                modality_input, p=self.pathway_dropout, training=self.training
            )

        # Encode - always get just the latent (output_features)
        latent = self.encoder(
            modality_input
        )  # (batch_size, n_pathway, output_features)

        # Transpose to match imaging format: [batch, channels, positions]
        # From [batch, n_pathway, output_features] to [batch, output_features, n_pathway]
        latent_transposed = latent.transpose(
            1, 2
        )  # (batch_size, output_features, n_pathway)

        # Classifier uses original (non-transposed) latent for consistency with existing weights
        # reshape (not view): the latent may be non-contiguous after norm/transpose.
        out_squeeze = latent.reshape(
            latent.shape[0], -1
        )  # (batch_size, n_pathway * output_features)
        out_squeeze = self._apply_norm(self.norm_1, out_squeeze)
        combined = self.dropout_1(self.relu_1(out_squeeze))
        classifier_1 = self.classifier_1(combined)
        classifier_1 = self._apply_norm(self.norm_2, classifier_1)
        classifier_1 = self.dropout_2(self.relu_2(classifier_1))
        logits = self.classifier_2(classifier_1)

        # Optionally compute reconstruction from latent (output_features)
        # This forces the 8D features used by classifier to preserve pathway information
        if return_reconstruction and self.use_reconstruction:
            # Reconstruct from latent (output_features), which is what classifier uses
            reconstructed = self.decoder(latent)  # [batch_size, n_pathway]
            return latent_transposed, logits, reconstructed

        return latent_transposed, logits
