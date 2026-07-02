import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedLatentProjector(nn.Module):
    """
    Learnable linear maps that project modality-specific tokens into a shared latent matrix.
    """

    def __init__(self, latent_dim: int, n_rois: int, n_pathways: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_rois = n_rois
        self.n_pathways = n_pathways

        # W_I ∈ ℝ^{d × n_i}, W_G ∈ ℝ^{d × n_g}
        self.W_I = nn.Parameter(torch.empty(latent_dim, n_rois))
        self.W_G = nn.Parameter(torch.empty(latent_dim, n_pathways))
        nn.init.orthogonal_(self.W_I, gain=1.0)
        nn.init.orthogonal_(self.W_G, gain=1.0)

    def project_image_tokens(self, roi_tokens: torch.Tensor) -> torch.Tensor:
        """
        roi_tokens: [B, n_rois, d]
        Returns: [B, d, d] latent matrices
        """
        # Normalize per-token to equalize scale across modalities
        roi_tokens = F.layer_norm(roi_tokens, roi_tokens.shape[-1:])
        W = self.W_I.unsqueeze(0).expand(roi_tokens.size(0), -1, -1)
        return torch.bmm(W, roi_tokens)

    def project_genetics_tokens(self, pathway_tokens: torch.Tensor) -> torch.Tensor:
        """
        pathway_tokens: [B, n_pathways, d]
        Returns: [B, d, d] latent matrices
        """
        pathway_tokens = pathway_tokens.transpose(1, 2)
        # Normalize per-token to equalize scale across modalities
        pathway_tokens = F.layer_norm(pathway_tokens, pathway_tokens.shape[-1:])
        W = self.W_G.unsqueeze(0).expand(pathway_tokens.size(0), -1, -1)
        return torch.bmm(W, pathway_tokens)

    def association_matrix(self) -> torch.Tensor:
        """
        ROI-pathway association matrix T = W_I^T W_G ∈ R^{n_i x n_g}
        """
        return torch.matmul(self.W_I.t(), self.W_G)
