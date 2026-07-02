import torch
import torch.nn as nn
import torch.nn.functional as F


class PathwayDecoder(nn.Module):
    """
    Decoder for reconstructing genetic pathway values from latent representations.

    Mirrors the PathwayEncoder architecture in reverse.

    Examples:
        encoder_dims=[1,6], output_features=8
        → Encoder: 1 → 6 → 8
        → Decoder: 8 → 6 → 1

        encoder_dims=[1,6,3], output_features=8
        → Encoder: 1 → 6 → 3 → 8
        → Decoder: 8 → 3 → 6 → 1
    """

    def __init__(
        self,
        n_pathway: int,
        encoder_dims: list,       # e.g., [1, 6] or [1, 6, 3]
        output_features: int,     # e.g., 8
        normalization: str = "layer",
    ):
        """
        Args:
            n_pathway: Number of genetic pathways (e.g., 300)
            encoder_dims: List of encoder dimensions (hidden layers), e.g., [1, 6] or [1, 6, 3]
            output_features: Output dimension (separate, must match imaging)
            normalization: Type of normalization ("batch", "layer", or "None")
        """
        super(PathwayDecoder, self).__init__()

        self.n_pathway = n_pathway
        self.encoder_dims = encoder_dims
        self.output_features = output_features
        self.normalization = normalization

        # Full decoder dims: [output_features] + reversed(encoder_dims[1:]) + [1]
        # Example: encoder_dims=[1,6,3], output=8 → decoder=[8,3,6,1]
        decoder_hidden = encoder_dims[1:][::-1]  # Reverse, skip first 1
        self.decoder_dims = [output_features] + decoder_hidden + [1]
        self.n_layers = len(self.decoder_dims) - 1

        # Create decoder layers dynamically
        self.decoder_params = nn.ParameterList()
        self.norm_layers = nn.ModuleList()

        for i in range(self.n_layers):
            in_dim = self.decoder_dims[i]
            out_dim = self.decoder_dims[i + 1]

            # Linear layer as Parameter
            param = nn.Parameter(torch.randn(in_dim, out_dim))
            nn.init.xavier_uniform_(param)
            self.decoder_params.append(param)

            # Normalization (skip for last layer)
            if i < self.n_layers - 1:
                if self.normalization == "batch":
                    norm = nn.BatchNorm1d(out_dim)
                elif self.normalization == "layer":
                    norm = nn.LayerNorm(out_dim)
                else:
                    norm = nn.Identity()
                self.norm_layers.append(norm)

    def forward(self, latent):
        """
        Reconstruct pathway values from latent representation.

        Args:
            latent: [batch_size, n_pathway, output_features] or [batch_size, output_features, n_pathway]
                   Latent representation from encoder

        Returns:
            reconstructed: [batch_size, n_pathway] Reconstructed pathway values
        """
        # Handle different input formats
        if latent.shape[1] == self.output_features and latent.shape[2] == self.n_pathway:
            # Input is [batch, output_features, n_pathway] - transpose to [batch, n_pathway, output_features]
            latent = latent.transpose(1, 2)

        batch_size = latent.shape[0]
        out = latent

        # Apply decoder layers dynamically
        for i in range(self.n_layers):
            # Linear transformation
            out = torch.matmul(out, self.decoder_params[i])

            # Normalization + ReLU (except last layer)
            if i < self.n_layers - 1:
                if self.normalization == "batch":
                    out = out.reshape(-1, self.decoder_dims[i + 1])
                    out = self.norm_layers[i](out)
                    out = out.view(batch_size, self.n_pathway, self.decoder_dims[i + 1])
                else:
                    out = self.norm_layers[i](out)
                out = F.relu(out)

        # Squeeze last dimension [batch, n_pathway, 1] -> [batch, n_pathway]
        reconstructed = out.squeeze(-1)

        return reconstructed


class PathwayReconstructionLoss(nn.Module):
    """
    Loss for pathway reconstruction. Uses MSE loss for pathway values.
    """

    def __init__(self, loss_type: str = "mse"):
        """
        Args:
            loss_type: Type of loss ("mse" or "l1")
        """
        super(PathwayReconstructionLoss, self).__init__()

        if loss_type == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss_type == "l1":
            self.loss_fn = nn.L1Loss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def forward(self, reconstructed, target):
        """
        Compute reconstruction loss.

        Args:
            reconstructed: [batch_size, n_pathway] Reconstructed pathway values
            target: [batch_size, n_pathway] Original pathway values

        Returns:
            loss: Scalar reconstruction loss
        """
        return self.loss_fn(reconstructed, target)
