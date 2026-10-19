"""Local masked-context module used by the cDDPM-SparK + MAEDiff experiment."""

import torch
from torch import nn
import torch.nn.functional as F


class MaskedGridContext(nn.Module):
    """Inject MAE-style visible-grid context into one U-Net feature map.

    The diffusion target patch is hidden from the MAE encoder.  The decoder uses
    cross-attention so that *all* grid tokens, including the hidden target grids,
    can query the visible context.  Keeping this module at a low-resolution U-Net
    stage limits direct pixel-wise copying from the conditioning pathway.
    """

    def __init__(self, channels, grid_size=4, embed_dim=None, heads=4, depth=1,
                 max_tokens=1024):
        super().__init__()
        self.channels = channels
        self.grid_size = grid_size
        self.embed_dim = embed_dim or channels
        self.max_tokens = max_tokens
        patch_dim = channels * grid_size * grid_size

        self.tokenize = nn.Linear(patch_dim, self.embed_dim)
        self.detokenize = nn.Linear(self.embed_dim, patch_dim)
        self.position = nn.Parameter(torch.zeros(1, max_tokens, self.embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim, nhead=heads, dim_feedforward=self.embed_dim * 4,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.visible_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.cross_attention = nn.MultiheadAttention(
            self.embed_dim, heads, batch_first=True,
        )
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim, nhead=heads, dim_feedforward=self.embed_dim * 4,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(self.embed_dim)
        # Start conservatively: training must learn how much local context to use.
        self.gate = nn.Parameter(torch.tensor(-2.0))

    def forward(self, features, context_mask):
        """Return a gated local-context residual for ``features``.

        Args:
            features: U-Net feature map [B, C, H, W].
            context_mask: [B, 1, H0, W0], one where MAE context is hidden.
        """
        batch, channels, height, width = features.shape
        if height % self.grid_size or width % self.grid_size:
            raise ValueError(
                f"Feature map {(height, width)} is not divisible by grid_size={self.grid_size}."
            )

        # MAEDiff's sub-level grids live on the feature map, not on a second image.
        tokens = F.unfold(features, kernel_size=self.grid_size, stride=self.grid_size)
        tokens = tokens.transpose(1, 2)  # [B, N, C * grid_size^2]
        num_tokens = tokens.shape[1]
        if num_tokens > self.max_tokens:
            raise ValueError(f"Need {num_tokens} positional tokens; max_tokens={self.max_tokens}.")

        dtype = self.tokenize.weight.dtype
        tokens = self.tokenize(tokens.to(dtype)) + self.position[:, :num_tokens]

        # A grid is hidden when any original-resolution pixel in it belongs to the
        # target/context exclusion mask.  Nearest pooling preserves the patch layout.
        grid_mask = F.adaptive_max_pool2d(context_mask.float(), (height // self.grid_size, width // self.grid_size))
        grid_mask = grid_mask.flatten(1).bool()

        decoded = []
        for sample_tokens, sample_mask in zip(tokens, grid_mask):
            visible = sample_tokens[~sample_mask]
            if visible.numel() == 0:
                raise ValueError("The context mask hides every grid; reduce patch/context size.")
            visible = self.visible_encoder(visible.unsqueeze(0))
            query = sample_tokens.unsqueeze(0)
            attended, _ = self.cross_attention(query, visible, visible, need_weights=False)
            decoded.append(self.decoder(self.norm(query + attended)))
        decoded = torch.cat(decoded, dim=0)

        patches = self.detokenize(decoded).transpose(1, 2)
        residual = F.fold(
            patches, output_size=(height, width), kernel_size=self.grid_size, stride=self.grid_size,
        )
        # Sigmoid gating keeps this extra pathway from overwhelming the pretrained U-Net.
        return residual.to(features.dtype) * torch.sigmoid(self.gate)
