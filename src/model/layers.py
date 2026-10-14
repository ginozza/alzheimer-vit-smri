"""Individual layer implementations for ViT-B/16.

Each layer is a self-contained nn.Module that can be unit-tested independently.
Attention weights are stored for future Attention Rollout (Entregable E5).
"""

import torch
import torch.nn as nn
from typing import Optional


class PatchEmbedding(nn.Module):
    """Projects non-overlapping image patches into token embeddings.

    Converts an input image (B, C, H, W) into a sequence of patch embeddings
    with a prepended learnable [CLS] token and additive positional embeddings.

    Architecture:
        Conv2d(in_channels, embed_dim, kernel=patch_size, stride=patch_size)
        -> flatten -> prepend [CLS] -> add position embeddings

    Output shape: (B, num_patches + 1, embed_dim) = (B, 197, 768) for ViT-B/16.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_patches = (image_size // patch_size) ** 2
        self.seq_length = self.num_patches + 1

        # Linear projection of flattened patches via Conv2d
        self.projection = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )

        # Learnable [CLS] token prepended to the sequence
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Learnable absolute positional embeddings for all tokens
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, self.seq_length, embed_dim)
        )

        self.dropout = nn.Dropout(p=dropout)

        # Initialize with truncated normal (following ViT paper)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position_embeddings, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images of shape (B, C, H, W).

        Returns:
            Token embeddings of shape (B, num_patches + 1, embed_dim).
        """
        if x.ndim != 4:
            raise ValueError(f"Expected a 4D input tensor (B, C, H, W), got shape {tuple(x.shape)}")

        expected_shape = (self.in_channels, self.image_size, self.image_size)
        if tuple(x.shape[1:]) != expected_shape:
            raise ValueError(
                f"Expected input shape (B, {expected_shape[0]}, {expected_shape[1]}, "
                f"{expected_shape[2]}), got {tuple(x.shape)}"
            )
        if not torch.is_floating_point(x):
            raise TypeError(f"Expected a floating-point input tensor, got dtype {x.dtype}")

        batch_size = x.shape[0]

        # (B, C, H, W) -> (B, embed_dim, H/P, W/P) -> (B, embed_dim, num_patches)
        patches = self.projection(x)
        patches = patches.flatten(2)  # (B, embed_dim, num_patches)
        patches = patches.transpose(1, 2)  # (B, num_patches, embed_dim)

        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, embed_dim)
        embeddings = torch.cat([cls_tokens, patches], dim=1)  # (B, N+1, D)

        # Add positional embeddings
        embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings)

        return embeddings


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention mechanism with attention weight storage.

    Computes scaled dot-product attention across multiple heads. Stores
    the raw attention weights after softmax in ``self.attn_weights`` for
    downstream interpretability (Attention Rollout in E5).

    Architecture:
        Input (B, N, D) -> QKV projection -> split heads -> scaled dot-product
        -> concat heads -> output projection -> (B, N, D)
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        qkv_bias: bool = True,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Single linear layer produces Q, K, V concatenated
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(p=attn_dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(p=proj_dropout)

        # Storage for attention weights (set during forward pass)
        self.attn_weights: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, N, D).

        Returns:
            Output tensor of shape (B, N, D).
            Side effect: stores attention weights in self.attn_weights
            with shape (B, num_heads, N, N).
        """
        B, N, D = x.shape

        # Compute Q, K, V in a single projection
        qkv = self.qkv(x)  # (B, N, 3*D)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)  # each: (B, heads, N, head_dim)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N)
        attn = attn.softmax(dim=-1)

        # Store attention weights for Attention Rollout (E5)
        self.attn_weights = attn.detach()

        attn = self.attn_drop(attn)

        # Weighted sum of values
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)  # (B, N, D)
        out = self.proj(out)
        out = self.proj_drop(out)

        return out


class MLP(nn.Module):
    """Two-layer feedforward network with GELU activation.

    Architecture: Linear(D, mlp_dim) -> GELU -> Dropout -> Linear(mlp_dim, D) -> Dropout
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Single Transformer encoder block with pre-normalization.

    Architecture (pre-norm, following ViT paper):
        x -> LayerNorm -> MHSA -> + residual
          -> LayerNorm -> MLP  -> + residual

    Pre-norm is preferred over post-norm for training stability in
    deep Transformer architectures (Xiong et al., 2020).
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(
            in_features=embed_dim,
            hidden_features=int(embed_dim * mlp_ratio),
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, N, D).

        Returns:
            Output tensor of shape (B, N, D).
        """
        # Pre-norm MHSA + residual
        x = x + self.attn(self.norm1(x))
        # Pre-norm MLP + residual
        x = x + self.mlp(self.norm2(x))
        return x


class ClassificationHead(nn.Module):
    """Triclass classification head operating on the [CLS] token.

    Architecture: LayerNorm(embed_dim) -> Linear(embed_dim, num_classes)

    Takes only the [CLS] token output (index 0) from the Transformer
    encoder sequence and projects it to class logits.
    """

    def __init__(self, embed_dim: int = 768, num_classes: int = 3):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Full sequence from Transformer encoder, shape (B, N, D).

        Returns:
            Class logits of shape (B, num_classes).
        """
        cls_token = x[:, 0]  # Extract [CLS] token: (B, D)
        cls_token = self.norm(cls_token)
        logits = self.head(cls_token)
        return logits
