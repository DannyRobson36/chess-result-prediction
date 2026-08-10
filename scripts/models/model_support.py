"""
model_support.py
Building blocks for models

Latest changes: 10/08/26:
- Initial commit
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

####################
# CONSTANTS
####################

ACTIVATION_REGISTRY = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
}

BINARY_BASE_NAMES = {
    "inc_flag", "total_length_flag", "mover_is_white",
    "has_history_mover", "has_history_opponent",
}

####################
# FUNCTIONS
####################

# (a) REGISTRY-BACKED BUILDERS

def get_activation(name: str) -> nn.Module:
    """Returns an instantiated activation module by name."""
    if name not in ACTIVATION_REGISTRY:
        raise ValueError(f"Unknown activation '{name}', choose from {list(ACTIVATION_REGISTRY)}")
    return ACTIVATION_REGISTRY[name]()

def build_pool(pool_type: str, dim: int, dropout: float | None = None) -> nn.Module:
    """Builds a pooling module by name from POOL_REGISTRY; dropout must be None unless pool_type is 'attn'."""
    if pool_type not in POOL_REGISTRY:
        raise ValueError(f"Unknown pool_type '{pool_type}', choose from {list(POOL_REGISTRY)}")
    if pool_type == "mean" and dropout is not None:
        raise ValueError("pool_dropout was set but pool_type is 'mean', which has no dropout to apply.")
    if pool_type == "attn" and dropout is None:
        dropout = 0.1
    return POOL_REGISTRY[pool_type](dim, dropout)

def build_cnn_block(block_type: str, planes: int, kernel_size: int, activation: str) -> nn.Module:
    """Builds a CNN residual block by name from CNN_BLOCK_REGISTRY."""
    if block_type not in CNN_BLOCK_REGISTRY:
        raise ValueError(f"Unknown cnn_block_type '{block_type}', choose from {list(CNN_BLOCK_REGISTRY)}")
    return CNN_BLOCK_REGISTRY[block_type](planes, kernel_size, activation)

# (b) FEATURE TOKEN HELPERS

def resolve_active_features(features_dict: dict) -> list[str]:
    """Returns active '<base>_scaled'/'<base>_unscaled' keys from a features dict."""
    features_dict = features_dict or {}
    active = [name for name, on in features_dict.items() if on]

    bases_seen = {}
    for name in active:
        if name.endswith("_scaled"):
            base = name[: -len("_scaled")]
        elif name.endswith("_unscaled"):
            base = name[: -len("_unscaled")]
        else:
            raise ValueError(f"cfg.features key {name!r} must end in '_scaled' or '_unscaled'.")
        bases_seen.setdefault(base, []).append(name)

    collisions = {b: v for b, v in bases_seen.items() if len(v) > 1}
    if collisions:
        raise ValueError(f"cfg.features has both scaled and unscaled True for: {collisions} -- choose one.")

    return active


def _is_binary_feature(name: str) -> bool:
    """True if name's base (stripped of its _scaled/_unscaled suffix) is a binary feature."""
    return name.rsplit("_", 1)[0] in BINARY_BASE_NAMES


def build_feature_embeds(feature_names: list[str], dim_vit: int) -> nn.ModuleDict:
    """Returns one embedding module per feature: Embedding(2, dim_vit) for binaries, Linear(1, dim_vit) for continuous."""
    return nn.ModuleDict({
        name: (nn.Embedding(2, dim_vit) if _is_binary_feature(name) else nn.Linear(1, dim_vit))
        for name in feature_names
    })


def embed_feature(feature_embeds: nn.ModuleDict, name: str, val: torch.Tensor) -> torch.Tensor:
    """Embeds one feature column into a (b, 1, dim_vit) token."""
    if _is_binary_feature(name):
        return feature_embeds[name](val.long()).unsqueeze(1)
    return feature_embeds[name](val.unsqueeze(-1)).unsqueeze(1)

####################
# CLASSES
####################

# (a) CNN RES-NET BLOCKS

class BasicBlock(nn.Module):
    """Residual block: 2x (conv -> bn -> activation) plus a skip connection, Maia2-style.
    Out: (b, planes, h, w).
    """
    def __init__(self, planes, kernel_size=3, activation="relu"):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd for symmetric 'same' padding"
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(planes, planes, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.act   = get_activation(activation)

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + x
        return self.act(out)

CNN_BLOCK_REGISTRY = {
    "basic": BasicBlock,
}

class ChessCNN(nn.Module):
    """CNN trunk: board tensor to vit_length feature maps of shape (8, 8).
    Out: (b, vit_length, 8, 8).
    """
    def __init__(self, cfg):
        super().__init__()
        padding = cfg.kernel_size // 2
        self.conv1 = nn.Conv2d(cfg.input_channels, cfg.dim_cnn, kernel_size=cfg.kernel_size, padding=padding, bias=False)
        self.bn1   = nn.BatchNorm2d(cfg.dim_cnn)
        self.act   = get_activation(cfg.cnn_activation)
        self.blocks = nn.Sequential(*[
            build_cnn_block(cfg.cnn_block_type, cfg.dim_cnn, cfg.kernel_size, cfg.cnn_activation)
            for _ in range(cfg.num_blocks_cnn)
        ])
        self.conv_last = nn.Conv2d(cfg.dim_cnn, cfg.vit_length, kernel_size=cfg.kernel_size, padding=padding, bias=False)
        self.bn_last   = nn.BatchNorm2d(cfg.vit_length)

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.blocks(out)
        out = self.bn_last(self.conv_last(out))
        return out


# (b) ATTENTION

class Attention(nn.Module):
    """Multi-head self-attention, with an optional elo-derived offset added to the query.
    Out: (b, n, dim).
    """
    def __init__(self, dim, heads, dim_head, elo_dim=None, dropout=0.1):
        super().__init__()
        inner_dim = heads * dim_head

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.elo_query = nn.Linear(elo_dim, inner_dim, bias=False) if elo_dim is not None else None
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, elo_emb=None):
        b, n, _ = x.shape
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(b, n, self.heads, -1).transpose(1, 2) for t in qkv]

        if self.elo_query is not None:
            if elo_emb is None:
                raise ValueError("Attention was built with elo_dim set, but no elo_emb was passed.")
            elo_effect = self.elo_query(elo_emb).view(b, self.heads, 1, -1)
            q = q + elo_effect
        elif elo_emb is not None:
            raise ValueError("elo_emb was passed but this Attention has no elo_dim configured.")

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.dropout(self.attend(dots))
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)

class RotaryEmbedding(nn.Module):
    """Precomputes rotary (RoPE) cos/sin caches for q/k rotation.
    Out: (b, h, n, dim_head), same shape as input.
    """
    def __init__(self, dim, max_seq_len=128):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len, dtype=inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None], persistent=False)

    def forward(self, x):
        seq_len = x.shape[-2]
        cos = self.cos_cached[:, :, :seq_len]
        sin = self.sin_cached[:, :, :seq_len]
        x1, x2 = x.chunk(2, dim=-1)
        rotated = torch.cat([-x2, x1], dim=-1)
        return x * cos + rotated * sin

class RotaryAttention(nn.Module):
    """Multi-head self-attention; RoPE applied to the first n_rotary tokens, rest pass through unrotated.
    Out: (b, n, dim).
    """
    def __init__(self, dim, heads, dim_head, n_rotary, dropout=0.1):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head
        self.n_rotary = n_rotary
        self.dropout_p = dropout
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.rotary = RotaryEmbedding(dim_head, max_seq_len=n_rotary)

    def forward(self, x):
        b, n, _ = x.shape
        x_norm = self.norm(x)
        qkv = self.to_qkv(x_norm).chunk(3, dim=-1)
        q, k, v = [t.view(b, n, self.heads, self.dim_head).transpose(1, 2) for t in qkv]

        q = torch.cat([self.rotary(q[:, :, :self.n_rotary]), q[:, :, self.n_rotary:]], dim=2)
        k = torch.cat([self.rotary(k[:, :, :self.n_rotary]), k[:, :, self.n_rotary:]], dim=2)

        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout_p if self.training else 0.0
        )
        out = out.transpose(1, 2).reshape(b, n, self.heads * self.dim_head)
        return self.dropout(self.to_out(out))


# (c) NEURAL-NET FEEDFORWARD & POOLING

class FeedForward(nn.Module):
    """LayerNorm -> linear -> activation -> dropout -> linear -> dropout.
    Out: (b, n, dim).
    """
    def __init__(self, dim, hidden_dim, activation="gelu", dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class MeanPool(nn.Module):
    """Unweighted average over the token dimension.
    Out: (b, n, d) -> (b, d).
    """
    def __init__(self, dim=None, dropout=None):
        super().__init__()

    def forward(self, x):
        return x.mean(dim=1)

class AttentionPool(nn.Module):
    """Single learned query attends over all tokens to produce one pooled vector.
    Out: (b, n, d) -> (b, d).
    """
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim))
        self.scale = dim ** -0.5
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(dim, dim * 2, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b = x.size(0)
        q = self.to_q(self.query).expand(b, -1, -1)  # (b, 1, d)
        k, v = self.to_kv(x).chunk(2, dim=-1)         # each (b, n, d)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (b, 1, n)
        attn = self.dropout(self.attend(dots))
        out = torch.matmul(attn, v).squeeze(1)  # (b, d)
        return self.to_out(out)

POOL_REGISTRY = {
    "mean": lambda dim, dropout: MeanPool(dim, dropout),
    "attn": lambda dim, dropout: AttentionPool(dim, dropout=dropout),
}