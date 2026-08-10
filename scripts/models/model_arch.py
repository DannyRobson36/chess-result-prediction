"""
model_arch.py
Contains all models using PyTorch

Latest changes: 10/08/26:
- Prediction registries created for compatible training/inference with different models
"""

import torch
import torch.nn as nn
from collections.abc import Callable

from scripts.utils.utils_chess import FEN_VOCAB, BOARD_SEQ_LEN

from scripts.models.model_support import (
    ChessCNN, Attention,
    RotaryAttention, FeedForward,
    build_pool, get_activation, resolve_active_features,
    build_feature_embeds, embed_feature
)

from scripts.models.model_config import (
    BaseModelConfig,
    Maia2ValueBoardConfig, Maia2ValueFeatureConfig,
    Maia2ValueReplicaConfig, PureTransformerConfig
)

####################
# CONSTANTS
####################

OUTPUT_TYPE_REGISTRY = {
    "maia2_value_board": "regression",
    "maia2_value_replica": "regression",
    "maia2_value_feature": "classification",
    "pure_transformer": "classification",
}

####################
# FUNCTIONS
####################

def get_output_type(arch_name: str) -> str:
    """Returns 'regression' or 'classification' for arch_name."""
    if arch_name not in OUTPUT_TYPE_REGISTRY:
        raise ValueError(f"Unknown arch_name '{arch_name}', choose from {list(OUTPUT_TYPE_REGISTRY)}")
    return OUTPUT_TYPE_REGISTRY[arch_name]

def get_model_kwargs(arch_name: str, n_elo_bins: int | None = None, two_way: bool | None = None) -> dict:
    """Returns the extra constructor kwargs (beyond cfg) that arch_name needs."""
    if arch_name == "maia2_value_replica":
        if n_elo_bins is None:
            raise ValueError("maia2_value_replica requires n_elo_bins.")
        return {"n_elo_bins": n_elo_bins}
    if arch_name in ("maia2_value_feature", "pure_transformer"):
        if two_way is None:
            raise ValueError(f"{arch_name} requires two_way.")
        return {"n_result_classes": 2 if two_way else 3}
    return {}

####################
# CLASSES
####################

# (a) CNN/ATTENTION BASED MODELS

class Maia2ValueBoard(nn.Module):
    """Elo-unaware ablation of Maia2ValueReplica."""
    def __init__(self, cfg: Maia2ValueBoardConfig):
        super().__init__()
        self.cfg = cfg
        self.cnn = ChessCNN(cfg)
        self.to_patch_embedding = nn.Sequential(
            nn.Linear(8 * 8, cfg.dim_vit),
            nn.LayerNorm(cfg.dim_vit),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, cfg.vit_length, cfg.dim_vit))
        self.embed_dropout = nn.Dropout(cfg.embed_dropout)

        ff_hidden_dim = cfg.ff_hidden_dim if cfg.ff_hidden_dim is not None else cfg.dim_vit
        self.attn_blocks = nn.ModuleList([
            nn.ModuleList([
                Attention(cfg.dim_vit, cfg.heads, cfg.dim_head, dropout=cfg.attn_dropout),
                FeedForward(cfg.dim_vit, ff_hidden_dim, cfg.ff_activation, cfg.ff_dropout),
            ])
            for _ in range(cfg.n_vit_layer)
        ])
        self.norm = nn.LayerNorm(cfg.dim_vit)
        self.pool = build_pool(cfg.pool_type, cfg.dim_vit, cfg.pool_dropout)
        self.last_ln = nn.LayerNorm(cfg.dim_vit)

        self.value_hidden = nn.Linear(cfg.dim_vit, cfg.value_hidden_dim)
        self.value_act = get_activation(cfg.value_activation)
        self.value_dropout = nn.Dropout(cfg.value_dropout)
        self.value_out = nn.Linear(cfg.value_hidden_dim, 1)

    def forward(self, boards):
        b = boards.size(0)
        feats = self.cnn(boards)
        feats = feats.view(b, feats.size(1), 8 * 8)
        x = self.to_patch_embedding(feats)
        x = x + self.pos_embedding
        x = self.embed_dropout(x)

        for attn, ff in self.attn_blocks:
            x = attn(x) + x
            x = ff(x) + x
        x = self.pool(self.norm(x))
        x = self.last_ln(x)

        value_hidden = self.value_dropout(self.value_act(self.value_hidden(x)))
        value_pred = self.value_out(value_hidden).squeeze(-1)
        return value_pred


class Maia2ValueReplica(nn.Module):
    """Non-exact replica of Maia2, value head only."""
    def __init__(self, cfg: Maia2ValueReplicaConfig, n_elo_bins: int):
        super().__init__()
        self.cfg = cfg
        self.cnn = ChessCNN(cfg)
        self.to_patch_embedding = nn.Sequential(
            nn.Linear(8 * 8, cfg.dim_vit),
            nn.LayerNorm(cfg.dim_vit),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, cfg.vit_length, cfg.dim_vit))
        self.embed_dropout = nn.Dropout(cfg.embed_dropout)

        self.elo_embedding = nn.Embedding(n_elo_bins, cfg.elo_dim)
        ff_hidden_dim = cfg.ff_hidden_dim if cfg.ff_hidden_dim is not None else cfg.dim_vit
        self.attn_blocks = nn.ModuleList([
            nn.ModuleList([
                Attention(cfg.dim_vit, cfg.heads, cfg.dim_head, elo_dim=cfg.elo_dim * 2, dropout=cfg.attn_dropout),
                FeedForward(cfg.dim_vit, ff_hidden_dim, cfg.ff_activation, cfg.ff_dropout),
            ])
            for _ in range(cfg.n_vit_layer)
        ])
        self.norm = nn.LayerNorm(cfg.dim_vit)
        self.pool = build_pool(cfg.pool_type, cfg.dim_vit, cfg.pool_dropout)
        self.last_ln = nn.LayerNorm(cfg.dim_vit)

        self.value_hidden = nn.Linear(cfg.dim_vit, cfg.value_hidden_dim)
        self.value_act = get_activation(cfg.value_activation)
        self.value_dropout = nn.Dropout(cfg.value_dropout)
        self.value_out = nn.Linear(cfg.value_hidden_dim, 1)

    def forward(self, boards, elo_self_bin, elo_oppo_bin):
        b = boards.size(0)
        feats = self.cnn(boards)
        feats = feats.view(b, feats.size(1), 8 * 8)
        x = self.to_patch_embedding(feats)
        x = x + self.pos_embedding
        x = self.embed_dropout(x)

        elo_emb = torch.cat([self.elo_embedding(elo_self_bin), self.elo_embedding(elo_oppo_bin)], dim=1)

        for attn, ff in self.attn_blocks:
            x = attn(x, elo_emb) + x
            x = ff(x) + x
        x = self.pool(self.norm(x))
        x = self.last_ln(x)

        value_hidden = self.value_dropout(self.value_act(self.value_hidden(x)))
        value_pred = self.value_out(value_hidden).squeeze(-1)
        return value_pred


class Maia2ValueFeature(nn.Module):
    """Maia2ValueReplica's trunk with features as additional attention tokens."""
    def __init__(self, cfg: Maia2ValueFeatureConfig, n_result_classes: int = 3):
        super().__init__()
        if n_result_classes not in (2, 3):
            raise ValueError(f"n_result_classes must be 2 or 3, got {n_result_classes}.")

        self.cfg = cfg
        self.feature_names = resolve_active_features(cfg.features)
        self.cnn = ChessCNN(cfg)
        self.to_patch_embedding = nn.Sequential(
            nn.Linear(8 * 8, cfg.dim_vit),
            nn.LayerNorm(cfg.dim_vit),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, cfg.vit_length, cfg.dim_vit))
        self.board_embed_dropout = nn.Dropout(cfg.board_embed_dropout)
        self.aux_embed_dropout = nn.Dropout(cfg.aux_embed_dropout)

        self.feature_embeds = build_feature_embeds(self.feature_names, cfg.dim_vit)
        self.feature_pos = nn.Parameter(torch.randn(1, len(self.feature_names), cfg.dim_vit))

        ff_hidden_dim = cfg.ff_hidden_dim if cfg.ff_hidden_dim is not None else cfg.dim_vit
        self.attn_blocks = nn.ModuleList([
            nn.ModuleList([
                Attention(cfg.dim_vit, cfg.heads, cfg.dim_head, dropout=cfg.attn_dropout),
                FeedForward(cfg.dim_vit, ff_hidden_dim, cfg.ff_activation, cfg.ff_dropout),
            ])
            for _ in range(cfg.n_vit_layer)
        ])
        self.norm = nn.LayerNorm(cfg.dim_vit)
        self.pool = build_pool(cfg.pool_type, cfg.dim_vit, cfg.pool_dropout)
        self.last_ln = nn.LayerNorm(cfg.dim_vit)

        self.value_hidden = nn.Linear(cfg.dim_vit, cfg.value_hidden_dim)
        self.value_act = get_activation(cfg.value_activation)
        self.value_dropout = nn.Dropout(cfg.value_dropout)
        self.value_out = nn.Linear(cfg.value_hidden_dim, n_result_classes)

    def forward(self, boards, features: dict):
        b = boards.size(0)
        feats = self.cnn(boards)
        feats = feats.view(b, feats.size(1), 8 * 8)
        board_tokens = self.to_patch_embedding(feats)
        board_tokens = board_tokens + self.pos_embedding
        board_tokens = self.board_embed_dropout(board_tokens)

        feature_tokens = torch.cat([
            embed_feature(self.feature_embeds, name, features[name])
            for name in self.feature_names
        ], dim=1)
        feature_tokens = feature_tokens + self.feature_pos
        feature_tokens = self.aux_embed_dropout(feature_tokens)

        x = torch.cat([board_tokens, feature_tokens], dim=1)

        for attn, ff in self.attn_blocks:
            x = attn(x) + x
            x = ff(x) + x
        x = self.pool(self.norm(x))
        x = self.last_ln(x)

        value_hidden = self.value_dropout(self.value_act(self.value_hidden(x)))
        value_logits = self.value_out(value_hidden)
        return value_logits


# (b) PURE-TRANSFORMER MODELS

class PureTransformer(nn.Module):
    """
    FEN-token transformer: 64 board squares, 12 FEN metadata chars, and feature tokens
    Each get separate positional/identity treatment, then joint self-attention -> pool -> MLP head.
    """
    N_META = BOARD_SEQ_LEN - 64 

    def __init__(self, cfg: PureTransformerConfig, n_result_classes: int = 3):
        super().__init__()
        if n_result_classes not in (2, 3):
            raise ValueError(f"n_result_classes must be 2 or 3, got {n_result_classes}.")
        if cfg.pos_embed_type not in ("rope", "learned"):
            raise ValueError(f"pos_embed_type must be 'rope' or 'learned', got {cfg.pos_embed_type!r}")

        self.cfg = cfg
        self.feature_names = resolve_active_features(cfg.features)

        self.token_embedding = nn.Embedding(len(FEN_VOCAB), cfg.dim_vit)
        self.board_embed_dropout = nn.Dropout(cfg.board_embed_dropout)
        self.aux_embed_dropout = nn.Dropout(cfg.aux_embed_dropout)

        self.feature_embeds = build_feature_embeds(self.feature_names, cfg.dim_vit)

        self.meta_pos = nn.Parameter(torch.randn(1, self.N_META, cfg.dim_vit))
        self.feature_pos = nn.Parameter(torch.randn(1, len(self.feature_names), cfg.dim_vit))
        if cfg.pos_embed_type == "learned":
            self.board_pos_embedding = nn.Parameter(torch.randn(1, 64, cfg.dim_vit))

        ff_hidden_dim = cfg.ff_hidden_dim if cfg.ff_hidden_dim is not None else cfg.dim_vit
        if cfg.pos_embed_type == "learned":
            self.blocks = nn.ModuleList([
                nn.ModuleList([
                    Attention(cfg.dim_vit, cfg.heads, cfg.dim_head, dropout=cfg.attn_dropout),
                    FeedForward(cfg.dim_vit, ff_hidden_dim, cfg.ff_activation, cfg.ff_dropout),
                ])
                for _ in range(cfg.n_vit_layer)
            ])
        else:
            self.blocks = nn.ModuleList([
                nn.ModuleList([
                    RotaryAttention(cfg.dim_vit, cfg.heads, cfg.dim_head, n_rotary=64, dropout=cfg.attn_dropout),
                    FeedForward(cfg.dim_vit, ff_hidden_dim, cfg.ff_activation, cfg.ff_dropout),
                ])
                for _ in range(cfg.n_vit_layer)
            ])

        self.norm = nn.LayerNorm(cfg.dim_vit)
        self.pool = build_pool(cfg.pool_type, cfg.dim_vit, cfg.pool_dropout)
        self.last_ln = nn.LayerNorm(cfg.dim_vit)

        self.value_hidden = nn.Linear(cfg.dim_vit, cfg.value_hidden_dim)
        self.value_act = get_activation(cfg.value_activation)
        self.value_dropout = nn.Dropout(cfg.value_dropout)
        self.value_out = nn.Linear(cfg.value_hidden_dim, n_result_classes)

    def forward(self, board_token_ids, features: dict):
        x = self.token_embedding(board_token_ids)  
        board_tok, meta_tok = x[:, :64], x[:, 64:]

        if self.cfg.pos_embed_type == "learned":
            board_tok = board_tok + self.board_pos_embedding
        board_tok = self.board_embed_dropout(board_tok)

        meta_tok = meta_tok + self.meta_pos
        meta_tok = self.aux_embed_dropout(meta_tok)

        feature_tokens = torch.cat([
            embed_feature(self.feature_embeds, name, features[name])
            for name in self.feature_names
        ], dim=1)
        feature_tokens = feature_tokens + self.feature_pos
        feature_tokens = self.aux_embed_dropout(feature_tokens)

        x = torch.cat([board_tok, meta_tok, feature_tokens], dim=1)

        for attn, ff in self.blocks:
            x = attn(x) + x
            x = ff(x) + x
        x = self.pool(self.norm(x))
        x = self.last_ln(x)

        value_hidden = self.value_dropout(self.value_act(self.value_hidden(x)))
        value_logits = self.value_out(value_hidden)
        return value_logits


####################
# EXTRAS
####################

# (a) MODEL BUILDING

MODEL_REGISTRY: dict[str, tuple[type[nn.Module], type[BaseModelConfig]]] = {
    "maia2_value_board": (Maia2ValueBoard, Maia2ValueBoardConfig),
    "maia2_value_replica": (Maia2ValueReplica, Maia2ValueReplicaConfig),
    "maia2_value_feature": (Maia2ValueFeature, Maia2ValueFeatureConfig),
    "pure_transformer": (PureTransformer, PureTransformerConfig),
}

def _validate_model_registry() -> None:
    """Raises if a MODEL_REGISTRY config's default arch_name mismatches its key, or is missing from OUTPUT_TYPE_REGISTRY."""
    for key, (_, cfg_cls) in MODEL_REGISTRY.items():
        default_arch_name = cfg_cls().arch_name
        if default_arch_name != key:
            raise ValueError(f"MODEL_REGISTRY[{key!r}] config class {cfg_cls.__name__} has "
                              f"default arch_name={default_arch_name!r}, which doesn't match its registry key.")
        if key not in OUTPUT_TYPE_REGISTRY:
            raise ValueError(f"MODEL_REGISTRY[{key!r}] has no entry in OUTPUT_TYPE_REGISTRY.")

_validate_model_registry()

def build_model(arch_name: str, cfg: BaseModelConfig,
                 n_elo_bins: int | None = None, two_way: bool | None = None) -> nn.Module:
    """Builds a model by arch_name from MODEL_REGISTRY, using cfg plus any data-derived kwargs it needs."""
    if arch_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown arch_name '{arch_name}', choose from {list(MODEL_REGISTRY)}")
    model_cls, cfg_cls = MODEL_REGISTRY[arch_name]

    if not isinstance(cfg, cfg_cls):
        raise TypeError(f"build_model('{arch_name}', ...) expects a {cfg_cls.__name__}, got {type(cfg).__name__}.")
    if cfg.arch_name != arch_name:
        raise ValueError(f"cfg.arch_name is '{cfg.arch_name}' but build_model was called with arch_name='{arch_name}'.")

    extra_kwargs = get_model_kwargs(arch_name, n_elo_bins, two_way)
    return model_cls(cfg, **extra_kwargs)

# (b) MODEL INFERENCE

def predict_maia2_value_board(model: nn.Module, batch: dict) -> torch.Tensor:
    """Runs Maia2ValueBoard on a batch dict, returning its predictions."""
    return model(batch["boards"])

def predict_maia2_value_replica(model: nn.Module, batch: dict) -> torch.Tensor:
    """Runs Maia2ValueReplica on a batch dict, returning its predictions."""
    return model(batch["boards"], batch["elo_self_bin"], batch["elo_oppo_bin"])


def _batch_features(model: nn.Module, batch: dict) -> dict:
    """Pulls just model.feature_names out of a flat batch dict, raising if any are missing."""
    missing = [name for name in model.feature_names if name not in batch]
    if missing:
        raise KeyError(f"cfg.features asks for {missing}, which the batch does not contain.")
    return {name: batch[name] for name in model.feature_names}

def predict_maia2_value_feature(model: nn.Module, batch: dict) -> torch.Tensor:
    """Runs Maia2ValueFeature on a batch dict, returning its predictions."""
    return model(batch["boards"], _batch_features(model, batch))

def predict_pure_transformer(model: nn.Module, batch: dict) -> torch.Tensor:
    """Runs PureTransformer on a batch dict, returning its predictions."""
    return model(batch["board_token_ids"], _batch_features(model, batch))


PREDICT_REGISTRY: dict[str, Callable[[nn.Module, dict], torch.Tensor]] = {
    "maia2_value_board": predict_maia2_value_board,
    "maia2_value_replica": predict_maia2_value_replica,
    "maia2_value_feature": predict_maia2_value_feature,
    "pure_transformer": predict_pure_transformer,
}

def get_predict_fn(arch_name: str) -> Callable[[nn.Module, dict], torch.Tensor]:
    """Returns the predict(model, batch) function for arch_name."""
    if arch_name not in PREDICT_REGISTRY:
        raise ValueError(f"Unknown arch_name '{arch_name}', choose from {list(PREDICT_REGISTRY)}")
    return PREDICT_REGISTRY[arch_name]