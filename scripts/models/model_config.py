"""
model_config.py
Allows for configuration of model's parameters, depends on flexibility allowed in model_arch.py

Latest changes: 10/08/26:
- Dropout split into per-component fields, cnn_block_type added, defaults rescaled to one shared small preset
"""

from dataclasses import dataclass, field, asdict

####################
# FUNCTIONS
####################

####################
# CLASSES
####################

@dataclass
class BaseModelConfig:
    """Base config every architecture's config subclasses."""
    arch_name: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

# (a) CNN/ATTENTION BASED MODELS - CONFIG

@dataclass
class Maia2ValueBoardConfig(BaseModelConfig):
    """Config for Maia2ValueBoard: plain-attention, elo-blind CNN+attention value model."""
    arch_name: str = "maia2_value_board"

    input_channels: int = 18
    dim_cnn: int = 64
    num_blocks_cnn: int = 1
    cnn_block_type: str = "basic"
    kernel_size: int = 3
    cnn_activation: str = "relu"

    vit_length: int = 8
    dim_vit: int = 64
    embed_dropout: float = 0.1

    n_vit_layer: int = 1
    heads: int = 4
    dim_head: int = 16
    ff_hidden_dim: int | None = None
    ff_activation: str = "gelu"
    attn_dropout: float = 0.1
    ff_dropout: float = 0.1

    pool_type: str = "mean"
    pool_dropout: float | None = None

    value_hidden_dim: int = 64
    value_activation: str = "relu"
    value_dropout: float = 0.1

@dataclass
class Maia2ValueReplicaConfig(BaseModelConfig):
    """Config for Maia2ValueReplica: non-exact replica of Maia2, value head only."""
    arch_name: str = "maia2_value_replica"

    input_channels: int = 18
    dim_cnn: int = 64
    num_blocks_cnn: int = 1
    cnn_block_type: str = "basic"
    kernel_size: int = 3
    cnn_activation: str = "relu"

    vit_length: int = 8
    dim_vit: int = 64
    embed_dropout: float = 0.1

    n_vit_layer: int = 1
    heads: int = 4
    dim_head: int = 16
    elo_dim: int = 32
    ff_hidden_dim: int | None = None
    ff_activation: str = "gelu"
    attn_dropout: float = 0.1
    ff_dropout: float = 0.1

    pool_type: str = "mean"
    pool_dropout: float | None = None

    value_hidden_dim: int = 64
    value_activation: str = "relu"
    value_dropout: float = 0.1

@dataclass
class Maia2ValueFeatureConfig(BaseModelConfig):
    """Config for Maia2ValueFeature: Maia2ValueReplica's trunk with features as additional tokens."""
    arch_name: str = "maia2_value_feature"

    input_channels: int = 18
    dim_cnn: int = 64
    num_blocks_cnn: int = 1
    cnn_block_type: str = "basic"
    kernel_size: int = 3
    cnn_activation: str = "relu"

    vit_length: int = 8
    dim_vit: int = 64
    board_embed_dropout: float = 0.1
    aux_embed_dropout: float = 0.1

    features: dict = field(default_factory=dict)

    n_vit_layer: int = 1
    heads: int = 4
    dim_head: int = 16
    ff_hidden_dim: int | None = None
    ff_activation: str = "gelu"
    attn_dropout: float = 0.1
    ff_dropout: float = 0.1

    pool_type: str = "mean"
    pool_dropout: float | None = None

    value_hidden_dim: int = 64
    value_activation: str = "relu"
    value_dropout: float = 0.1


# (b) PURE-TRANSFORMER MODELS - CONFIG

@dataclass
class PureTransformerConfig(BaseModelConfig):
    """Config for PureTransformer: FEN-token-sequence transformer, rope or learned position embedding."""
    arch_name: str = "pure_transformer"

    dim_vit: int = 64
    board_embed_dropout: float = 0.1
    aux_embed_dropout: float = 0.1

    features: dict = field(default_factory=dict)
    pos_embed_type: str = "rope"

    n_vit_layer: int = 1
    heads: int = 4
    dim_head: int = 16
    ff_hidden_dim: int | None = None
    ff_activation: str = "gelu"
    attn_dropout: float = 0.1
    ff_dropout: float = 0.1

    pool_type: str = "mean"
    pool_dropout: float | None = None

    value_hidden_dim: int = 64
    value_activation: str = "relu"
    value_dropout: float = 0.1