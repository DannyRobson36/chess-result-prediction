"""
model_config.py
Per-architecture model config dataclasses.

Latest changes: 27/09/26:
- Tightening docstring
"""

from dataclasses import dataclass, field, asdict

####################
# CLASSES
####################

@dataclass
class BaseModelConfig:
    """Base config every architecture's config subclasses."""
    arch_name: str = ""

    def to_dict(self) -> dict:
        """Returns config as dict."""
        return asdict(self)

@dataclass
class AuxHeadConfig:
    """Aux head on/off flag and loss weight."""
    enabled: bool = False
    loss_weight: float | None = None

# (a) LOGISTIC REGRESSION (BASELINE) - CONFIG

@dataclass
class LogRegBaselineConfig(BaseModelConfig):
    """Config for LogRegBaseline."""
    arch_name: str = "log_reg_baseline"

    features: dict = field(default_factory=dict)

# (b) CNN/ATTENTION BASED MODELS - CONFIG

@dataclass
class Maia2ValueBoardConfig(BaseModelConfig):
    """Config for Maia2ValueBoard."""
    arch_name: str = "maia2_value_board"

    input_channels: int = 18
    dim_cnn: int = 64
    num_blocks_cnn: int = 1
    cnn_block_type: str = "basic"
    kernel_size: int = 3
    cnn_activation: str = "relu"

    vit_length: int = 8
    dim_vit: int = 64
    board_embed_dropout: float = 0.1

    n_vit_layer: int = 1
    heads: int = 4
    dim_head: int = 16
    ff_hidden_dim: int | None = None
    ff_activation: str = "gelu"
    trunk_dropout: float = 0.1

    pool_type: str = "mean"
    pool_dropout: float | None = None

    value_hidden_dim: int = 64
    value_activation: str = "relu"
    value_dropout: float = 0.1

    aux_head: AuxHeadConfig = field(default_factory=AuxHeadConfig)

@dataclass
class Maia2ValueReplicaConfig(BaseModelConfig):
    """Config for Maia2ValueReplica."""
    arch_name: str = "maia2_value_replica"

    input_channels: int = 18
    dim_cnn: int = 64
    num_blocks_cnn: int = 1
    cnn_block_type: str = "basic"
    kernel_size: int = 3
    cnn_activation: str = "relu"

    vit_length: int = 8
    dim_vit: int = 64
    board_embed_dropout: float = 0.1

    n_vit_layer: int = 1
    heads: int = 4
    dim_head: int = 16
    elo_dim: int = 32
    ff_hidden_dim: int | None = None
    ff_activation: str = "gelu"
    trunk_dropout: float = 0.1

    pool_type: str = "mean"
    pool_dropout: float | None = None

    value_hidden_dim: int = 64
    value_activation: str = "relu"
    value_dropout: float = 0.1

    aux_head: AuxHeadConfig = field(default_factory=AuxHeadConfig)

@dataclass
class Maia2ValueFeatureConfig(BaseModelConfig):
    """Config for Maia2ValueFeature."""
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
    feat_embed_dropout: float = 0.1

    features: dict = field(default_factory=dict)

    n_vit_layer: int = 1
    heads: int = 4
    dim_head: int = 16
    ff_hidden_dim: int | None = None
    ff_activation: str = "gelu"
    trunk_dropout: float = 0.1

    pool_type: str = "mean"
    pool_dropout: float | None = None

    value_hidden_dim: int = 64
    value_activation: str = "relu"
    value_dropout: float = 0.1

    aux_head: AuxHeadConfig = field(default_factory=AuxHeadConfig)


# (c) PURE-TRANSFORMER MODELS - CONFIG

@dataclass
class PureTransformerConfig(BaseModelConfig):
    """Config for PureTransformer."""
    arch_name: str = "pure_transformer"

    dim_vit: int = 64
    board_embed_dropout: float = 0.1
    feat_embed_dropout: float = 0.1

    features: dict = field(default_factory=dict)
    pos_embed_type: str = "rope"

    n_vit_layer: int = 1
    heads: int = 4
    dim_head: int = 16
    ff_hidden_dim: int | None = None
    ff_activation: str = "gelu"
    trunk_dropout: float = 0.1

    pool_type: str = "mean"
    pool_dropout: float | None = None

    value_hidden_dim: int = 64
    value_activation: str = "relu"
    value_dropout: float = 0.1

    aux_head: AuxHeadConfig = field(default_factory=AuxHeadConfig)