"""Configuration shared by training, inference and saved checkpoints."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    image_size: int = 224
    patch_size: int = 16
    n_embed: int = 768
    n_head: int = 12
    n_layer: int = 15
    fusion_layers: int = 2
    view_queries: int = 4
    max_views: int = 4
    mlp_ratio: int = 4
    dropout: float = 0.1
    coarse_lat_bins: int = 18
    coarse_lon_bins: int = 36
    fine_bins: int = 4
    cell_embed_dim: int = 128
    offset_loss_weight: float = 1.0
    spherical_loss_weight: float = 1.0

    def __post_init__(self):
        integer_fields = (
            "image_size", "patch_size", "n_embed", "n_head", "n_layer",
            "fusion_layers", "view_queries", "max_views", "mlp_ratio",
            "coarse_lat_bins", "coarse_lon_bins", "fine_bins", "cell_embed_dim",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.image_size % self.patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        if self.n_embed % self.n_head:
            raise ValueError("n_embed must be divisible by n_head")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.offset_loss_weight < 0 or self.spherical_loss_weight < 0:
            raise ValueError("loss weights cannot be negative")

    @property
    def coarse_cells(self):
        return self.coarse_lat_bins * self.coarse_lon_bins

    @property
    def children_per_cell(self):
        return self.fine_bins ** 2

    @property
    def total_cells(self):
        return self.coarse_cells * self.children_per_cell


@dataclass
class TrainConfig:
    batch_size: int = 64
    epochs: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_steps: int = 1000
    num_workers: int = 4
    device: str = "auto"
    seed: int = 1337
    model_path: str = "checkpoints/streetgeo_514m/best.pt"
    save_every_steps: int = 500
    save_every_seconds: float = 300.0
    log_every_steps: int = 10
