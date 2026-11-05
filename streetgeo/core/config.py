"""Configuration shared by training, inference and saved checkpoints."""

from dataclasses import dataclass
import math


LEGACY_BACKBONE = "svgt_vit"
SIGLIP2_BACKBONE = "siglip2_base_patch16_224"
SIGLIP2_REPO = "google/siglip2-base-patch16-224"
SIGLIP2_REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
LEGACY_PREPROCESSING = "svgt_letterbox_v1"
SIGLIP2_PREPROCESSING = "siglip2_resize_v1"
SIGLIP2_CONFIG_PATH = "configs/streetgeo_siglip2_nmpz.json"
SIGLIP2_MODEL_PATH = "checkpoints/streetgeo_siglip2_nmpz/best.pt"
SIGLIP2_WEIGHTS_PATH = "checkpoints/pretrained/siglip2-base-patch16-224/model.safetensors"


@dataclass
class ModelConfig:
    image_size: int = 224
    patch_size: int = 16
    n_embed: int = 1024
    n_head: int = 16
    n_layer: int = 28
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
    vision_backbone: str = LEGACY_BACKBONE
    backbone_revision: str | None = None
    preprocessing: str | None = None
    geo_smoothing: float = 0.0
    geo_smoothing_km: float = 250.0

    def __post_init__(self):
        integer_fields = (
            "image_size", "patch_size", "n_embed", "n_head", "n_layer",
            "view_queries", "max_views", "mlp_ratio",
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
        if not 0 <= self.geo_smoothing <= 0.5:
            raise ValueError("geo_smoothing must be in [0, 0.5]")
        if not math.isfinite(self.geo_smoothing_km) or self.geo_smoothing_km <= 0:
            raise ValueError("geo_smoothing_km must be positive and finite")
        if type(self.fusion_layers) is not int:
            raise ValueError("fusion_layers must be an integer")
        if self.vision_backbone == SIGLIP2_BACKBONE:
            shape = (self.image_size, self.patch_size, self.n_embed, self.n_head, self.n_layer,
                     self.fusion_layers, self.view_queries, self.max_views, self.mlp_ratio)
            if shape != (224, 16, 768, 12, 12, 0, 4, 1, 4):
                raise ValueError("SigLIP2 NMPZ requires Base/16/224, four queries and one view without fusion")
            if self.backbone_revision not in (None, SIGLIP2_REVISION):
                raise ValueError("Unsupported SigLIP2 backbone revision")
            self.backbone_revision = SIGLIP2_REVISION
            expected_preprocessing = SIGLIP2_PREPROCESSING
        elif self.vision_backbone == LEGACY_BACKBONE:
            if self.fusion_layers <= 0 or self.backbone_revision is not None:
                raise ValueError("The original visual encoder requires fusion layers and no backbone revision")
            expected_preprocessing = LEGACY_PREPROCESSING
        else:
            raise ValueError(f"Unsupported vision backbone: {self.vision_backbone}")
        if self.preprocessing not in (None, expected_preprocessing):
            raise ValueError("Preprocessing does not match the vision backbone")
        self.preprocessing = expected_preprocessing

    @property
    def is_siglip2(self):
        return self.vision_backbone == SIGLIP2_BACKBONE

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
    batch_size: int = 32
    epochs: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_steps: int = 1000
    num_workers: int = 4
    device: str = "auto"
    seed: int = 1337
    model_path: str = "checkpoints/streetgeo_400m_mapillary/best.pt"
    save_every_steps: int = 500
    save_every_seconds: float = 300.0
    log_every_steps: int = 10
