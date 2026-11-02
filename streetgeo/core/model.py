"""Street-view images -> hierarchical location hypotheses -> numeric coordinates.

Images must be preprocessed with streetgeo.datasets.images.load_image; optional headings are degrees
clockwise from north. This module never downloads weights or training data.
"""

from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F

from streetgeo.core import EmptyPredictionDomainError
from streetgeo.core.config import ModelConfig
from streetgeo.core.geography import HierarchicalGeoGrid, LocationPrediction, coordinates_to_unit_vectors


class MultiHeadAttention(nn.Module):
    """Bidirectional self-attention; True in valid_mask means a real token."""

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.head_size = config.n_embed // config.n_head
        self.qkv = nn.Linear(config.n_embed, 3 * config.n_embed)
        self.proj = nn.Linear(config.n_embed, config.n_embed)
        self.dropout = config.dropout

    def forward(self, x, valid_mask=None):
        batch, length, width = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.n_head, self.head_size)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        mask = None if valid_mask is None else valid_mask[:, None, None, :]
        attended = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=False,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(batch, length, width)
        return F.dropout(self.proj(attended), self.dropout, self.training)


class Block(nn.Module):
    """Preserve the original pre-norm Transformer residual structure."""

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embed)
        self.attention = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embed)
        self.ffn = nn.Sequential(
            nn.Linear(config.n_embed, config.mlp_ratio * config.n_embed), nn.GELU(),
            nn.Linear(config.mlp_ratio * config.n_embed, config.n_embed),
            nn.Dropout(config.dropout),
        )

    def forward(self, x, valid_mask=None):
        x = x + self.attention(self.ln1(x), valid_mask)
        return x + self.ffn(self.ln2(x))


class StreetViewEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        patches = (config.image_size // config.patch_size) ** 2
        self.patch_embedding = nn.Conv2d(
            3, config.n_embed, kernel_size=config.patch_size, stride=config.patch_size,
        )
        self.position_embedding = nn.Parameter(torch.zeros(1, patches, config.n_embed))
        self.visual_blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.visual_norm = nn.LayerNorm(config.n_embed)
        self.pool_queries = nn.Parameter(torch.zeros(1, config.view_queries, config.n_embed))
        self.pool_attention = nn.MultiheadAttention(
            config.n_embed, config.n_head, dropout=config.dropout, batch_first=True,
        )
        # No view-index embedding: unordered views remain permutation invariant.
        self.heading_embedding = nn.Sequential(nn.Linear(3, config.n_embed), nn.GELU(),
                                               nn.Linear(config.n_embed, config.n_embed))
        self.geo_token = nn.Parameter(torch.zeros(1, 1, config.n_embed))
        self.fusion_blocks = nn.ModuleList([Block(config) for _ in range(config.fusion_layers)])
        self.fusion_norm = nn.LayerNorm(config.n_embed)

    def forward(self, images, view_mask=None, headings=None):
        if images.ndim == 4:
            images = images.unsqueeze(1)
        size = self.config.image_size
        if images.ndim != 5 or tuple(images.shape[2:]) != (3, size, size):
            raise ValueError(f"images must have shape (B, 3, {size}, {size}) or (B, V, 3, {size}, {size})")
        batch, views = images.shape[:2]
        if batch == 0 or not 1 <= views <= self.config.max_views:
            raise ValueError(f"nonempty batch and 1..{self.config.max_views} views required")
        if not images.is_floating_point():
            raise ValueError("images must be normalized floating-point RGB tensors")
        if view_mask is None:
            view_mask = torch.ones(batch, views, dtype=torch.bool, device=images.device)
        elif view_mask.shape != (batch, views) or view_mask.dtype != torch.bool:
            raise ValueError("view_mask must be boolean with shape (B, V)")
        view_mask = view_mask.to(images.device)
        if not view_mask.any(dim=1).all():
            raise ValueError("every sample needs at least one valid view")
        # Only valid views are encoded, so padding cannot contaminate gradients.
        valid_images = images[view_mask]
        if not torch.isfinite(valid_images).all():
            raise ValueError("valid images must contain finite values")
        x = self.patch_embedding(valid_images).flatten(2).transpose(1, 2)
        x = x + self.position_embedding
        for block in self.visual_blocks:
            x = block(x)
        x = self.visual_norm(x)
        queries = self.pool_queries.expand(x.shape[0], -1, -1)
        pooled, _ = self.pool_attention(queries, x, x, need_weights=False)
        pooled = pooled + queries
        tokens = pooled.new_zeros(batch, views, self.config.view_queries, self.config.n_embed)
        tokens[view_mask] = pooled
        if headings is None:
            headings = torch.full((batch, views), float("nan"), device=images.device)
        elif headings.shape != (batch, views):
            raise ValueError("headings must have shape (B, V); NaN means unknown")
        headings = headings.to(device=images.device, dtype=torch.float32)
        if torch.isinf(headings[view_mask]).any():
            raise ValueError("headings must be finite or NaN (unknown)")
        known = torch.isfinite(headings) & view_mask
        angle = torch.deg2rad(torch.where(known, headings, 0.0).remainder(360))
        features = torch.stack((angle.sin() * known, angle.cos() * known, known.float()), -1)
        tokens = tokens + self.heading_embedding(features).unsqueeze(2)
        tokens = tokens.flatten(1, 2)
        x = torch.cat((self.geo_token.expand(batch, -1, -1), tokens), dim=1)
        valid = torch.cat((torch.ones(batch, 1, dtype=torch.bool, device=images.device),
                           view_mask.repeat_interleave(self.config.view_queries, dim=1)), dim=1)
        for block in self.fusion_blocks:
            x = block(x, valid)
        return self.fusion_norm(x[:, 0])


@dataclass
class GeolocationOutput:
    coordinates: torch.Tensor              # (B, 2), highest-probability candidate
    candidate_coordinates: torch.Tensor    # (B, K, 2)
    candidate_probabilities: torch.Tensor  # (B, K), probabilities over ALL cells
    candidate_cell_ids: torch.Tensor       # (B, K)
    coarse_logits: torch.Tensor            # (B, coarse_cells)
    fine_logits: torch.Tensor              # (B, coarse_cells, children_per_cell)
    embedding: torch.Tensor                # (B, n_embed), for future retrieval
    loss: torch.Tensor | None
    loss_components: dict[str, torch.Tensor]


class StreetGeoModel(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        cfg = self.config
        self.grid = HierarchicalGeoGrid(cfg)
        self.encoder = StreetViewEncoder(cfg)
        self.coarse_head = nn.Linear(cfg.n_embed, cfg.coarse_cells)
        self.fine_head = nn.Linear(cfg.n_embed, cfg.total_cells)
        self.cell_embedding = nn.Embedding(cfg.total_cells, cfg.cell_embed_dim)
        self.offset_head = nn.Sequential(
            nn.Linear(cfg.n_embed + cfg.cell_embed_dim, cfg.n_embed), nn.GELU(),
            nn.Linear(cfg.n_embed, 2), nn.Tanh(),
        )
        self.apply(self._init_weights)
        for parameter in (self.encoder.position_embedding, self.encoder.pool_queries,
                          self.encoder.geo_token):
            nn.init.trunc_normal_(parameter, std=0.02)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.Embedding)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def _offsets(self, embedding, cell_ids):
        visual = embedding.unsqueeze(1).expand(-1, cell_ids.shape[1], -1)
        return self.offset_head(torch.cat((visual, self.cell_embedding(cell_ids)), dim=-1))

    def _rank_candidates(self, embedding, log_probs, top_k, coordinate_filter=None):
        if coordinate_filter is None:
            values, cell_ids = log_probs.topk(top_k, dim=-1)
            coordinates = self.grid.decode(cell_ids, self._offsets(embedding, cell_ids))
            return values, cell_ids, coordinates
        # Scan in probability order with bounded offset-head memory. Filtering the
        # original top five alone could miss every valid lower-ranked candidate.
        order = log_probs.argsort(dim=-1, descending=True, stable=True)
        selected_ids, selected_coordinates = [], []
        for sample in range(embedding.shape[0]):
            kept_ids, kept_coordinates = [], []
            count = 0
            for start in range(0, order.shape[1], 256):
                ids = order[sample:sample + 1, start:start + 256]
                coordinates = self.grid.decode(ids, self._offsets(embedding[sample:sample + 1], ids))
                mask = torch.as_tensor(coordinate_filter(coordinates[0].detach().cpu().numpy()),
                                       dtype=torch.bool, device=ids.device)
                if mask.shape != ids[0].shape:
                    raise ValueError("coordinate filter must return one boolean per coordinate")
                kept_ids.append(ids[0, mask])
                kept_coordinates.append(coordinates[0, mask])
                count += int(mask.sum())
                if count >= top_k:
                    break
            if not count:
                raise EmptyPredictionDomainError("No predicted coordinates fall inside the selected coverage domain")
            selected_ids.append(torch.cat(kept_ids)[:top_k])
            selected_coordinates.append(torch.cat(kept_coordinates)[:top_k])
        # Tiny/custom grids can have fewer than top_k eligible locations.
        size = min(len(ids) for ids in selected_ids)
        ids = torch.stack([ids[:size] for ids in selected_ids])
        coordinates = torch.stack([coords[:size] for coords in selected_coordinates])
        return log_probs.gather(1, ids), ids, coordinates

    def forward(self, images, targets=None, *, view_mask=None, headings=None, top_k=5,
                coordinate_filter=None):
        if type(top_k) is not int or not 1 <= top_k <= self.config.total_cells:
            raise ValueError(f"top_k must be an integer in [1, {self.config.total_cells}]")
        embedding = self.encoder(images, view_mask, headings)
        return self.forward_embedding(embedding, targets, top_k=top_k, coordinate_filter=coordinate_filter)

    def forward_embedding(self, embedding, targets=None, *, top_k=5, coordinate_filter=None):
        """Run the geographic heads on features from this model's frozen encoder."""
        if type(top_k) is not int or not 1 <= top_k <= self.config.total_cells:
            raise ValueError(f"top_k must be an integer in [1, {self.config.total_cells}]")
        if embedding.ndim != 2 or embedding.shape[1] != self.config.n_embed:
            raise ValueError("embedding must have shape [batch, n_embed]")
        batch = embedding.shape[0]
        coarse = self.coarse_head(embedding)
        fine = self.fine_head(embedding).reshape(batch, self.config.coarse_cells, -1)
        # Exact joint probabilities over all leaves, without pruning coarse cells.
        log_probs = F.log_softmax(coarse.float(), -1).unsqueeze(-1)
        log_probs = (log_probs + F.log_softmax(fine.float(), -1)).flatten(1)
        values, cell_ids, coordinates = self._rank_candidates(embedding, log_probs, top_k, coordinate_filter)
        losses = {}
        loss = None
        if targets is not None:
            if targets.shape != (batch, 2):
                raise ValueError("targets must have shape (B, 2): latitude, longitude")
            targets = targets.to(device=embedding.device, dtype=torch.float32)
            parent, child, target_offsets = self.grid.encode(targets)
            true_ids = parent * self.config.children_per_cell + child
            # Teacher forcing trains the true cell even when classification is wrong.
            offsets = self._offsets(embedding, true_ids.unsqueeze(1)).squeeze(1).float()
            predicted_xyz = self.grid.unit_vectors(true_ids, offsets)
            target_xyz = coordinates_to_unit_vectors(targets)
            losses = {
                "coarse": F.cross_entropy(coarse.float(), parent),
                "fine": F.cross_entropy(fine[torch.arange(batch, device=embedding.device), parent].float(), child),
                "offset": F.smooth_l1_loss(offsets, target_offsets),
                "spherical": torch.linalg.vector_norm(predicted_xyz - target_xyz, dim=-1).mean(),
            }
            loss = (losses["coarse"] + losses["fine"]
                    + self.config.offset_loss_weight * losses["offset"]
                    + self.config.spherical_loss_weight * losses["spherical"])
        return GeolocationOutput(coordinates[:, 0], coordinates, values.exp(), cell_ids,
                                 coarse, fine, F.normalize(embedding, dim=-1), loss, losses)

    @torch.inference_mode()
    def predict(self, images, *, view_mask=None, headings=None, top_k=5, coordinate_filter=None):
        """One ranked list per sample. Restores the caller's train/eval state."""
        was_training = self.training
        self.eval()
        try:
            output = self(images, view_mask=view_mask, headings=headings, top_k=top_k,
                          coordinate_filter=coordinate_filter)
            coords = output.candidate_coordinates.cpu().tolist()
            probs = output.candidate_probabilities.cpu().tolist()
            ids = output.candidate_cell_ids.cpu().tolist()
            return [[LocationPrediction(lat, lon, prob, cell)
                     for (lat, lon), prob, cell in zip(sample, scores, cells)]
                    for sample, scores, cells in zip(coords, probs, ids)]
        finally:
            self.train(was_training)
