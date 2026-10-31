"""Spherical geometry and a data-independent hierarchical equal-area grid.

Cells partition sin(latitude) and longitude. IDs are parent-major.
All coordinates use (latitude, longitude) in degrees.
"""

import math
from dataclasses import dataclass
from urllib.parse import urlencode

import torch
from streetgeo.core.config import ModelConfig

EARTH_RADIUS_KM = 6371.0088


def validate_coordinates(coordinates):
    if coordinates.ndim < 1 or coordinates.shape[-1] != 2:
        raise ValueError("coordinates must have shape (..., 2): latitude, longitude")
    if not torch.isfinite(coordinates).all():
        raise ValueError("coordinates must be finite")
    if (coordinates[..., 0].abs() > 90).any():
        raise ValueError("latitude must be in [-90, 90]")
    if (coordinates[..., 1].abs() > 180).any():
        raise ValueError("longitude must be in [-180, 180]")


def coordinates_to_unit_vectors(coordinates):
    lat, lon = torch.deg2rad(coordinates.double()).unbind(-1)
    return torch.stack((lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()), -1)


def haversine_km(predicted, target):
    """Great-circle error for evaluation, including the antimeridian."""
    lat1, lon1 = torch.deg2rad(predicted.double()).unbind(-1)
    lat2, lon2 = torch.deg2rad(target.double()).unbind(-1)
    a = ((lat2 - lat1) / 2).sin().square()
    a = a + lat1.cos() * lat2.cos() * ((lon2 - lon1) / 2).sin().square()
    a = a.clamp(0, 1)
    return 2 * EARTH_RADIUS_KM * torch.atan2(a.sqrt(), (1 - a).sqrt())


@dataclass(frozen=True)
class LocationPrediction:
    latitude: float
    longitude: float
    probability: float  # Raw joint cell probability, not calibrated accuracy.
    cell_id: int

    @property
    def coordinates(self):
        return f"{self.latitude:.6f}, {self.longitude:.6f}"

    @property
    def maps_url(self):
        query = f"{self.latitude:.6f},{self.longitude:.6f}"
        return "https://www.google.com/maps/search/?" + urlencode({"api": 1, "query": query})


class HierarchicalGeoGrid:
    def __init__(self, config: ModelConfig):
        self.lat_bins = config.coarse_lat_bins
        self.lon_bins = config.coarse_lon_bins
        self.fine_bins = config.fine_bins
        self.children = config.children_per_cell
        self.total_cells = config.total_cells

    def encode(self, coordinates):
        """Return coarse ID, local child ID and within-cell offsets in [-1, 1]."""
        validate_coordinates(coordinates)
        latitude, longitude = coordinates.double().unbind(-1)
        u = (torch.deg2rad(latitude).sin() + 1) / 2
        v = torch.remainder(longitude + 180, 360) / 360
        rows, cols = self.lat_bins * self.fine_bins, self.lon_bins * self.fine_bins
        row_position, col_position = u * rows, v * cols
        row = row_position.floor().long().clamp(0, rows - 1)
        col = col_position.floor().long().clamp(0, cols - 1)
        coarse = (row // self.fine_bins) * self.lon_bins + col // self.fine_bins
        child = (row % self.fine_bins) * self.fine_bins + col % self.fine_bins
        offsets = torch.stack((2 * (row_position - row) - 1,
                               2 * (col_position - col) - 1), -1).clamp(-1, 1)
        return coarse, child, offsets

    def _positions(self, cell_ids, offsets):
        parent, child = cell_ids // self.children, cell_ids % self.children
        row = (parent // self.lon_bins) * self.fine_bins + child // self.fine_bins
        col = (parent % self.lon_bins) * self.fine_bins + child % self.fine_bins
        # FP64 avoids losing latitude precision close to the poles, where
        # sin(latitude) can round to exactly 1 in FP32. Only geometry uses FP64.
        offsets = offsets.double()
        u = (row + (offsets[..., 0] + 1) / 2) / (self.lat_bins * self.fine_bins)
        v = (col + (offsets[..., 1] + 1) / 2) / (self.lon_bins * self.fine_bins)
        return (2 * u - 1).clamp(-1, 1), (v * 2 - 1) * math.pi

    def decode(self, cell_ids, offsets):
        z, longitude = self._positions(cell_ids, offsets)
        latitude = torch.rad2deg(z.asin())
        longitude = torch.remainder(torch.rad2deg(longitude) + 180, 360) - 180
        return torch.stack((latitude, longitude), -1)

    def unit_vectors(self, cell_ids, offsets):
        """Differentiable spherical positions without asin's polar singularity."""
        z, longitude = self._positions(cell_ids, offsets)
        radius = (1 - z.square()).clamp_min(1e-12).sqrt()
        return torch.stack((radius * longitude.cos(), radius * longitude.sin(), z), -1)
