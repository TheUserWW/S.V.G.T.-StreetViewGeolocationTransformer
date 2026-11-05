"""Normalized, bottom-anchored road trapezoid shared by training and preview."""
from dataclasses import dataclass
import math

from PIL import ImageDraw

MASK_FILL = (124, 116, 104)


@dataclass(frozen=True)
class RoadMask:
    top_width: float = .20
    bottom_width: float = 1.0
    height: float = .45

    def __post_init__(self):
        if (not all(math.isfinite(value) for value in (self.top_width, self.bottom_width, self.height))
                or not 0 < self.top_width <= self.bottom_width <= 1 or not 0 < self.height <= 1):
            raise ValueError("梯形参数须满足：0 < 上底 ≤ 下底 ≤ 100%，0 < 高 ≤ 100%")

    def vertices(self, size):
        width, height = size[0] - 1, size[1] - 1
        return [(round((1 - self.top_width) * width / 2), round((1 - self.height) * height)),
                (round((1 + self.top_width) * width / 2), round((1 - self.height) * height)),
                (round((1 + self.bottom_width) * width / 2), height),
                (round((1 - self.bottom_width) * width / 2), height)]

    @property
    def area_fraction(self):
        return (self.top_width + self.bottom_width) * self.height / 2

    def apply(self, image):
        result = image.copy()
        ImageDraw.Draw(result).polygon(self.vertices(result.size), fill=MASK_FILL)
        return result
