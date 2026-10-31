"""Web Mercator coordinates and screenshot calibration, independent of the GUI/model."""

from dataclasses import dataclass
import math

MAX_LATITUDE = 85.0511287798066


def coordinates(latitude, longitude):
    latitude, longitude = float(latitude), float(longitude)
    if not math.isfinite(latitude) or not math.isfinite(longitude):
        raise ValueError("经纬度必须为有限数字")
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        raise ValueError("纬度范围为 -90～90，经度范围为 -180～180；顺序为纬度、经度")
    return latitude, longitude


def parse_coordinates(text):
    parts = text.strip().replace("，", ",").replace(",", " ").split()
    if len(parts) != 2:
        raise ValueError("请输入“纬度, 经度”，例如 34.0522, -118.2437")
    return coordinates(*parts)


def project(latitude, longitude):
    """Normalized Mercator world coordinates; poles are clipped for display only."""
    latitude, longitude = coordinates(latitude, longitude)
    latitude = max(-MAX_LATITUDE, min(MAX_LATITUDE, latitude))
    return ((longitude + 180) / 360,
            (1 - math.asinh(math.tan(math.radians(latitude))) / math.pi) / 2)


def unproject(x, y):
    return (math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * max(0, min(1, y)))))),
            (x % 1) * 360 - 180)


def wrap_delta(x, reference):
    return (x - reference + 0.5) % 1 - 0.5


@dataclass(frozen=True)
class ControlPoint:
    latitude: float
    longitude: float
    x: float  # Original screenshot pixel coordinates, never resized preview pixels.
    y: float


class ScreenshotCalibration:
    """Least-squares affine fit in Mercator space, including date-line wrapping."""

    def __init__(self, points):
        import numpy as np

        if len(points) < 3:
            raise ValueError("至少需要 3 对分散且不在同一直线上的对应点；建议使用 4 对以上")
        self.points = list(points)
        for point in points:
            coordinates(point.latitude, point.longitude)
            if abs(point.latitude) > MAX_LATITUDE:
                raise ValueError("截图校准不支持墨卡托地图范围以外的极区")
            if not all(math.isfinite(value) and value >= 0 for value in (point.x, point.y)):
                raise ValueError("截图对应点必须是有效的非负像素坐标")
        projected = np.array([project(p.latitude, p.longitude) for p in points], dtype=np.float64)
        self.reference = float(projected[0, 0])
        projected[:, 0] = [wrap_delta(x, self.reference) for x in projected[:, 0]]
        if np.ptp(projected[:, 0]) > 0.5:
            raise ValueError("校准范围过大，请放大到同一局部区域后重新截图与选点")
        self.center = projected.mean(axis=0)
        self.scale = float(np.linalg.norm(projected - self.center, axis=1).max())
        if self.scale < 1e-12:
            raise ValueError("对应点过于接近，请分散选择")
        source = np.column_stack(((projected - self.center) / self.scale, np.ones(len(points))))
        destination = np.array([[p.x, p.y] for p in points], dtype=np.float64)
        if np.linalg.matrix_rank(source) != 3 or np.linalg.cond(source) > 1e4:
            raise ValueError("地图对应点接近共线，请选择分散的三角形或四角位置")
        self.matrix = np.linalg.lstsq(source, destination, rcond=None)[0]
        # Also reject a screenshot whose controls collapse onto a line.
        singular = np.linalg.svd(self.matrix[:2], compute_uv=False)
        if singular[-1] < 1e-8 or singular[0] / singular[-1] > 1e4:
            raise ValueError("截图对应点接近共线，无法建立可靠的位置变换")
        residuals = np.linalg.norm(source @ self.matrix - destination, axis=1)
        self.rms_pixels = float(np.sqrt(np.mean(residuals ** 2)))
        self.max_pixels = float(residuals.max())

    def locate(self, latitude, longitude):
        import numpy as np

        if abs(float(latitude)) > MAX_LATITUDE:
            raise ValueError("该坐标位于墨卡托地图可显示的极区范围之外")
        x, y = project(latitude, longitude)
        normalized = (np.array([wrap_delta(x, self.reference), y]) - self.center) / self.scale
        result = np.array([*normalized, 1.0]) @ self.matrix
        return float(result[0]), float(result[1])

