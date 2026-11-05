"""Mapillary admission rules shared by preparation, the workbench and training."""
import json
from pathlib import Path
import re

MAPILLARY_SOURCE = "Mapillary"


def require_mapillary_row(row):
    if not isinstance(row, dict) or row.get("source") != MAPILLARY_SOURCE:
        raise ValueError("预训练仅接受 Mapillary；请使用 Mapillary API 下载生成的清单")
    identifier, image = row.get("mapillary_id"), row.get("image")
    if (not isinstance(identifier, str) or not re.fullmatch(r"[0-9]+", identifier) or
            not isinstance(image, str) or not image or "images" in row or
            Path(image).stem != identifier or "osv5m_id" in row):
        raise ValueError("Mapillary 清单缺少有效的原始图片 ID，或图片文件名与 ID 不匹配；请重新下载/导入")


def check_mapillary_manifest_header(path):
    """Fast GUI preflight; training validates every row, including resumed runs."""
    path = Path(path)
    with path.open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    require_mapillary_row(json.loads(line))
                except ValueError as error:
                    raise ValueError(f"{path}:{number}: {error}") from error
                return
    raise ValueError(f"{path}: Mapillary 清单为空")
