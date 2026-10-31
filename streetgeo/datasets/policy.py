"""Dataset admission rules shared by the workbench and training loader."""
import json
from pathlib import Path

OSV_SOURCE = "OpenStreetView-5M"


def require_osv_row(row):
    if not isinstance(row, dict) or row.get("source") != OSV_SOURCE:
        raise ValueError("训练仅接受 OpenStreetView-5M；请使用 OSV-5M 导入器生成的清单")
    identifier, image = row.get("osv5m_id"), row.get("image")
    if (not isinstance(identifier, str) or not identifier.strip() or
            not isinstance(image, str) or not image or "images" in row or
            Path(image).stem != identifier):
        raise ValueError("OpenStreetView-5M 清单缺少有效的原始图片 ID，或图片文件名与 ID 不匹配；请重新导入")


def check_osv_manifest_header(path):
    """Fast GUI preflight; training validates every row, including resumed runs."""
    path = Path(path)
    with path.open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    require_osv_row(json.loads(line))
                except ValueError as error:
                    raise ValueError(f"{path}:{number}: {error}") from error
                return
    raise ValueError(f"{path}: OpenStreetView-5M 清单为空")
