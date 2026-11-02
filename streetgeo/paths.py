"""Stable workspace paths, independent of package and launch locations."""
import os
from pathlib import Path, PureWindowsPath

ROOT = Path(__file__).resolve().parents[1]
MAP_ASSETS = ROOT / "assets" / "maps"


def validate_checkpoint_path(value):
    """Reject invalid Windows names before model allocation or temporary writes."""
    path = Path(value)
    if not path.name or path.name in (".", ".."):
        raise ValueError("请选择有效的模型输出文件")
    if os.name == "nt":
        for part in path.parts:
            if part == path.anchor or part in (".", ".."):
                continue
            if any(char in '<>:"|?*' or ord(char) < 32 for char in part):
                raise ValueError(f"模型保存路径含 Windows 非法字符：{part!r}。例如请用 SVGT_4B.pt，文件名不能含冒号。")
            if part.endswith((" ", ".")) or PureWindowsPath(part).is_reserved():
                raise ValueError(f"模型保存路径含 Windows 保留名称或无效结尾：{part!r}")
    return path
