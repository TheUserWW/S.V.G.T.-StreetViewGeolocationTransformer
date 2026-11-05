"""GUI commands and background processes; no GUI toolkit or ML import here."""

from dataclasses import dataclass
from datetime import datetime
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import uuid

from streetgeo.paths import ROOT, validate_checkpoint_path
DATA_SOURCES = ("Mapillary（API 街景）",)
MODEL_PRESETS = {
    "1.35 亿参数（原版 · 514 MiB）": {
        "config": "configs/streetgeo_514m.json", "batch_size": "64", "directory": "streetgeo_514m_mapillary",
    },
    "3.98 亿参数（新版 · 1.48 GiB）": {
        "config": "configs/streetgeo_400m.json", "batch_size": "32", "directory": "streetgeo_400m_mapillary",
    },
}
CUSTOM_MODEL_PRESET = "自定义 JSON 配置"
DEFAULT_MODEL_PRESET = "3.98 亿参数（新版 · 1.48 GiB）"
DEFAULTS = {
    "sidebar_width": "196",
    "split_output": "data/world_mapillary", "data_source": DATA_SOURCES[0],
    "mapillary_root": "data/mapillary", "block_degrees": "0.02", "guard_m": "250",
    "train_manifest": "data/world_mapillary/train.jsonl", "val_manifest": "data/world_mapillary/val.jsonl",
    "output": "checkpoints/streetgeo_400m_mapillary/best.pt", "resume": "checkpoints/streetgeo_400m_mapillary/best.last.pt",
    "mode": "新建训练", "epochs": "4", "batch_size": "32", "num_workers": "4",
    "device": "cuda", "amp": "bf16", "learning_rate": "0.0003", "warmup_steps": "1000",
    "save_every_steps": "500", "save_every_seconds": "300", "model_config": "configs/streetgeo_400m.json", "init_from": "",
    "model_preset": DEFAULT_MODEL_PRESET,
    "download_continents": "", "download_countries": "USA", "download_regions": "",
    "download_geojson": "", "download_bbox": "", "max_images": "10000", "max_pages": "2000",
    "max_gb": "20", "min_free_gb": "5", "workers": "4", "requests_per_second": "2",
    "resolution": "1024",
    "environment_specialist": "关闭", "unfreeze_last": "2", "visual_learning_rate": "0.000005",
    "road_mask_top_width": "20", "road_mask_bottom_width": "100", "road_mask_height": "45",
}


def resolve_path(value):
    path = Path(str(value).strip()).expanduser()
    return path if path.is_absolute() else ROOT / path


def model_preset_for_config(value):
    if isinstance(value, dict):
        return next((name for name, preset in MODEL_PRESETS.items()
                     if json.loads(resolve_path(preset["config"]).read_text(encoding="utf-8")) == value),
                    CUSTOM_MODEL_PRESET)
    if not str(value).strip():
        return DEFAULT_MODEL_PRESET
    selected = resolve_path(value).resolve()
    return next((name for name, preset in MODEL_PRESETS.items()
                 if selected == resolve_path(preset["config"]).resolve()), CUSTOM_MODEL_PRESET)


def new_model_output(model_config, filename="best.pt"):
    preset = MODEL_PRESETS.get(model_preset_for_config(model_config), {})
    directory = ROOT / "checkpoints" / preset.get("directory", "streetgeo_custom_mapillary")
    candidate = directory / filename
    suffix = 2
    while candidate.exists() or candidate.with_name(candidate.stem + ".last" + candidate.suffix).exists():
        candidate = directory / f"{Path(filename).stem}_{suffix}{Path(filename).suffix}"
        suffix += 1
    return candidate


def load_settings(path):
    result = dict(DEFAULTS)
    try:
        saved = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(saved, dict):
            result.update({key: str(value) for key, value in saved.items() if key in DEFAULTS})
            if saved.get("data_source") != DATA_SOURCES[0]:
                for key in ("split_output", "train_manifest", "val_manifest", "output", "resume", "mode", "init_from"):
                    result[key] = DEFAULTS[key]
    except (OSError, ValueError, TypeError):
        pass
    result["data_source"] = DATA_SOURCES[0]
    result["model_preset"] = model_preset_for_config(result["model_config"])
    return result


def save_settings(path, values):
    """Whitelist ordinary preferences; credentials are never persisted."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({key: str(values.get(key, default)) for key, default in DEFAULTS.items()},
                                    ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def number(values, name, label, *, minimum=0, integer=False, strict=False):
    try:
        value = int(values[name]) if integer else float(values[name])
    except (TypeError, ValueError, KeyError):
        raise ValueError(f"{label}需要填写有效的{'整数' if integer else '数字'}") from None
    if not math.isfinite(value) or value < minimum or (strict and value == minimum):
        raise ValueError(f"{label}需要{'大于' if strict else '不小于'} {minimum}")
    return value


def input_file(value, label):
    if not str(value).strip() or not resolve_path(value).is_file():
        raise ValueError(f"找不到{label}，请先选择有效文件")
    return str(resolve_path(value))


def build_download(values, stop_file=None):
    from streetgeo.datasets.regions import parse_codes, validate_selection
    continents = sorted(set(part.strip().lower() for part in values["download_continents"].split(",") if part.strip()))
    countries, regions = parse_codes(values["download_countries"]), parse_codes(values["download_regions"])
    validate_selection(continents, countries, regions)
    if not any((continents, countries, regions, values["download_geojson"].strip(), values["download_bbox"].strip())):
        raise ValueError("请选择至少一个大洲、国家/地区或自定义区域")
    if not values["mapillary_root"].strip():
        raise ValueError("请选择 Mapillary 下载目录")
    arguments = ["--output", str(resolve_path(values["mapillary_root"])), "--resolution", values["resolution"]]
    if values["resolution"] not in ("256", "1024", "2048"):
        raise ValueError("请选择有效的图片分辨率")
    for flag, selected in (("continents", continents), ("countries", countries), ("regions", regions)):
        if selected:
            arguments += ["--" + flag, ",".join(selected)]
    if values["download_geojson"].strip():
        arguments += ["--geojson", input_file(values["download_geojson"], "区域 GeoJSON")]
    if values["download_bbox"].strip():
        try:
            west, south, east, north = map(float, values["download_bbox"].replace(",", " ").split())
        except (ValueError, TypeError):
            raise ValueError("区域范围需填写：西经度 南纬度 东经度 北纬度") from None
        if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
            raise ValueError("区域经纬度无效；跨日期变更线的区域请分两次下载")
        arguments += ["--bbox", *map(str, (west, south, east, north))]
    for key, title in (("max_images", "下载目标张数"), ("max_pages", "每次查询页数"), ("workers", "下载并发数")):
        arguments += ["--" + key.replace("_", "-"), str(number(values, key, title, integer=True, strict=True))]
    for key, title in (("max_gb", "图片容量上限"), ("min_free_gb", "预留磁盘容量"), ("requests_per_second", "每秒请求数")):
        arguments += ["--" + key.replace("_", "-"), str(number(values, key, title, strict=key != "min_free_gb"))]
    if stop_file:
        arguments += ["--stop-file", str(stop_file)]
    return arguments


def build_prepare(values, stop_file=None):
    if values.get("data_source", DATA_SOURCES[0]) != DATA_SOURCES[0]:
        raise ValueError("训练仅支持 Mapillary，不接受其他数据来源")
    from streetgeo.datasets.sources import resolve_source
    root = resolve_path(values.get("mapillary_root") or DEFAULTS["mapillary_root"])
    resolve_source(root)
    if not values["split_output"].strip():
        raise ValueError("请选择数据划分输出目录")
    arguments = ["--dataset-root", str(root), "--output-dir", str(resolve_path(values["split_output"])),
                 "--block-degrees", str(number(values, "block_degrees", "地理分组大小", strict=True)),
                 "--guard-m", str(number(values, "guard_m", "近邻隔离距离"))]
    if stop_file:
        arguments += ["--stop-file", str(stop_file)]
    return arguments


def build_gpu_check(values):
    arguments = ["--batch-size", str(number(values, "batch_size", "批次大小", integer=True, strict=True))]
    arguments += ["--amp", values["amp"]]
    if values["model_config"].strip():
        arguments += ["--model-config", input_file(values["model_config"], "模型配置")]
    return arguments


def build_road_mask(values):
    from streetgeo.datasets.road_mask import RoadMask
    return RoadMask(*(number(values, key, title, strict=True) / 100 for key, title in (
        ("road_mask_top_width", "梯形上底"), ("road_mask_bottom_width", "梯形下底"), ("road_mask_height", "梯形高度"))))


def build_training(values, stop_file):
    device, amp = values["device"], values["amp"]
    if device not in ("auto", "cuda", "cpu") or amp not in ("auto", "bf16", "fp16", "off"):
        raise ValueError("请选择有效的训练设备和精度")
    arguments = ["--device", device, "--num-workers", str(number(values, "num_workers", "读取进程数", integer=True)),
                 "--save-every-steps", str(number(values, "save_every_steps", "保存步数间隔", integer=True, strict=True)),
                 "--save-every-seconds", str(number(values, "save_every_seconds", "保存时间间隔")),
                 "--log-every-steps", "1", "--stop-file", str(stop_file)]
    if values["mode"] == "继续上次训练":
        # Saved batch size, schedule, manifests and precision remain authoritative.
        return arguments + ["--resume", input_file(values["resume"], "续训 checkpoint")]
    if device == "cpu" and amp in ("bf16", "fp16"):
        raise ValueError("CPU 训练请选择 off 或 auto 精度")
    if not values["output"].strip():
        raise ValueError("请选择模型输出文件")
    output = validate_checkpoint_path(resolve_path(values["output"]))
    if output.exists() or output.with_name(output.stem + ".last" + output.suffix).exists():
        raise ValueError("该位置已有训练进度，请选择“继续上次训练”或更换输出文件")
    environment = values.get("environment_specialist", "关闭")
    if environment not in ("关闭", "开启 · 全图/专项交替"):
        raise ValueError("请选择有效的环境专项微调模式")
    if environment != "关闭":
        input_file(values.get("init_from", ""), "环境专项初始模型")
        if not values["val_manifest"].strip():
            raise ValueError("环境专项微调需要 Mapillary 验证清单")
        arguments += ["--environment-specialist",
                      "--unfreeze-last", str(number(values, "unfreeze_last", "末端解冻层数", integer=True, strict=True)),
                      "--visual-learning-rate", str(number(values, "visual_learning_rate", "视觉层学习率", strict=True))]
        mask = build_road_mask(values)
        arguments += ["--road-mask-top-width", str(mask.top_width),
                      "--road-mask-bottom-width", str(mask.bottom_width), "--road-mask-height", str(mask.height)]
    from streetgeo.datasets.policy import check_mapillary_manifest_header
    check_mapillary_manifest_header(input_file(values["train_manifest"], "Mapillary 训练清单"))
    if values["val_manifest"].strip():
        check_mapillary_manifest_header(input_file(values["val_manifest"], "Mapillary 验证清单"))
    arguments += ["--train-manifest", input_file(values["train_manifest"], "Mapillary 训练清单"),
                  "--output", str(output), "--amp", amp]
    if values["val_manifest"].strip():
        arguments += ["--val-manifest", input_file(values["val_manifest"], "验证清单")]
    for name, label in (("epochs", "训练轮数"), ("batch_size", "批次大小")):
        arguments += ["--" + name.replace("_", "-"), str(number(values, name, label, integer=True, strict=True))]
    arguments += ["--learning-rate", str(number(values, "learning_rate", "学习率", strict=True)),
                  "--warmup-steps", str(number(values, "warmup_steps", "学习率预热步数", integer=True))]
    for name, label in (("model_config", "模型配置"), ("init_from", "初始权重")):
        if values[name].strip():
            arguments += ["--" + name.replace("_", "-"), input_file(values[name], label)]
    return arguments


@dataclass
class ProcessEvent:
    kind: str
    job: str
    data: object


class ProcessRunner:
    """Own one worker process; deliver output to Tk through a thread-safe queue."""

    def __init__(self, events=None, workspace=None):
        self.events = events if events is not None else queue.Queue()
        self.workspace = Path(workspace) if workspace else ROOT / ".gui"
        self.process = None
        self.thread = None
        self.stop_file = None
        self.log_path = None
        self.job = None
        self.stopping = False

    @property
    def busy(self):
        # Keep ownership until output and the final event have been collected.
        return self.thread is not None and self.thread.is_alive()

    def new_stop_file(self):
        path = self.workspace / "runtime" / (uuid.uuid4().hex + ".stop")
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def start(self, job, module, arguments, *, stop_file=None, extra_env=None):
        if self.busy:
            raise RuntimeError("当前任务尚未结束")
        if not module or not all(part.isidentifier() for part in module.split(".")):
            raise ValueError("后台任务必须使用 Python 模块名，例如 streetgeo.training.train")
        python = Path(sys.executable)
        if python.name.lower() == "pythonw.exe":
            python = python.with_name("python.exe")
        environment = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
        environment.setdefault("OMP_NUM_THREADS", "4")
        environment.setdefault("MKL_NUM_THREADS", "4")
        environment.update(extra_env or {})
        secrets = [environment.get("MAPILLARY_ACCESS_TOKEN", "")]
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.log_path = self.workspace / "logs" / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6] + ".log")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.stop_file, self.job, self.stopping = stop_file, job, False
        command = [str(python), "-u", "-m", module, *map(str, arguments)]
        self.process = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                        encoding="utf-8", errors="replace", bufsize=1,
                                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        self.events.put(ProcessEvent("started", job, {"log_path": str(self.log_path)}))
        self.thread = threading.Thread(target=self._read, args=(self.process, job, self.log_path, secrets, stop_file), daemon=True)
        self.thread.start()

    def _read(self, process, job, log_path, secrets, stop_file):
        log = None
        try:
            try:
                log = log_path.open("w", encoding="utf-8", buffering=1)
            except OSError:
                self.events.put(ProcessEvent("line", job, "无法创建日志文件；窗口仍继续显示实时输出。"))
            for line in process.stdout:
                for secret in secrets:
                    if secret:
                        line = line.replace(secret, "[令牌已隐藏]")
                if log is not None:
                    try:
                        log.write(line)
                    except OSError:
                        try:
                            log.close()
                        except OSError:
                            pass
                        log = None
                        self.events.put(ProcessEvent("line", job, "日志文件写入失败；窗口仍继续显示实时输出。"))
                self.events.put(ProcessEvent("line", job, line.rstrip("\r\n")))
            code = process.wait()
        except Exception as error:
            self.events.put(ProcessEvent("line", job, f"日志读取失败：{type(error).__name__}"))
            code = process.wait()
        finally:
            if log is not None:
                log.close()
            process.stdout.close()
            if stop_file:
                Path(stop_file).unlink(missing_ok=True)
        self.events.put(ProcessEvent("finished", job, {"returncode": code, "stopped": self.stopping}))

    def request_stop(self):
        if not self.busy or not self.stop_file:
            return False
        Path(self.stop_file).touch()
        self.stopping = True
        return True
