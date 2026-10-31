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

from streetgeo.paths import ROOT
DATA_SOURCES = ("GSV-Cities（全球）", "Mapillary / JSONL", "混合数据（全球）", "OpenStreetView-5M（部分图片可用）")
DEFAULTS = {
    "dataset_dir": "data/mapillary_us", "region": "conus", "states": "",
    "max_images": "10000", "resolution": "1024", "download_workers": "4",
    "requests_per_second": "2", "max_gb": "20", "min_free_gb": "5", "max_pages": "2000",
    "split_manifest": "data/mapillary_us/all.jsonl", "split_output": "data/world_osv_514m",
    "data_source": DATA_SOURCES[3], "gsv_root": "data/archive", "osv_root": "",
    "block_degrees": "0.02", "guard_m": "250",
    "train_manifest": "data/world_osv_514m/train.jsonl", "val_manifest": "data/world_osv_514m/val.jsonl",
    "output": "checkpoints/streetgeo_514m_osv/best.pt", "resume": "checkpoints/streetgeo_514m_osv/best.last.pt",
    "mode": "新建训练", "epochs": "4", "batch_size": "64", "num_workers": "4",
    "device": "cuda", "amp": "bf16", "learning_rate": "0.0003", "warmup_steps": "1000",
    "save_every_steps": "500", "save_every_seconds": "300", "model_config": "", "init_from": "",
}


def resolve_path(value):
    path = Path(str(value).strip()).expanduser()
    return path if path.is_absolute() else ROOT / path


def load_settings(path):
    result = dict(DEFAULTS)
    try:
        saved = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(saved, dict):
            result.update({key: str(value) for key, value in saved.items() if key in DEFAULTS})
    except (OSError, ValueError, TypeError):
        pass
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


def build_download(values, stop_file):
    arguments = ["--output", str(resolve_path(values["dataset_dir"])), "--region", values["region"],
                 "--stop-file", str(stop_file)]
    if values["region"] not in ("conus", "us") or values["resolution"] not in ("256", "1024", "2048"):
        raise ValueError("请选择有效的地区和图像分辨率")
    for name, label in (("max_images", "目标图片数量"), ("download_workers", "下载线程数"),
                        ("max_pages", "本次查询页数")):
        value = number(values, name, label, integer=True, minimum=0, strict=True)
        flag = "workers" if name == "download_workers" else name.replace("_", "-")
        arguments += ["--" + flag, str(value)]
    for name, label, strict in (("requests_per_second", "每秒请求数", True), ("max_gb", "图片磁盘预算", True),
                                ("min_free_gb", "保留磁盘空间", False)):
        arguments += ["--" + name.replace("_", "-"), str(number(values, name, label, strict=strict))]
    arguments += ["--resolution", values["resolution"]]
    states = values["states"].strip().upper().replace("，", ",")
    if states:
        if not all(len(part.strip()) == 2 and part.strip().isascii() and part.strip().isalpha() for part in states.split(",")):
            raise ValueError("州代码使用两个英文字母，多个州用逗号分隔，例如 CA,NV,AZ")
        arguments += ["--states", ",".join(part.strip() for part in states.split(","))]
    return arguments


def build_prepare(values, stop_file=None):
    source = values.get("data_source", "Mapillary / JSONL")
    if source not in DATA_SOURCES:
        raise ValueError("请选择有效的数据来源")
    arguments = []
    mixed = source == DATA_SOURCES[2]
    if source == DATA_SOURCES[0] or (mixed and values.get("gsv_root", "").strip()):
        root = resolve_path(values.get("gsv_root", "data/archive"))
        if not (root / "Images").is_dir() or not (root / "Dataframes").is_dir():
            raise ValueError("请选择包含 Images 与 Dataframes 的 GSV-Cities 解压目录")
        arguments += ["--dataset-root", str(root)]
    if source == DATA_SOURCES[1] or (mixed and values.get("split_manifest", "").strip()):
        arguments += ["--manifest", input_file(values["split_manifest"], "Mapillary / JSONL 清单")]
    if source == DATA_SOURCES[3] or (mixed and values.get("osv_root", "").strip()):
        from streetgeo.datasets.osv5m import osv_layout
        root = resolve_path(values.get("osv_root", "data/OpenStreetView-5M"))
        osv_layout(root)
        arguments += ["--dataset-root", str(root)]
    if mixed and len(arguments) < 4:
        raise ValueError("混合数据至少填写两个来源；不使用的来源路径请留空")
    if not values["split_output"].strip():
        raise ValueError("请选择数据划分输出目录")
    arguments += ["--output-dir", str(resolve_path(values["split_output"])),
                  "--block-degrees", str(number(values, "block_degrees", "地理分组大小", strict=True)),
                  "--guard-m", str(number(values, "guard_m", "近邻隔离距离"))]
    if stop_file:
        arguments += ["--stop-file", str(stop_file)]
    return arguments


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
    output = resolve_path(values["output"])
    if not values["output"].strip():
        raise ValueError("请选择模型输出文件")
    if output.exists() or output.with_name(output.stem + ".last" + output.suffix).exists():
        raise ValueError("该位置已有训练进度，请选择“继续上次训练”或更换输出文件")
    arguments += ["--train-manifest", input_file(values["train_manifest"], "训练清单"),
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
