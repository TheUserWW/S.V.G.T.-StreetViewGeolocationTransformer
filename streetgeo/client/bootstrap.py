"""Standard-library launcher copied into the standalone client directory."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import struct
import subprocess
import sys
import traceback
import venv

if __package__:
    from .tk_runtime import check_tk
else:
    from tkinter_runtime import check_tk

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / ".runtime"


def child_environment():
    environment = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME"):
        environment.pop(key, None)
    environment.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONNOUSERSITE="1",
                       PIP_DISABLE_PIP_VERSION_CHECK="1", PIP_CACHE_DIR=str(RUNTIME / "pip-cache"))
    return environment


def safe_runtime_path(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT) or path == ROOT:
        raise RuntimeError("客户端运行目录指向了程序文件夹之外，请检查 .runtime")
    return path


def check_python():
    if not (3, 10) <= sys.version_info[:2] <= (3, 15):
        raise RuntimeError("请使用 64 位 Python 3.10–3.15，推荐 Python 3.12。")
    if struct.calcsize("P") != 8:
        raise RuntimeError("当前是 32 位 Python，请改用 64 位 Python 3.12。")
    if os.name == "nt" and platform.machine().lower() not in ("amd64", "x86_64"):
        raise RuntimeError("该客户端使用 Windows x64 依赖，请使用 x64 版本的 Python。")
    try:
        import tkinter  # noqa: F401
    except ImportError as error:
        raise RuntimeError("Python 缺少 tkinter；请在 Python 安装器中添加 Tcl/Tk and IDLE 组件。") from error
    check_tk()


@contextmanager
def startup_lock():
    RUNTIME.mkdir(parents=True, exist_ok=True)
    path = safe_runtime_path(RUNTIME / "launcher.lock")
    with path.open("a+b") as lock:
        lock.seek(0)
        if os.fstat(lock.fileno()).st_size == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError("客户端已经启动或正在安装依赖，请查看已打开的窗口。") from error
        try:
            yield
        finally:
            lock.seek(0)
            if os.name == "nt":
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock, fcntl.LOCK_UN)


def environment_path():
    # Virtual environments cannot be shipped between machines. A different base
    # Python or client location gets its own environment without deleting files.
    identity = [os.path.normcase(str(ROOT)), os.path.normcase(str(getattr(sys, "_base_executable", sys.executable))),
                list(sys.version_info[:2]), platform.machine()]
    key = hashlib.sha256(json.dumps(identity).encode("utf-8")).hexdigest()[:12]
    return safe_runtime_path(RUNTIME / "envs" / key)


def environment_python(folder):
    return folder / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def run_logged(command, log):
    log.write("\nCOMMAND " + subprocess.list2cmdline([str(arg) for arg in command]) + "\n")
    log.flush()
    process = subprocess.Popen([str(arg) for arg in command], cwd=ROOT, env=child_environment(),
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding="utf-8", errors="replace")
    for line in process.stdout:
        print(line, end="", flush=True)
        log.write(line)
        log.flush()
    process.stdout.close()
    code = process.wait()
    if code:
        raise RuntimeError(f"命令执行失败（{code}），详情见 {RUNTIME / 'startup.log'}")


PROBE = r'''
import importlib, importlib.metadata, json, re, sys
expected_torch = '2.14.1' if sys.version_info[:2] >= (3,15) else '2.11.0'
requirements = [('Pillow','PIL',(10,0),(14,0)), ('numpy','numpy',(1,26),(3,0)),
                ('shapely','shapely',(2,0),(3,0)), ('torch','torch',(2,11),(3,0))]
result = {}
for distribution, module, lower, upper in requirements:
    try:
        version = importlib.metadata.version(distribution)
        numbers = tuple(int(n) for n in re.match(r'\d+(?:\.\d+)*',version).group(0).split('.'))
        if not lower <= numbers < upper:
            raise RuntimeError('Unsupported version: '+version)
        if module == 'torch' and version.split('+')[0] != expected_torch:
            raise RuntimeError('Expected torch '+expected_torch)
        imported = importlib.import_module(module)
        if module == 'torch':
            imported.ones(1).sum().item()
        result[distribution] = {'ok':True,'version':version}
    except Exception as error:
        result[distribution] = {'ok':False,'error':str(error)}
print('STREETGEO_PROBE='+json.dumps(result))
'''


def probe(python):
    result = subprocess.run([str(python), "-c", PROBE], cwd=ROOT, env=child_environment(),
                            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=90)
    for line in reversed(result.stdout.splitlines()):
        if line.startswith("STREETGEO_PROBE="):
            return json.loads(line.split("=", 1)[1])
    raise RuntimeError("依赖检查无法运行：" + (result.stderr or result.stdout)[-1600:])


def ensure_environment(log):
    folder = environment_path()
    python = environment_python(folder)
    usable = False
    if python.is_file():
        try:
            usable = subprocess.run([str(python), "-c", "import sys"], cwd=ROOT, env=child_environment(),
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15).returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            pass
    if not usable:
        print("[1/3] 正在创建客户端独立 Python 环境…", flush=True)
        venv.EnvBuilder(with_pip=True).create(folder)
    else:
        print("[1/3] 已找到客户端独立 Python 环境。", flush=True)
    run_logged([python, ROOT / "tkinter_runtime.py"], log)
    status = probe(python)
    core_missing = any(not status[name]["ok"] for name in ("Pillow", "numpy", "shapely"))
    torch_missing = not status["torch"]["ok"]
    pip = [python, "-m", "pip", "install", "--only-binary=:all:", "--progress-bar", "off", "--timeout", "60", "--retries", "3"]
    if core_missing or torch_missing:
        print("[2/3] 正在安装或修复依赖，首次运行需要联网，请稍候…", flush=True)
        run_logged([python, "-m", "ensurepip", "--upgrade"], log)
        if core_missing:
            broken = any("version" not in value and "No module named" not in value.get("error", "")
                         and "No package metadata" not in value.get("error", "")
                         for name, value in status.items() if name != "torch" and not value["ok"])
            run_logged([*pip, *(["--force-reinstall"] if broken else []),
                        "--index-url", "https://pypi.org/simple", "-r", ROOT / "requirements.txt"], log)
        if torch_missing:
            run_logged([*pip, "--force-reinstall", "-r", ROOT / "requirements-torch-cpu.txt"], log)
        status = probe(python)
    else:
        print("[2/3] 依赖检查通过，无需重新安装。", flush=True)
    failed = {name: result for name, result in status.items() if not result["ok"]}
    if failed:
        raise RuntimeError("依赖仍无法加载：" + json.dumps(failed, ensure_ascii=False)
                           + "。若提示 DLL load failed，请检查 Microsoft Visual C++ 运行库。")
    log.write("\nDEPENDENCIES " + json.dumps(status, ensure_ascii=False) + "\n")
    log.flush()
    return python


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup-only", action="store_true", help="prepare dependencies without opening a window")
    parser.add_argument("--self-test", action="store_true", help="prepare dependencies and test a CPU prediction")
    args = parser.parse_args()
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    try:
        check_python()
        safe_runtime_path(RUNTIME)
        with startup_lock(), (RUNTIME / "startup.log").open("a", encoding="utf-8", buffering=1) as log:
            if not (ROOT / "best.pt").is_file():
                raise RuntimeError("缺少 best.pt，请把模型权重放回客户端文件夹。")
            if not (ROOT / "app.py").is_file():
                raise RuntimeError("客户端文件不完整，请重新解压整个文件夹。")
            python = ensure_environment(log)
            if args.setup_only:
                print("[3/3] 环境准备完成。", flush=True)
            else:
                print("[3/3] 正在启动本地预测客户端…", flush=True)
                run_logged([python, ROOT / "app.py", *(["--self-test"] if args.self_test else [])], log)
        return 0
    except Exception as error:
        print("\n启动失败：" + str(error), file=sys.stderr)
        try:
            safe_runtime_path(RUNTIME)
            RUNTIME.mkdir(parents=True, exist_ok=True)
            with (RUNTIME / "startup.log").open("a", encoding="utf-8") as log:
                traceback.print_exc(file=log)
        except (OSError, RuntimeError):
            pass
        print("请把 .runtime/startup.log 发给开发者排查。", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
