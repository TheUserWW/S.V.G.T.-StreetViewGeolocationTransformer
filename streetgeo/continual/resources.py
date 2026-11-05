"""VM-local memory/disk budgets. No host idle detection or hidden worker daemon."""
import os
import shutil
import psutil

def lower_priority(pid=None):
    try:
        process=psutil.Process(pid)
        process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name=='nt' else 10)
    except (psutil.Error,OSError): pass

def budget(config):
    if psutil.virtual_memory().percent>=config.max_memory_percent: return '内存繁忙 / Memory busy'
    if shutil.disk_usage(config.runtime).free/1024**3<config.min_free_disk_gb: return '磁盘空间不足 / Low disk space'
    return None
