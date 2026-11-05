from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile

PROJECT = Path(__file__).resolve().parents[2]

def atomic_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd, temp=tempfile.mkstemp(dir=path.parent,suffix='.tmp')
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as out:
            json.dump(value,out,ensure_ascii=False,indent=2,allow_nan=False); out.flush(); os.fsync(out.fileno())
        os.replace(temp,path)
    finally: Path(temp).unlink(missing_ok=True)

@dataclass
class LearningConfig:
    enabled: bool=True
    train_every: int=20
    sample_mode: str='recent'
    min_train: int=1
    min_validation: int=1
    validation_fraction: float=.2
    spatial_block_degrees: float=1
    validation_buffer_km: float=1
    max_train_samples: int=512
    max_validation_samples: int=256
    batch_size: int=1
    steps: int=32
    learning_rate: float=1e-5
    last_visual_blocks: int=0
    checkpoint_every: int=4
    minimum_improvement_km: float=.01
