"""Deterministic batch positions and graceful stop requests for resumable runs."""

import hashlib
import signal
from pathlib import Path

import torch


def manifest_fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ignore_worker_interrupts(worker_id):
    """Let the parent finish/save on Ctrl+C instead of killing data workers."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, signal.SIG_IGN)


class EpochBatchSampler:
    """Rebuild an epoch's order without reading already-completed images."""

    def __init__(self, length, batch_size, seed, epoch, start_batch=0):
        self.length, self.batch_size = length, batch_size
        self.seed, self.epoch, self.start_batch = seed, epoch, start_batch
        self.total_batches = (length + batch_size - 1) // batch_size
        if not 0 <= start_batch <= self.total_batches:
            raise ValueError("checkpoint batch position is outside this epoch")

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.randperm(self.length, generator=generator).tolist()
        for batch in range(self.start_batch, self.total_batches):
            yield indices[batch * self.batch_size:(batch + 1) * self.batch_size]

    def __len__(self):
        return self.total_batches - self.start_batch


class AlternatingBatchSampler:
    """Full then masked views of the same batch; start_batch counts optimizer updates."""
    def __init__(self, length, batch_size, seed, epoch, start_batch=0):
        total = 2 * ((length + batch_size - 1) // batch_size)
        if not 0 <= start_batch <= total:
            raise ValueError("checkpoint batch position is outside this alternating epoch")
        self.source = EpochBatchSampler(length, batch_size, seed, epoch, start_batch // 2)
        self.start_phase, self.remaining = start_batch % 2, total - start_batch

    def __iter__(self):
        for number, batch in enumerate(self.source):
            for phase in range(self.start_phase if number == 0 else 0, 2):
                yield [(index, phase) for index in batch]

    def __len__(self):
        return self.remaining


class StopRequest:
    """First signal waits for a committed optimizer step, second signal aborts."""

    def __init__(self, stop_file=None):
        self._requested = False
        self.stop_file = Path(stop_file) if stop_file else None
        self.previous = {}

    @property
    def requested(self):
        return self._requested or (self.stop_file is not None and self.stop_file.exists())

    @requested.setter
    def requested(self, value):
        self._requested = bool(value)

    def _handle(self, signum, frame):
        if self.requested:
            raise KeyboardInterrupt
        self.requested = True
        print("Stop requested; finishing this batch and saving the resume checkpoint...", flush=True)

    def __enter__(self):
        for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
            sig = getattr(signal, name, None)
            if sig is not None:
                self.previous[sig] = signal.signal(sig, self._handle)
        return self

    def __exit__(self, *args):
        for sig, handler in self.previous.items():
            signal.signal(sig, handler)
