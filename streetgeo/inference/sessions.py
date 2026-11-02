"""Bounded local round history and owned clipboard captures."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import uuid


class SessionStore:
    def __init__(self, directory, limit=100):
        self.directory = Path(directory).resolve()
        self.captures = self.directory / "captures"
        self.path = self.directory / "history.json"
        self.limit = limit
        try:
            rows = json.loads(self.path.read_text(encoding="utf-8"))
            self.records = [r for r in rows if isinstance(r, dict) and isinstance(r.get("candidates"), list)
                            and isinstance(r.get("images"), list) and isinstance(r.get("round_id"), str)][-limit:]
        except (OSError, ValueError, TypeError):
            self.records = []

    def capture(self, picture):
        self.captures.mkdir(parents=True, exist_ok=True)
        path = self.captures / (uuid.uuid4().hex + ".png")
        picture.convert("RGB").save(path)
        return str(path)

    def save(self, record):
        item = json.loads(json.dumps(record, ensure_ascii=False, allow_nan=False))
        item.setdefault("round_id", uuid.uuid4().hex)
        item.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        self.records = [r for r in self.records if r["round_id"] != item["round_id"]]
        self.records.append(item)
        self.records = self.records[-self.limit:]
        self.directory.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(".tmp")
        temp.write_text(json.dumps(self.records, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        os.replace(temp, self.path)
        return item

    def prune_captures(self, keep=()):
        """Only delete our UUID-named PNGs, never imported user files."""
        retained = {str(Path(p).resolve()) for p in keep}
        retained.update(str(Path(p).resolve()) for r in self.records for p in r["images"])
        if not self.captures.is_dir():
            return
        for path in self.captures.glob("*.png"):
            if (len(path.stem) == 32 and all(c in "0123456789abcdef" for c in path.stem)
                    and path.resolve().is_relative_to(self.captures.resolve()) and str(path.resolve()) not in retained):
                path.unlink(missing_ok=True)
