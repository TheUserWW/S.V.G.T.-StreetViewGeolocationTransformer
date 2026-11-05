"""Durable, idempotent round queue and model registry in a private runtime folder."""

from contextlib import AbstractContextManager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sqlite3

from .geography import coordinate, distance_km, spatial_split


def now():
    return datetime.now(timezone.utc).isoformat()


class RunLock(AbstractContextManager):
    def __init__(self, root):
        self.path = Path(root) / "runner.lock"
        self.stream = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+b")
        if self.path.stat().st_size == 0:
            self.stream.write(b"0")
            self.stream.flush()
        self.stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.stream.close()
            self.stream = None
            raise RuntimeError("训练会话已运行，请先停止 / Training session already running") from exc
        return self

    def __exit__(self, *args):
        if self.stream:
            self.stream.close()
            self.stream = None


class Store:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / "rounds.sqlite3", timeout=20)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS rounds (
                id TEXT PRIMARY KEY, game_id TEXT NOT NULL, round_number INTEGER NOT NULL,
                status TEXT NOT NULL, images TEXT NOT NULL, headings TEXT NOT NULL,
                image_hash TEXT NOT NULL, prediction TEXT, guessed TEXT,
                latitude REAL, longitude REAL, error_km REAL, split TEXT, label_source TEXT,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                UNIQUE(game_id, round_number));
            CREATE INDEX IF NOT EXISTS rounds_training ON rounds(status,split);
            CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS updates (
                id TEXT PRIMARY KEY, status TEXT NOT NULL, report TEXT NOT NULL, updated_at TEXT NOT NULL);
        """)

    def close(self):
        self.db.close()

    def get(self, key, default=None):
        row = self.db.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set(self, key, value):
        with self.db:
            self.db.execute("INSERT INTO state VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                            (key, json.dumps(value, allow_nan=False)))

    def round(self, game_id, number):
        row = self.db.execute("SELECT * FROM rounds WHERE game_id=? AND round_number=?", (game_id, number)).fetchone()
        return self._decode(row) if row else None

    @staticmethod
    def _decode(row):
        result = dict(row)
        for key in ("images", "headings", "prediction", "guessed"):
            result[key] = json.loads(result[key]) if result[key] is not None else None
        return result

    def observe(self, game_id, number, images, headings):
        if type(number) is not int or number < 1 or not images or len(images) != len(headings):
            raise ValueError("回合或视角数据无效")
        existing = self.round(game_id, number)
        if existing:
            return existing
        paths = [Path(p).resolve() for p in images]
        # Only screenshots captured into this runtime can enter game learning.
        if any(not p.is_relative_to(self.root / "images") or not p.is_file() for p in paths):
            raise ValueError("反馈图片必须来自本模块的街景采集目录")
        digest = hashlib.sha256()
        for path in paths:
            digest.update(hashlib.sha256(path.read_bytes()).digest())
        identity = hashlib.sha256(f"{game_id}:{number}".encode()).hexdigest()[:24]
        timestamp = now()
        with self.db:
            self.db.execute("""INSERT INTO rounds
                (id,game_id,round_number,status,images,headings,image_hash,created_at,updated_at)
                VALUES (?,?,?,'observed',?,?,?,?,?)""",
                            (identity, game_id, number, json.dumps([str(p) for p in paths]),
                             json.dumps(headings, allow_nan=False), digest.hexdigest(), timestamp, timestamp))
        return self.round(game_id, number)

    def predict(self, row_id, prediction):
        coordinate(prediction["latitude"], prediction["longitude"])
        with self.db:
            self.db.execute("UPDATE rounds SET prediction=?,status='predicted',updated_at=? WHERE id=? AND status IN ('observed','predicted')",
                            (json.dumps(prediction, allow_nan=False), now(), row_id))

    def submit(self, row_id, guessed):
        guessed = coordinate(*guessed)
        with self.db:
            cursor = self.db.execute("UPDATE rounds SET guessed=?,status='submitted',updated_at=? WHERE id=? AND status IN ('predicted','submitted')",
                                     (json.dumps(guessed), now(), row_id))
            if cursor.rowcount != 1:
                raise ValueError("只能提交已有预测的回合")

    def complete(self, row_id, truth, actual_guess, *, learning, source="openguessr_completed_round"):
        lat, lon = coordinate(*truth)
        actual_guess = coordinate(*actual_guess)
        row = self.db.execute("SELECT * FROM rounds WHERE id=?", (row_id,)).fetchone()
        if row is None:
            raise ValueError("未知回合")
        row = self._decode(row)
        if row["status"] == "completed":
            if distance_km((row["latitude"], row["longitude"]), (lat, lon)) > .001:
                raise ValueError("同一回合返回冲突真值")
            return row
        if row["status"] != "submitted" or distance_km(row["guessed"], actual_guess) > 1:
            raise ValueError("赛后落点与本次提交不匹配，拒绝训练")
        duplicate = self.db.execute("SELECT id FROM rounds WHERE image_hash=? AND status='completed'", (row["image_hash"],)).fetchone()
        status = "duplicate" if duplicate else "completed"
        split = spatial_split(lat, lon, degrees=learning.spatial_block_degrees, fraction=learning.validation_fraction)
        with self.db:
            self.db.execute("""UPDATE rounds SET latitude=?,longitude=?,error_km=?,split=?,label_source=?,
                status=?,updated_at=? WHERE id=?""",
                            (lat, lon, distance_km(actual_guess, (lat, lon)), split, source, status, now(), row_id))
        return self.round(row["game_id"], row["round_number"])

    def samples(self, split, limit):
        rows = self.db.execute("SELECT * FROM rounds WHERE status='completed' AND split=? ORDER BY created_at,id", (split,)).fetchall()
        if split == "validation":
            rows = rows[:limit]  # Once full, keep the same validation set across updates.
        elif len(rows) > limit:
            recent = max(1, limit // 2)
            older = random.Random(len(rows)).sample(rows[:-recent], limit - recent)
            rows = older + rows[-recent:]
        return [self._decode(row) for row in rows]

    def completed_window(self, offset, size):
        # Completion time is immutable after complete(); this survives restart
        # and never counts observed, submitted, or duplicate screenshots.
        rows=self.db.execute("SELECT * FROM rounds WHERE status='completed' ORDER BY updated_at,id LIMIT ? OFFSET ?",
                             (size,offset)).fetchall()
        return [self._decode(row) for row in rows]

    def counts(self):
        result = {"train": 0, "validation": 0, "completed": 0, "duplicate": 0}
        for row in self.db.execute("SELECT status,split,COUNT(*) AS n FROM rounds GROUP BY status,split"):
            result[row["status"]] = result.get(row["status"], 0) + row["n"]
            if row["status"] == "completed":
                result[row["split"]] += row["n"]
        return result

    def update(self, identity, report):
        with self.db:
            self.db.execute("INSERT INTO updates VALUES (?,?,?,?) ON CONFLICT(id) DO UPDATE SET status=excluded.status,report=excluded.report,updated_at=excluded.updated_at",
                            (identity, report["status"], json.dumps(report, allow_nan=False), now()))

    def finish_training(self, identity, report, count):
        if report["status"] not in ("accepted", "rejected", "skipped"):
            raise ValueError("只能提交已验证完毕的训练结果")
        changes = {"last_training_count": count, "pending_train": None}
        if report["status"] == "accepted":
            if not Path(report["candidate"]).is_file():
                raise ValueError("候选模型尚未保存，不能发布")
            changes["previous_checkpoint"] = self.get("active_checkpoint")
            changes["active_checkpoint"] = report["candidate"]
        # Registry, rollback pointer and queue acknowledgement commit together.
        with self.db:
            self.db.execute("INSERT INTO updates VALUES (?,?,?,?) ON CONFLICT(id) DO UPDATE SET status=excluded.status,report=excluded.report,updated_at=excluded.updated_at",
                            (identity, report["status"], json.dumps(report, allow_nan=False), now()))
            for key, value in changes.items():
                self.db.execute("INSERT INTO state VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                                (key, json.dumps(value, allow_nan=False)))
