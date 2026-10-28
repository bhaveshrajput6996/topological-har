"""SQLite experiment store.

Every experiment run appends rows here; nothing is ever overwritten or
deleted, so the full model evaluation history (SVM run 1, CNN run 2, ...)
stays available for the frontend comparison charts.

Tables
------
experiments          run_id, name, dataset, model, window + TDA params, config JSON
evaluation_log       one row per evaluated *window* (with probabilities JSON)
evaluation_summary   run-level metrics + latency + RAM
confusion_matrix     (true_label, predicted_label, count) per run
training_history     per-epoch loss/accuracy/time
resource_measurements phase-level RSS/GPU measurements
split_meta           per-run split strategy and subject hats (leakage proof)
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    run_id          TEXT PRIMARY KEY,
    experiment_name TEXT,
    dataset         TEXT,
    model_name      TEXT,
    window_duration REAL,
    overlap         REAL,
    window_size     INTEGER,
    step_size       INTEGER,
    sampling_rate   REAL,
    tda_enabled     INTEGER,
    tda_points      INTEGER,
    tda_resolution  TEXT,
    created_at      TEXT,
    git_commit      TEXT,
    config_json     TEXT
);
CREATE TABLE IF NOT EXISTS evaluation_log (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id            TEXT,
    window_id         INTEGER,
    subject_id        TEXT,
    true_label        TEXT,
    predicted_label   TEXT,
    confidence        REAL,
    correct           INTEGER,
    inference_time_ms REAL,
    tda_ms            REAL,
    deep_ms           REAL,
    fusion_ms         REAL,
    ram_usage_mb      REAL,
    probabilities     TEXT,
    timestamp         TEXT
);
CREATE INDEX IF NOT EXISTS idx_eval_log_run ON evaluation_log(run_id);
CREATE TABLE IF NOT EXISTS evaluation_summary (
    run_id            TEXT PRIMARY KEY,
    model_name        TEXT,
    accuracy          REAL,
    precision_macro   REAL,
    recall_macro      REAL,
    f1_macro          REAL,
    precision_weighted REAL,
    recall_weighted   REAL,
    f1_weighted       REAL,
    avg_inference_ms  REAL,
    p95_inference_ms  REAL,
    peak_ram_mb       REAL,
    training_time_s   REAL,
    num_windows       INTEGER,
    fold              INTEGER,
    train_subjects    TEXT,
    val_subjects      TEXT,
    test_subjects     TEXT,
    created_at        TEXT
);
CREATE TABLE IF NOT EXISTS confusion_matrix (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT,
    true_label      TEXT,
    predicted_label TEXT,
    count           INTEGER
);
CREATE TABLE IF NOT EXISTS training_history (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT,
    epoch         INTEGER,
    train_loss    REAL,
    val_loss      REAL,
    train_accuracy REAL,
    val_accuracy  REAL,
    epoch_time_s  REAL
);
CREATE TABLE IF NOT EXISTS resource_measurements (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id    TEXT,
    phase     TEXT,
    metric    TEXT,
    value     REAL,
    timestamp TEXT
);
CREATE TABLE IF NOT EXISTS split_meta (
    run_id         TEXT,
    fold           INTEGER,
    split_strategy TEXT,
    train_subjects TEXT,
    val_subjects   TEXT,
    test_subjects  TEXT,
    leakage_passed INTEGER,
    created_at     TEXT
);
"""


@dataclass
class RunSummaryRow:
    run_id: str
    experiment_name: str
    dataset: str
    model_name: str
    accuracy: float
    f1_macro: float
    f1_weighted: float
    avg_inference_ms: float
    training_time_s: float
    peak_ram_mb: float
    num_windows: int
    fold: int
    created_at: str


class ExperimentStore:
    """SQLite wrapper (single-process; thread-guarded)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(str(self.path), check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        with self._connection:
            self._connection.executescript(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    # -- run ids ---------------------------------------------------------
    def next_run_id(self, prefix: str = "HAR") -> str:
        year = "2026"
        like = f"{prefix}-{year}-%"
        with self._lock:
            cur = self._connection.execute(
                "SELECT COUNT(*) FROM experiments WHERE run_id LIKE ?", (like,)
            )
            count = cur.fetchone()[0]
        return f"{prefix}-{year}-{count + 1:03d}"

    def run_id_exists(self, run_id: str) -> bool:
        with self._lock:
            cur = self._connection.execute(
                "SELECT 1 FROM experiments WHERE run_id = ?", (run_id,)
            )
            return cur.fetchone() is not None

    # -- writes ----------------------------------------------------------
    def insert_experiment(self, record: dict[str, Any]) -> None:
        if self.run_id_exists(record["run_id"]):
            raise ValueError(
                f"run_id {record['run_id']} already exists; experiments are "
                "never overwritten -- generate a new run id"
            )
        record = dict(record)
        record.setdefault("git_commit", None)
        record["config_json"] = json.dumps(record.get("config_json"), ensure_ascii=False)
        record["created_at"] = record.get("created_at") or _utcnow()
        with self._lock, self._connection as conn:
            conn.execute(
                """
                INSERT INTO experiments (
                    run_id, experiment_name, dataset, model_name, window_duration,
                    overlap, window_size, step_size, sampling_rate, tda_enabled,
                    tda_points, tda_resolution, created_at, git_commit, config_json
                ) VALUES (
                    :run_id, :experiment_name, :dataset, :model_name, :window_duration,
                    :overlap, :window_size, :step_size, :sampling_rate, :tda_enabled,
                    :tda_points, :tda_resolution, :created_at, :git_commit, :config_json
                )
                """,
                record,
            )

    def insert_window_evaluations(self, run_id: str, records: list[dict[str, Any]]) -> None:
        rows = [
            (
                run_id,
                r["window_id"],
                r.get("subject_id"),
                r.get("true_label"),
                r.get("predicted_label"),
                r.get("confidence"),
                None if r.get("correct") is None else int(r.get("correct")),
                r.get("inference_time_ms"),
                r.get("tda_ms"),
                r.get("deep_ms"),
                r.get("fusion_ms"),
                r.get("ram_usage_mb"),
                json.dumps(r.get("probabilities"), ensure_ascii=False),
                r.get("timestamp") or _utcnow(),
            )
            for r in records
        ]
        with self._lock, self._connection as conn:
            conn.executemany(
                """
                INSERT INTO evaluation_log (
                    run_id, window_id, subject_id, true_label, predicted_label,
                    confidence, correct, inference_time_ms, tda_ms, deep_ms,
                    fusion_ms, ram_usage_mb, probabilities, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def insert_summary(self, record: dict[str, Any]) -> None:
        record = dict(record)
        record.setdefault("created_at", _utcnow())
        for key in ("train_subjects", "val_subjects", "test_subjects"):
            if isinstance(record.get(key), (list, tuple)):
                record[key] = json.dumps(record[key])
        with self._lock, self._connection as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO evaluation_summary (
                    run_id, model_name, accuracy, precision_macro, recall_macro,
                    f1_macro, precision_weighted, recall_weighted, f1_weighted,
                    avg_inference_ms, p95_inference_ms, peak_ram_mb, training_time_s,
                    num_windows, fold, train_subjects, val_subjects, test_subjects, created_at
                ) VALUES (
                    :run_id, :model_name, :accuracy, :precision_macro, :recall_macro,
                    :f1_macro, :precision_weighted, :recall_weighted, :f1_weighted,
                    :avg_inference_ms, :p95_inference_ms, :peak_ram_mb, :training_time_s,
                    :num_windows, :fold, :train_subjects, :val_subjects, :test_subjects, :created_at
                )
                """,
                record,
            )

    def insert_confusion_matrix(self, run_id: str, rows: list[dict[str, Any]]) -> None:
        data = [
            (run_id, r["true_label"], r["predicted_label"], int(r["count"])) for r in rows
        ]
        with self._lock, self._connection as conn:
            conn.executemany(
                "INSERT INTO confusion_matrix (run_id, true_label, predicted_label, count) "
                "VALUES (?, ?, ?, ?)",
                data,
            )

    def insert_training_history(self, run_id: str, history: list[dict[str, Any]]) -> None:
        data = [
            (
                run_id,
                h["epoch"],
                h["train_loss"],
                h["val_loss"],
                h["train_accuracy"],
                h["val_accuracy"],
                h["epoch_time_s"],
            )
            for h in history
        ]
        with self._lock, self._connection as conn:
            conn.executemany(
                """
                INSERT INTO training_history (
                    run_id, epoch, train_loss, val_loss, train_accuracy, val_accuracy, epoch_time_s
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                data,
            )

    def insert_resource_measurements(
        self, run_id: str, measurements: dict[str, Any]
    ) -> None:
        now = _utcnow()
        rows = []
        for phase, payload in measurements.items():
            if phase == "gpu" and isinstance(payload, dict):
                for metric, value in payload.items():
                    rows.append((run_id, "gpu", metric, float(value) if value is not None else None, now))
                continue
            if not isinstance(payload, dict):
                continue
            for metric, value in payload.items():
                try:
                    rows.append((run_id, phase, metric, float(value) if value is not None else None, now))
                except (TypeError, ValueError):
                    continue
        with self._lock, self._connection as conn:
            conn.executemany(
                "INSERT INTO resource_measurements (run_id, phase, metric, value, timestamp) "
                "VALUES (?, ?, ?, ?, ?)",
                rows,
            )

    def insert_split_meta(
        self,
        run_id: str,
        fold: int,
        strategy: dict[str, Any],
        passed: bool = True,
    ) -> None:
        with self._lock, self._connection as conn:
            conn.execute(
                """
                INSERT INTO split_meta (
                    run_id, fold, split_strategy, train_subjects, val_subjects,
                    test_subjects, leakage_passed, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    fold,
                    json.dumps(strategy, ensure_ascii=False),
                    json.dumps([s for s in strategy.get("train_subjects", [])]),
                    json.dumps([s for s in strategy.get("val_subjects", [])]),
                    json.dumps([s for s in strategy.get("test_subjects", [])]),
                    int(passed),
                    _utcnow(),
                ),
            )

    # -- reads -----------------------------------------------------------
    def list_experiments(self) -> list[dict[str, Any]]:
        with self._lock:
            cur = self._connection.execute(
                "SELECT * FROM experiments ORDER BY created_at"
            )
            return [dict(r) for r in cur.fetchall()]

    def get_experiment(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            cur = self._connection.execute(
                "SELECT * FROM experiments WHERE run_id = ?", (run_id,)
            )
            row = cur.fetchone()
            if row is not None and row["config_json"]:
                row = dict(row)
                row["config_json"] = json.loads(row["config_json"])
            return dict(row) if row else None

    def get_summary(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            cur = self._connection.execute(
                "SELECT * FROM evaluation_summary WHERE run_id = ?", (run_id,)
            )
            row = cur.fetchone()
            if row:
                row = dict(row)
                for key in ("train_subjects", "val_subjects", "test_subjects"):
                    if row.get(key):
                        try:
                            row[key] = json.loads(row[key])
                        except (json.JSONDecodeError, TypeError):
                            pass
                return row
            return None

    def get_confusion_matrix(self, run_id: str) -> list[dict[str, Any]]:
        with self._lock:
            cur = self._connection.execute(
                "SELECT true_label, predicted_label, count FROM confusion_matrix "
                "WHERE run_id = ? ORDER BY true_label, predicted_label",
                (run_id,),
            )
            return [dict(r) for r in cur.fetchall()]

    def get_training_history(self, run_id: str) -> list[dict[str, Any]]:
        with self._lock:
            cur = self._connection.execute(
                "SELECT epoch, train_loss, val_loss, train_accuracy, val_accuracy, epoch_time_s "
                "FROM training_history WHERE run_id = ? ORDER BY epoch",
                (run_id,),
            )
            return [dict(r) for r in cur.fetchall()]

    def get_window_history(
        self,
        run_id: str,
        limit: int = 200,
        subject_id: str | None = None,
        correct: bool | None = None,
    ) -> list[dict[str, Any]]:
        query = (
            "SELECT id, window_id, subject_id, true_label, predicted_label, confidence, "
            "correct, inference_time_ms, tda_ms, deep_ms, fusion_ms, timestamp "
            "FROM evaluation_log WHERE run_id = ?"
        )
        params: list[Any] = [run_id]
        if subject_id is not None:
            query += " AND subject_id = ?"
            params.append(subject_id)
        if correct is not None:
            query += " AND correct = ?"
            params.append(int(correct))
        query += " ORDER BY id LIMIT ?"
        params.append(limit)
        with self._lock:
            cur = self._connection.execute(query, params)
            return [dict(r) for r in cur.fetchall()]

    def get_resource_measurements(self, run_id: str) -> list[dict[str, Any]]:
        with self._lock:
            cur = self._connection.execute(
                "SELECT phase, metric, value, timestamp FROM resource_measurements "
                "WHERE run_id = ?",
                (run_id,),
            )
            return [dict(r) for r in cur.fetchall()]

    def get_split_meta(self, run_id: str) -> list[dict[str, Any]]:
        with self._lock:
            cur = self._connection.execute(
                "SELECT fold, split_strategy, train_subjects, val_subjects, test_subjects, "
                "leakage_passed FROM split_meta WHERE run_id = ?",
                (run_id,),
            )
            rows = [dict(r) for r in cur.fetchall()]
            for row in rows:
                for key in ("train_subjects", "val_subjects", "test_subjects"):
                    if row.get(key):
                        try:
                            row[key] = json.loads(row[key])
                        except (json.JSONDecodeError, TypeError):
                            pass
                if row.get("split_strategy"):
                    try:
                        row["split_strategy"] = json.loads(row["split_strategy"])
                    except (json.JSONDecodeError, TypeError):
                        pass
            return rows

    def summary_rows(self) -> list[RunSummaryRow]:
        with self._lock:
            cur = self._connection.execute(
                """
                SELECT e.run_id, e.experiment_name, e.dataset, e.model_name,
                       COALESCE(s.accuracy, -1)   AS accuracy,
                       COALESCE(s.f1_macro, -1)   AS f1_macro,
                       COALESCE(s.f1_weighted, -1) AS f1_weighted,
                       COALESCE(s.avg_inference_ms, -1) AS avg_inference_ms,
                       COALESCE(s.training_time_s, -1)  AS training_time_s,
                       COALESCE(s.peak_ram_mb, -1)      AS peak_ram_mb,
                       COALESCE(s.num_windows, 0)       AS num_windows,
                       COALESCE(s.fold, -1)             AS fold,
                       e.created_at
                FROM experiments e
                LEFT JOIN evaluation_summary s ON s.run_id = e.run_id
                ORDER BY e.created_at
                """
            )
            return [
                RunSummaryRow(
                    run_id=r["run_id"],
                    experiment_name=r["experiment_name"],
                    dataset=r["dataset"],
                    model_name=r["model_name"],
                    accuracy=r["accuracy"],
                    f1_macro=r["f1_macro"],
                    f1_weighted=r["f1_weighted"],
                    avg_inference_ms=r["avg_inference_ms"],
                    training_time_s=r["training_time_s"],
                    peak_ram_mb=r["peak_ram_mb"],
                    num_windows=r["num_windows"],
                    fold=r["fold"],
                    created_at=r["created_at"],
                )
                for r in cur.fetchall()
            ]


def _utcnow() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def new_correlation_id() -> str:
    return uuid.uuid4().hex[:12]
