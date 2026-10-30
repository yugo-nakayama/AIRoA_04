"""Structured JSONL telemetry and deterministic summary metrics."""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


class Telemetry:
    def __init__(self, path: str | Path, run_metadata: dict[str, Any] | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.start = time.monotonic()
        self.records: list[dict[str, Any]] = []
        if run_metadata:
            self.emit("run_start", **run_metadata)

    def emit(self, event: str, **payload: Any) -> dict[str, Any]:
        record = {"t_s": round(time.monotonic() - self.start, 4), "event": event, **payload}
        self.records.append(record)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
        return record


def _json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "value"):
        return value.value
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def summarize(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    outcomes = [r for r in rows if r.get("event") == "object_result"]
    success = [r for r in outcomes if r.get("success")]
    failures = Counter(str(r.get("failure", "none")) for r in outcomes if not r.get("success"))
    classifications = [r for r in outcomes if r.get("true_label") is not None]
    correct = sum(r.get("predicted_label") == r.get("true_label") for r in classifications)
    return {
        "objects_attempted": len(outcomes),
        "objects_succeeded": len(success),
        "success_rate": (len(success) / len(outcomes)) if outcomes else None,
        "classification_accuracy": (correct / len(classifications)) if classifications else None,
        "mean_attempts": (
            sum(int(r.get("attempts", 1)) for r in outcomes) / len(outcomes) if outcomes else None
        ),
        "failure_distribution": dict(sorted(failures.items())),
    }


def write_summary(path: str | Path, records: Iterable[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    summary = {**summarize(records), **extra}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
