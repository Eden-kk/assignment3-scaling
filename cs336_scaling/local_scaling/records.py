from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class RunRecord:
    experiment_name: str
    run_name: str
    compute_budget_flops: float | None
    estimated_run_flops: float | None
    final_loss: float | None
    device: str
    status: str
    config: dict[str, Any]
    metrics: dict[str, Any]


def append_run_record(path: str | Path, record: RunRecord) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a") as handle:
        handle.write(json.dumps(asdict(record)) + "\n")


def load_run_records(path: str | Path) -> list[RunRecord]:
    records = []
    with Path(path).open() as handle:
        for line in handle:
            records.append(RunRecord(**json.loads(line)))
    return records
