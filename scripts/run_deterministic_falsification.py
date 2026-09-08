#!/usr/bin/env python3
"""Mechanically execute one frozen falsification contract from frozen labels."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from theory_discovery import TheoryDiscoveryProject
from theory_discovery.frozen_execution import evaluate_falsification_specification


def load_labels(run: dict[str, object], root: Path) -> dict[str, str]:
    artifact_path = root / str(run["output_artifact"])
    if hashlib.sha256(artifact_path.read_bytes()).hexdigest() != run["output_hash"]:
        raise RuntimeError("measurement artifact hash has changed: " + str(artifact_path))
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if payload.get("measurement_specification_id") != run["measurement_specification_id"]:
        raise RuntimeError("measurement artifact has the wrong specification ID")
    if payload.get("specification_hash") != run["specification_hash"]:
        raise RuntimeError("measurement artifact has the wrong frozen hash")
    values: dict[str, str] = {}
    for item in payload.get("labels", []):
        if not isinstance(item, dict):
            raise RuntimeError("measurement artifact contains a non-object label")
        record_id, label = item.get("record_id"), item.get("label")
        if not isinstance(record_id, str) or label not in {"PRESENT", "ABSENT", "UNCERTAIN"}:
            raise RuntimeError("measurement artifact has an invalid label")
        values[record_id] = str(label)
    return values


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: run_deterministic_falsification.py PROJECT_DIR")
    project = TheoryDiscoveryProject.open(sys.argv[1])
    packet = project.task_packet()
    if packet["action"] != "execute_falsification_tests":
        raise SystemExit("falsification execution is not currently requested: " + repr(packet["action"]))
    specifications = packet["falsification_specifications"]
    runs = packet["measurement_runs"]
    if not isinstance(specifications, list) or len(specifications) != 1 or not isinstance(runs, list):
        raise SystemExit("execution packet must contain one specification and its measurement runs")
    specification_id = specifications[0].get("id") if isinstance(specifications[0], dict) else None
    specification = next(
        (item for item in project.state.falsification_specifications if item.id == specification_id),
        None,
    )
    if specification is None:
        raise SystemExit("execution packet specification is not present in project state")
    labels = {
        str(run["measurement_specification_id"]): load_labels(run, project.root.parent)
        for run in runs
        if isinstance(run, dict)
    }
    status, summary, audit = evaluate_falsification_specification(specification, labels)
    prediction_id = specification.prediction_id
    artifact_path = Path(str(packet["execution_artifact_path"]))
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        "# Frozen deterministic falsification execution receipt.\n"
        f"SPECIFICATION_ID = {specification.id!r}\n"
        f"SPECIFICATION_HASH = {specification.specification_hash!r}\n"
        f"MEASUREMENT_RUN_HASHES = {[run['output_hash'] for run in runs if isinstance(run, dict)]!r}\n"
        f"AUDIT_COUNTS = {audit!r}\n",
        encoding="utf-8",
    )
    result = {
        "id": f"validation_{prediction_id}",
        "prediction_id": prediction_id,
        "status": status,
        "summary": summary,
        "evidence": [],
        "falsification_specification_id": specification.id,
        "specification_hash": specification.specification_hash,
        "execution_artifact": artifact_path.relative_to(project.root.parent).as_posix(),
    }
    results_path = Path(str(packet["execution_results_path"]))
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        json.dumps({"validation_results": [result]}, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(project.submit({"validation_results": [result]}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
