#!/usr/bin/env python3
"""Export the held-out workflow state into reviewable stage files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


STAGES = (
    ("01_imported_graph.json", "graph_nodes", "graph_edges"),
    ("02_candidate_relationships.json", "relationships"),
    ("03_candidate_theories.json", "theories"),
    ("04_candidate_predictions.json", "predictions"),
    ("05_falsification_tests.json", "falsification_tests"),
    ("05a_frozen_measurement_specifications.json", "measurement_specifications"),
    ("05_falsification_specifications.json", "falsification_specifications"),
    ("06_frozen_measurement_runs.json", "measurement_runs"),
)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def export(project_dir: Path, output_dir: Path) -> None:
    state_path = project_dir / "state.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    if not state_path.exists():
        write_json(output_dir / "00_run_status.json", {"state_available": False})
        return
    state = json.loads(state_path.read_text(encoding="utf-8"))
    active_measurement_ids = {
        str(item.get("id", ""))
        for item in state.get("measurement_specifications", [])
        if isinstance(item, dict)
    }
    active_contracts = {
        (
            str(item.get("id", "")),
            str(item.get("specification_hash", "")),
        )
        for item in state.get("falsification_specifications", [])
        if isinstance(item, dict)
    }
    active_contract_ids = {contract_id for contract_id, _ in active_contracts}
    active_readiness = [
        item
        for item in state.get("readiness_assessments", [])
        if isinstance(item, dict)
        and (
            (
                item.get("parent_type") == "measurement"
                and item.get("parent_id") in active_measurement_ids
            )
            or (
                item.get("parent_type") == "falsification_specification"
                and item.get("parent_id") in active_contract_ids
            )
        )
    ]
    active_validation_results = [
        item
        for item in state.get("validation_results", [])
        if isinstance(item, dict)
        and (
            str(item.get("falsification_specification_id", "")),
            str(item.get("specification_hash", "")),
        )
        in active_contracts
    ]
    counts = {
        "discovery_documents": len(state.get("discovery_documents", [])),
        "graph_nodes": len(state.get("graph_nodes", [])),
        "graph_edges": len(state.get("graph_edges", [])),
        "candidate_relationships": len(state.get("relationships", [])),
        "theories": len(state.get("theories", [])),
        "predictions": len(state.get("predictions", [])),
        "falsification_tests": len(state.get("falsification_tests", [])),
        "falsification_specifications": len(
            state.get("falsification_specifications", [])
        ),
        "legacy_falsification_specifications": len(
            state.get("legacy_falsification_specifications", [])
        ),
        "measurement_specifications": len(
            state.get("measurement_specifications", [])
        ),
        "measurement_runs": len(state.get("measurement_runs", [])),
        "active_readiness_assessments": len(active_readiness),
        "archived_legacy_readiness_assessments": len(
            state.get("readiness_assessments", [])
        ) - len(active_readiness),
        "active_validation_results": len(active_validation_results),
        "archived_legacy_validation_results": len(
            state.get("validation_results", [])
        ) - len(active_validation_results),
    }
    write_json(
        output_dir / "00_run_status.json",
        {
            "state_available": True,
            "study": state.get("config", {}),
            "schema_version": state.get("schema_version"),
            "revision": state.get("revision"),
            "updated_at": state.get("updated_at"),
            "counts": counts,
            "validation_partition": state.get("validation_data", {}),
            "canonical_state": str(state_path),
            "event_log": str(project_dir / "events.jsonl"),
        },
    )
    for file_name, *keys in STAGES:
        write_json(output_dir / file_name, {key: state.get(key, []) for key in keys})
    write_json(
        output_dir / "05_readiness_assessments.json",
        {"readiness_assessments": active_readiness},
    )
    write_json(
        output_dir / "07_validation_records.json",
        {"validation_results": active_validation_results},
    )
    # These were stale v9 exports.  Legacy objects remain in the canonical
    # state for audit, but must not sit beside active v13 stage results.
    for obsolete_name in (
        "05_legacy_falsification_tests.json",
        "06_validation_records.json",
    ):
        (output_dir / obsolete_name).unlink(missing_ok=True)
    events = project_dir / "events.jsonl"
    if events.exists():
        (output_dir / "07_workflow_events.jsonl").write_text(
            events.read_text(encoding="utf-8"), encoding="utf-8"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("project_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    arguments = parser.parse_args()
    export(arguments.project_dir, arguments.output_dir)
