"""Audited recovery when an isolated, non-terminal stage cannot submit."""

from __future__ import annotations

import json
from typing import Any

from .persistence import EVENTS_FILE
from .validation import remaining_record_ids


SKIPPABLE_ACTIONS = {
    "open_coding",
    "relational_process_analysis",
    "validate_relational_grounding",
}


def skip_failed_stage(project: Any, action: str, reason: str) -> dict[str, Any]:
    """Skip only an uncommitted batch after bounded retries and retain its audit."""
    if action not in SKIPPABLE_ACTIONS:
        raise ValueError(f"action cannot be skipped safely: {action}")
    if project.next_action()["action"] != action:
        raise ValueError(f"cannot skip {action}; project is at {project.next_action()['action']}")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("a non-empty recovery reason is required")

    state = project.state
    if action == "open_coding":
        record_ids = remaining_record_ids(state.records, state.open_coded_record_ids)[
            : state.config.open_coding_batch_size
        ]
        _extend_unique(state.open_coded_record_ids, record_ids)
    elif action == "relational_process_analysis":
        record_ids = remaining_record_ids(state.records, state.relationally_analyzed_record_ids)[
            : state.config.relational_batch_size
        ]
        _require_open_coded(state, record_ids)
        _extend_unique(state.relationally_analyzed_record_ids, record_ids)
    else:
        if state.pending_relational_payload is None:
            raise ValueError("there is no pending relational candidate to discard")
        record_ids = list(state.pending_relational_record_ids)
        _require_open_coded(state, record_ids)
        _extend_unique(state.relationally_analyzed_record_ids, record_ids)
        state.pending_relational_payload = None
        state.pending_relational_record_ids = []
        state.pending_relational_review_results = []
        state.relational_validation_feedback = []
        state.relational_validation_attempts = 0
        state.relational_validation_blocked = False

    if not record_ids:
        raise ValueError(f"{action} has no records to skip")
    event = {
        "action": action,
        "record_ids": record_ids,
        "reason": reason.strip(),
        "candidate_committed": False,
    }
    project._commit("stage_batch_skipped_after_agent_failure", event)
    return {"skipped": event, "status": project.status()}


def requeue_skipped_relational_batches(project: Any) -> dict[str, Any]:
    """Put once-skipped Stage 2 records back after a runner-capacity repair."""
    if project.state.pending_relational_payload is not None:
        raise ValueError("cannot requeue while a relational candidate is pending review")
    outstanding = _outstanding_skipped_relational_ids(project.root / EVENTS_FILE)
    current = set(project.state.relationally_analyzed_record_ids)
    record_ids = [record.id for record in project.state.records if record.id in outstanding and record.id in current]
    if not record_ids:
        raise ValueError("there are no outstanding skipped relational records to requeue")
    selected = set(record_ids)
    project.state.relationally_analyzed_record_ids = [
        record_id
        for record_id in project.state.relationally_analyzed_record_ids
        if record_id not in selected
    ]
    event = {
        "action": "relational_process_analysis",
        "record_ids": record_ids,
        "reason": "Requeued after runner capacity was increased; prior skipped candidates remain uncommitted.",
    }
    project._commit("stage_batch_requeued_after_runner_repair", event)
    return {"requeued": event, "status": project.status()}


def _extend_unique(target: list[str], values: list[str]) -> None:
    known = set(target)
    target.extend(value for value in values if value not in known)


def _require_open_coded(state: Any, record_ids: list[str]) -> None:
    open_coded = set(state.open_coded_record_ids)
    if any(record_id not in open_coded for record_id in record_ids):
        raise ValueError("only open-coded records can be skipped in Stage 2")


def _outstanding_skipped_relational_ids(events_path: Any) -> set[str]:
    skipped: set[str] = set()
    if not events_path.exists():
        return skipped
    for line in events_path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        payload = event.get("payload", {})
        if event.get("type") == "stage_batch_skipped_after_agent_failure" and payload.get("action") == "relational_process_analysis":
            skipped.update(payload.get("record_ids", []))
        elif event.get("type") == "stage_batch_requeued_after_runner_repair":
            skipped.difference_update(payload.get("record_ids", []))
    return skipped
