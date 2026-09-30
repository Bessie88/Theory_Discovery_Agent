"""Conservative, auditable adaptation of the mutable Stage-2 strategy only.

This module intentionally does not judge Grounded Theory evidence or alter a
relation review.  It keeps the frozen Stage-2 prompt separate from a small,
versioned strategy overlay and supplies deterministic bookkeeping around model
diagnosis, editing, blind comparison, and acceptance.
"""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from typing import Any

from .models import GroundedTheoryState, utc_now
from .packets import load_prompt


FAILURE_TYPES = frozenset({
    "cooccurrence_as_relation",
    "missing_intermediate_mechanism",
    "ignored_negative_case",
    "missing_condition",
})
PAIRWISE_DIMENSIONS = (
    "evidence_grounding",
    "directionality",
    "negative_case_handling",
    "conditionality",
    "mechanistic_depth",
    "parsimony_redundancy",
    "unsupported_inference",
    "analytic_sufficiency",
)
FAILURE_EXTRACTION_ACTION = "extract_stage2_methodological_failures"
CRITIC_ACTION = "diagnose_stage2_prompt_strategy"
EDITOR_ACTION = "edit_stage2_prompt_strategy"
VALIDATION_ACTION = "validate_stage2_prompt_strategy"
PAIRWISE_ACTION = "blind_pairwise_stage2_evaluation"


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()


def enabled(state: GroundedTheoryState) -> bool:
    return state.config.prompt_adaptation.enabled and not bool(
        state.analysis_metadata.get("prompt_adaptation_replay")
    )


def initialize_state(state: GroundedTheoryState, frozen_base_prompt: str | None = None) -> None:
    """Create the v1 empty overlay once, preserving the exact frozen base."""
    current = state.prompt_adaptation_state
    if current.get("active_overlay"):
        version = str(current["active_overlay"].get("version", "stage2-strategy-v1"))
        try:
            current.setdefault("next_strategy_version", int(version.rsplit("v", 1)[-1]) + 1)
        except ValueError:
            current.setdefault("next_strategy_version", 2)
        # Older persisted runs had one extraction per reviewed batch and did
        # not store this counter.  Its existing durable audit records provide
        # the exact conservative backfill needed for interval scheduling.
        current.setdefault("skipped_failure_extraction_batches", [])
        current.setdefault(
            "reviewed_stage2_batch_count",
            len(current.get("failure_memory", []))
            + len(current["skipped_failure_extraction_batches"]),
        )
        return
    base = (frozen_base_prompt or load_prompt("gt-02-relational-process.md")).strip()
    state.prompt_adaptation_state = {
        "frozen_base_prompt": base,
        "frozen_base_sha256": canonical_hash(base),
        "active_overlay": _overlay("stage2-strategy-v1", None, []),
        "failure_memory": [],
        "reviewed_stage2_batch_count": 0,
        "skipped_failure_extraction_batches": [],
        "history": [],
        "pending_candidate": None,
        "next_strategy_version": 2,
    }


def _overlay(version: str, parent_version: str | None, rules: list[dict[str, Any]]) -> dict[str, Any]:
    value = {"version": version, "parent_version": parent_version, "rules": rules}
    value["sha256"] = canonical_hash(value)
    return value


def active_overlay(state: GroundedTheoryState) -> dict[str, Any]:
    initialize_state(state)
    return deepcopy(state.prompt_adaptation_state["active_overlay"])


def stage2_prompt(state: GroundedTheoryState) -> tuple[str, dict[str, str]]:
    """Render the frozen base plus the sole mutable strategy overlay."""
    # Replay sandboxes retain an enabled configuration so they render the
    # frozen base and requested overlay, but ``enabled()`` deliberately hides
    # adaptation control actions from their state machine.
    if not state.config.prompt_adaptation.enabled:
        return load_prompt("gt-02-relational-process.md"), {"mode": "frozen_only"}
    initialize_state(state)
    memory = state.prompt_adaptation_state
    overlay = active_overlay(state)
    rules = overlay["rules"]
    if not rules:
        suffix = ""
    else:
        body = "\n".join(f"- {rule['text']}" for rule in rules)
        suffix = (
            "\n\n## Versioned analytical strategy overlay\n"
            "This overlay may guide how to inspect relationships and processes. "
            "It cannot weaken or replace any preceding methodological rule.\n"
            f"{body}"
        )
    return memory["frozen_base_prompt"] + suffix, {
        "mode": "frozen_base_plus_strategy_overlay",
        "frozen_base_sha256": memory["frozen_base_sha256"],
        "strategy_version": overlay["version"],
        "strategy_sha256": overlay["sha256"],
    }


def failure_extraction_schema(batch_id: str) -> dict[str, Any]:
    return {
        "batch_id": batch_id,
        "events": [{
            "failure_type": "|".join(sorted(FAILURE_TYPES)),
            "claim_id": "a supplied reviewed claim ID",
            "instance_count": "positive integer",
            "reason": "evidence-grounded methodological diagnosis",
            "evidence_record_ids": ["record IDs present in the supplied audit"],
        }],
    }


def critic_schema() -> dict[str, Any]:
    return {
        "failure_type": "|".join(sorted(FAILURE_TYPES)),
        "prompt_related": "boolean",
        "diagnosis": "specific evidence-supported diagnosis",
        "evidence": [{"batch_id": "...", "claim_id": "...", "reason": "..."}],
        "recommended_change": "minimal strategy-only change",
        "risk": "possible regression or empty string",
    }


def editor_schema(parent_version: str, failure_type: str) -> dict[str, Any]:
    return {
        "parent_version": parent_version,
        "failure_addressed": failure_type,
        "operation": "add|replace|remove",
        "rule_id": "strategy rule ID; new for add, existing for replace/remove",
        "old_text": "exact existing strategy text for replace/remove; empty for add",
        "new_text": "one concise strategy instruction; empty for remove",
        "rationale": "why this is the smallest useful patch",
        "expected_effect": "expected analytic change",
        "possible_regression": "risk or empty string",
    }


def pairwise_schema() -> dict[str, Any]:
    dimensions = {
        name: {"winner": "A|B|TIE", "rationale": "brief evidence-grounded rationale"}
        for name in PAIRWISE_DIMENSIONS
    }


def normalize_pairwise(payload: dict[str, Any]) -> dict[str, Any]:
    if set(payload) != {"dimensions", "overall", "rationale"} or payload.get("overall") not in {"A", "B", "TIE"}:
        raise ValueError("pairwise evaluation must contain only dimensions, overall, and rationale")
    dimensions = payload.get("dimensions")
    if not isinstance(dimensions, dict) or set(dimensions) != set(PAIRWISE_DIMENSIONS) or not isinstance(payload.get("rationale"), str):
        raise ValueError("pairwise evaluation must score every frozen methodological dimension")
    normalized: dict[str, Any] = {"dimensions": {}, "overall": payload["overall"], "rationale": payload["rationale"]}
    for name in PAIRWISE_DIMENSIONS:
        item = dimensions[name]
        if not isinstance(item, dict) or set(item) != {"winner", "rationale"} or item.get("winner") not in {"A", "B", "TIE"} or not isinstance(item.get("rationale"), str):
            raise ValueError("each pairwise dimension needs an A/B/TIE winner and rationale")
        normalized["dimensions"][name] = deepcopy(item)
    return normalized
    return {
        "dimensions": dimensions,
        "overall": "A|B|TIE",
        "rationale": "which analysis better meets frozen Grounded Theory requirements",
    }


def normalize_failure_extraction(payload: dict[str, Any], expected_batch_id: str, known_claims: set[str], known_records: set[str]) -> dict[str, Any]:
    if payload.get("batch_id") != expected_batch_id or not isinstance(payload.get("events"), list):
        raise ValueError("failure extraction must name its exact batch_id and events list")
    events: list[dict[str, Any]] = []
    for item in payload["events"]:
        if not isinstance(item, dict) or set(item) != {"failure_type", "claim_id", "instance_count", "reason", "evidence_record_ids"}:
            raise ValueError("each failure event must use the frozen failure-extraction schema")
        failure_type, claim_id = item["failure_type"], item["claim_id"]
        if failure_type not in FAILURE_TYPES or claim_id not in known_claims:
            raise ValueError("failure event names an unknown failure type or claim")
        if type(item["instance_count"]) is not int or item["instance_count"] < 1:
            raise ValueError("failure event instance_count must be positive")
        record_ids = item["evidence_record_ids"]
        if not isinstance(item["reason"], str) or not isinstance(record_ids, list) or not record_ids:
            raise ValueError("failure events require a reason and supplied evidence record IDs")
        if any(not isinstance(record_id, str) or record_id not in known_records for record_id in record_ids):
            raise ValueError("failure events may cite only supplied audit records")
        events.append(deepcopy(item))
    return {"batch_id": expected_batch_id, "events": events}


def normalize_critic(payload: dict[str, Any], failure_type: str, known_batches: set[str], known_claims: set[str]) -> dict[str, Any]:
    required = {"failure_type", "prompt_related", "diagnosis", "evidence", "recommended_change", "risk"}
    if set(payload) != required or payload.get("failure_type") != failure_type:
        raise ValueError("critic result must use the frozen critic schema and triggered failure")
    if type(payload["prompt_related"]) is not bool or not all(isinstance(payload[key], str) for key in ("diagnosis", "recommended_change", "risk")):
        raise ValueError("critic result has invalid scalar fields")
    if not isinstance(payload["evidence"], list):
        raise ValueError("critic evidence must be a list")
    for item in payload["evidence"]:
        if not isinstance(item, dict) or set(item) != {"batch_id", "claim_id", "reason"}:
            raise ValueError("critic evidence must contain batch_id, claim_id, and reason")
        if item["batch_id"] not in known_batches or item["claim_id"] not in known_claims or not isinstance(item["reason"], str):
            raise ValueError("critic evidence must cite supplied failure-memory items")
    return deepcopy(payload)


def normalize_patch(payload: dict[str, Any], overlay: dict[str, Any], failure_type: str) -> dict[str, Any]:
    required = {"parent_version", "failure_addressed", "operation", "rule_id", "old_text", "new_text", "rationale", "expected_effect", "possible_regression"}
    if set(payload) != required or payload.get("parent_version") != overlay["version"] or payload.get("failure_addressed") != failure_type:
        raise ValueError("strategy patch has the wrong parent version or failure type")
    operation, rule_id = payload.get("operation"), payload.get("rule_id")
    if operation not in {"add", "replace", "remove"} or not isinstance(rule_id, str) or not rule_id.replace("_", "").isalnum():
        raise ValueError("strategy patch must use one allowlisted operation and a simple rule ID")
    strings = ("old_text", "new_text", "rationale", "expected_effect", "possible_regression")
    if any(not isinstance(payload[key], str) for key in strings):
        raise ValueError("strategy patch text fields must be strings")
    if len(payload["new_text"]) > 1000 or len(payload["old_text"]) > 1000:
        raise ValueError("strategy patch text exceeds the minimal-patch limit")
    existing = {rule["id"]: rule["text"] for rule in overlay["rules"]}
    if operation == "add" and (rule_id in existing or not payload["new_text"].strip() or payload["old_text"]):
        raise ValueError("an add patch needs one new non-empty strategy rule")
    if operation in {"replace", "remove"} and (existing.get(rule_id) != payload["old_text"]):
        raise ValueError("a patch may modify only an exact existing strategy rule")
    if operation == "replace" and not payload["new_text"].strip():
        raise ValueError("a replacement strategy rule must be non-empty")
    if operation == "remove" and payload["new_text"]:
        raise ValueError("a removed strategy rule must not contain replacement text")
    return deepcopy(payload)


def candidate_overlay(
    parent: dict[str, Any], patch: dict[str, Any], *, version_number: int | None = None
) -> dict[str, Any]:
    rules = deepcopy(parent["rules"])
    by_id = {rule["id"]: index for index, rule in enumerate(rules)}
    operation, rule_id = patch["operation"], patch["rule_id"]
    if operation == "add":
        rules.append({"id": rule_id, "text": patch["new_text"], "failure_types": [patch["failure_addressed"]]})
    elif operation == "replace":
        rules[by_id[rule_id]]["text"] = patch["new_text"]
    else:
        del rules[by_id[rule_id]]
    number = version_number if version_number is not None else int(parent["version"].rsplit("v", 1)[-1]) + 1
    return _overlay(f"stage2-strategy-v{number}", parent["version"], rules)


def append_failure_memory(state: GroundedTheoryState, context: dict[str, Any], extraction: dict[str, Any]) -> dict[str, Any]:
    initialize_state(state)
    counts = {name: 0 for name in sorted(FAILURE_TYPES)}
    examples: dict[str, list[dict[str, Any]]] = {name: [] for name in sorted(FAILURE_TYPES)}
    for event in extraction["events"]:
        name = event["failure_type"]
        counts[name] += event["instance_count"]
        examples[name].append({key: event[key] for key in ("claim_id", "reason", "evidence_record_ids")})
    record = {
        "batch_id": context["batch_id"],
        "snapshot_id": context["snapshot_id"],
        "prompt_version": context["prompt_version"],
        "prompt_sha256": context["prompt_sha256"],
        "methodological_failures": counts,
        "examples": examples,
        # These are compact audit references, not duplicate corpus text. The
        # critic receives the small original-record subset on demand from the
        # immutable project records.
        "reviewer_audit": {
            "claims": deepcopy(context.get("audit", {}).get("review_claims", [])),
            "decisions": deepcopy(context.get("audit", {}).get("review_decisions", {})),
            "proposal_summary": deepcopy(context.get("audit", {}).get("proposal_summary", {})),
        },
        "eligible": True,
        "created_at": utc_now(),
    }
    state.prompt_adaptation_state["failure_memory"].append(record)
    state.pending_prompt_adaptation_failure_extraction = None
    return record


def begin_candidate_if_triggered(state: GroundedTheoryState) -> dict[str, Any] | None:
    """Create at most one candidate, only with untouched validation snapshots."""
    initialize_state(state)
    memory = state.prompt_adaptation_state
    if memory.get("pending_candidate") is not None:
        return None
    config = state.config.prompt_adaptation
    # Count only candidates that reached blind replay evaluation. A critic
    # dismissal does not consume the production-run allowance because it did
    # not incur the expensive parent/candidate replay work.
    validated_trials = sum(
        1
        for item in memory.get("history", [])
        if isinstance(item, dict) and item.get("validation_batches")
    )
    if validated_trials >= config.max_online_adaptation_trials:
        return None
    active = memory["active_overlay"]
    completed = [item for item in memory["failure_memory"] if item.get("eligible") and item.get("prompt_version") == active["version"]]
    window = completed[-config.failure_window_batches:]
    if len(window) < config.failure_window_batches:
        return None
    candidates: list[tuple[int, str, list[dict[str, Any]]]] = []
    for failure_type in sorted(FAILURE_TYPES):
        triggered = [item for item in window if item["methodological_failures"].get(failure_type, 0) >= config.failure_min_instances]
        if len(triggered) >= config.failure_trigger_batches:
            candidates.append((sum(item["methodological_failures"][failure_type] for item in triggered), failure_type, triggered))
    if not candidates:
        return None
    _, failure_type, triggered = sorted(candidates, key=lambda item: (-item[0], item[1]))[0]
    trigger_ids = {item["batch_id"] for item in window}
    validation_pool = [item for item in completed if item["batch_id"] not in trigger_ids]
    selected = deterministic_validation_selection(validation_pool, config.validation_batch_count, config.random_seed)
    if len(selected) < config.validation_batch_count:
        return None
    candidate = {
        "id": f"candidate-{active['version']}-{failure_type}",
        "parent_overlay": deepcopy(active),
        "trigger_failure": failure_type,
        "trigger_batches": [item["batch_id"] for item in triggered],
        "trigger_window": [item["batch_id"] for item in window],
        "validation_batches": [item["batch_id"] for item in selected],
        "validation_snapshot_ids": [item["snapshot_id"] for item in selected],
        "phase": "critic",
        "created_at": utc_now(),
    }
    memory["pending_candidate"] = candidate
    return candidate


def deterministic_validation_selection(items: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    ordered = sorted(
        items,
        key=lambda item: (canonical_hash({"seed": seed, "snapshot_id": item["snapshot_id"]}), item["batch_id"]),
    )
    return deepcopy(ordered[:count])


def blind_candidate_label(candidate_id: str, snapshot_id: str, round_index: int, seed: int) -> str:
    """Deterministically randomize A/B order without exposing it to the model."""
    value = canonical_hash({"seed": seed, "candidate": candidate_id, "snapshot": snapshot_id, "round": round_index})
    return "A" if int(value[:8], 16) % 2 == 0 else "B"


def aggregate_validation(validation: list[dict[str, Any]], config: Any) -> dict[str, Any]:
    """Apply the pre-registered, conservative accept/rollback rule."""
    conclusive = [item for item in validation if item.get("overall") in {"candidate", "parent"}]
    candidate_wins = sum(item.get("overall") == "candidate" for item in conclusive)
    parent_wins = sum(item.get("overall") == "parent" for item in conclusive)
    pairwise_overall_votes = {"candidate": 0, "parent": 0, "tie": 0}
    dimension_votes = {name: {"candidate": 0, "parent": 0, "tie": 0} for name in PAIRWISE_DIMENSIONS}
    for item in validation:
        raw_overall_votes = item.get("pairwise_overall_votes")
        if isinstance(raw_overall_votes, dict):
            for side in pairwise_overall_votes:
                pairwise_overall_votes[side] += int(raw_overall_votes.get(side, 0))
        elif item.get("overall") in {"candidate", "parent"}:
            pairwise_overall_votes[item["overall"]] += 1
        else:
            pairwise_overall_votes["tie"] += 1
        for name, votes in item.get("dimension_votes", {}).items():
            if name in dimension_votes:
                for side in dimension_votes[name]:
                    dimension_votes[name][side] += int(votes.get(side, 0))
    grounding_ok = (not config.require_grounding_non_regression or dimension_votes["evidence_grounding"]["candidate"] >= dimension_votes["evidence_grounding"]["parent"])
    unsupported_ok = (not config.require_unsupported_inference_non_regression or dimension_votes["unsupported_inference"]["candidate"] >= dimension_votes["unsupported_inference"]["parent"])
    accepted = (
        len(conclusive) >= 2
        and candidate_wins >= 2
        and candidate_wins > parent_wins
        and pairwise_overall_votes["candidate"] > pairwise_overall_votes["parent"]
        and grounding_ok
        and unsupported_ok
    )
    return {
        "accepted": accepted,
        "conclusive_batches": len(conclusive),
        "candidate_batch_wins": candidate_wins,
        "parent_batch_wins": parent_wins,
        "pairwise_overall_votes": pairwise_overall_votes,
        "grounding_non_regression": grounding_ok,
        "unsupported_inference_non_regression": unsupported_ok,
        "dimension_votes": dimension_votes,
    }
