"""Stage 1: constant-comparative open coding and concept memoing."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .models import Concept, GroundedTheoryState, OpenCodingJudgment
from .validation import (
    by_id,
    choice,
    comparison,
    dedupe_evidence,
    evidence_list,
    expected_records,
    extend_evidence,
    next_id,
    object_list,
    optional_text,
    require_exact_fields,
    text,
)


OPEN_CODING_COVERAGE_ACTION = "open_coding_coverage_audit"
_JUDGMENTS_REQUIRING_CONCEPT = {
    "SUPPORTS_EXISTING", "VARIATION", "BOUNDARY",
}
_JUDGMENTS_REQUIRING_EVIDENCE = {
    "SUPPORTS_EXISTING", "VARIATION", "BOUNDARY", "POSSIBLE_NEW",
}
_JUDGMENT_VALUES = _JUDGMENTS_REQUIRING_EVIDENCE | {"NO_RELEVANT_MECHANISM"}


def apply_open_coding(
    state: GroundedTheoryState, payload: dict[str, Any], action: dict[str, Any]
) -> dict[str, Any]:
    expected = expected_records(payload, action)
    judgments = apply_open_coding_judgments(
        state, payload, expected, required=False
    )
    updates = object_list(payload, "concept_updates")
    memo_updates = object_list(payload, "memo_updates", allow_empty=True)
    concepts_created = apply_concept_updates(
        state, updates, allowed_record_ids=set(expected)
    )
    from .memoing import apply_memo_updates

    apply_memo_updates(state, memo_updates, allowed_record_ids=set(expected))
    state.open_coded_record_ids.extend(expected)
    return {
        "record_ids": expected,
        "concepts_created": concepts_created,
        "concept_updates": len(updates),
        "memo_updates": len(memo_updates),
        "record_judgments": len(judgments),
    }


def apply_open_coding_coverage_audit(
    state: GroundedTheoryState, payload: dict[str, Any], action: dict[str, Any]
) -> dict[str, Any]:
    """Persist one exact, reviewable open-coding judgment for every record.

    Unlike the historic batch-level extraction pass, this action cannot mark a
    record complete merely by echoing its ID.  It is used both to repair
    existing projects and as a gate before relational analysis.
    """
    expected = expected_records(payload, action)
    judgments = apply_open_coding_judgments(state, payload, expected, required=True)
    return {"record_ids": expected, "record_judgments": len(judgments)}


def apply_open_coding_judgments(
    state: GroundedTheoryState,
    payload: dict[str, Any],
    expected: list[str],
    *,
    required: bool,
) -> list[OpenCodingJudgment]:
    """Validate and retain a one-to-one record-level coding decision.

    Older accepted Stage-1 payloads did not contain this field.  They remain
    readable, but their records are routed through the required coverage audit
    before any further Stage-2 work can occur.
    """
    if "record_judgments" not in payload:
        if required:
            raise ValueError("record_judgments is required for coverage audit")
        return []
    raw_judgments = object_list(payload, "record_judgments")
    received_ids = [text(item, "record_id") for item in raw_judgments]
    if received_ids != expected:
        raise ValueError(
            "record_judgments must contain exactly one judgment for every requested record in order"
        )
    known = {item.record_id for item in state.open_coding_judgments}
    if any(record_id in known for record_id in received_ids):
        raise ValueError("a record already has a persisted open-coding judgment")

    allowed_ids = set(expected)
    concepts = {item.id: item for item in state.concepts}
    judgments: list[OpenCodingJudgment] = []
    for raw in raw_judgments:
        disposition = choice(raw, "disposition", _JUDGMENT_VALUES)
        require_exact_fields(
            raw,
            required={"record_id", "disposition", "rationale"},
            optional={"concept_id", "evidence", "candidate_label"},
            label="record judgment",
        )
        record_id = text(raw, "record_id")
        rationale = text(raw, "rationale")
        if len(rationale.split()) > 24:
            raise ValueError("record judgment rationale exceeds the 24-word limit")
        concept_id = optional_text(raw, "concept_id")
        candidate_label = optional_text(raw, "candidate_label")
        evidence = None
        if disposition in _JUDGMENTS_REQUIRING_EVIDENCE:
            evidence_raw = raw.get("evidence")
            if not isinstance(evidence_raw, dict):
                raise ValueError(f"{disposition} judgment requires one evidence object")
            evidence = evidence_list(
                {"evidence": [evidence_raw]}, state, allowed_record_ids=allowed_ids
            )[0]
            if evidence.record_id != record_id:
                raise ValueError(
                    f"record judgment {record_id} evidence cites {evidence.record_id}; "
                    f"it must cite {record_id} and quote only that record"
                )
        elif raw.get("evidence") is not None:
            raise ValueError("NO_RELEVANT_MECHANISM judgment must not include evidence")

        if disposition in _JUDGMENTS_REQUIRING_CONCEPT:
            if concept_id not in concepts:
                raise ValueError("record judgment must cite an existing concept_id")
            if candidate_label is not None:
                raise ValueError("existing-concept judgment must not include candidate_label")
        elif disposition == "POSSIBLE_NEW":
            if concept_id is not None or candidate_label is None:
                raise ValueError("POSSIBLE_NEW judgment requires candidate_label and no concept_id")
        elif concept_id is not None or candidate_label is not None:
            raise ValueError("NO_RELEVANT_MECHANISM judgment cannot name a concept or candidate")

        judgment = OpenCodingJudgment(
            record_id=record_id,
            disposition=disposition,  # type: ignore[arg-type]
            rationale=rationale,
            concept_id=concept_id,
            evidence=evidence,
            candidate_label=candidate_label,
        )
        judgments.append(judgment)
        if evidence is not None and concept_id is not None:
            if disposition == "BOUNDARY":
                extend_evidence(concepts[concept_id].negative_or_boundary_cases, [evidence])
            else:
                extend_evidence(concepts[concept_id].evidence, [evidence])
            concepts[concept_id].last_modified_revision = state.revision + 1
    state.open_coding_judgments.extend(judgments)
    return judgments


def apply_concept_updates(
    state: GroundedTheoryState,
    updates: list[dict[str, Any]],
    *,
    allowed_record_ids: set[str] | None,
) -> int:
    """Apply constant-comparative concept work from either Stage 1 or Stage 2."""
    before = len(state.concepts)
    for update in updates:
        decision = comparison(update)
        evidence = evidence_list(update, state, allowed_record_ids=allowed_record_ids)
        if decision == "NEW":
            _create_concept(state, update, evidence)
            continue
        _update_concept(state, update, evidence, decision)
    return len(state.concepts) - before


def _create_concept(
    state: GroundedTheoryState, update: dict[str, Any], evidence: list[Any]
) -> None:
    if optional_text(update, "existing_concept_id"):
        raise ValueError("NEW concept must not name existing_concept_id")
    concept = Concept(
        id=next_id(state.concepts, "concept"),
        label=text(update, "label"),
        definition=text(update, "definition"),
        evidence=dedupe_evidence(evidence),
        parent_category_id=optional_text(update, "parent_category_id"),
        level=choice(update, "level", {"concept", "category"}, default="concept"),  # type: ignore[arg-type]
        last_modified_revision=state.revision + 1,
    )
    state.concepts.append(concept)
    if concept.parent_category_id is not None:
        _set_parent_category(state, concept, concept.parent_category_id)


def _update_concept(
    state: GroundedTheoryState,
    update: dict[str, Any],
    evidence: list[Any],
    decision: str,
) -> None:
    concept = by_id(state.concepts, text(update, "existing_concept_id"), "concept")
    definition_change = _refine_definition(concept, update, evidence)
    placement_change = _revise_category_placement(state, concept, update)
    if decision == "SAME":
        extend_evidence(concept.evidence, evidence)
        concept.last_modified_revision = state.revision + 1
        if definition_change or placement_change:
            from .memoing import add_analytic_memo

            add_analytic_memo(
                state,
                topic=f"Concept/category revised: {concept.label}",
                observation=(definition_change + placement_change).strip(),
                evidence=evidence,
                comparisons=(
                    "Constant comparison revised the concept definition or category "
                    "placement while retaining its prior state in the audit trail."
                ),
                concept_ids=[concept.id],
            )
        return
    if decision == "VARIATION":
        variation = text(update, "variation")
        concept.variations.append(
            {"description": variation, "evidence": [asdict(item) for item in evidence]}
        )
        extend_evidence(concept.evidence, evidence)
        concept.last_modified_revision = state.revision + 1
        from .memoing import add_analytic_memo

        add_analytic_memo(
            state,
            topic=f"Concept variation: {concept.label}",
            observation=variation + definition_change + placement_change,
            evidence=evidence,
            comparisons="Constant comparison retained the concept while recording a meaningful variation.",
            concept_ids=[concept.id],
        )
        return
    description = text(update, "contradiction")
    extend_evidence(concept.negative_or_boundary_cases, evidence)
    concept.last_modified_revision = state.revision + 1
    from .memoing import add_analytic_memo, add_negative_case

    add_negative_case(state, "concept", concept.id, description, evidence)
    add_analytic_memo(
        state,
        topic=f"Concept boundary case: {concept.label}",
        observation=description + definition_change + placement_change,
        evidence=evidence,
        comparisons="Constant comparison found evidence that challenges the current concept boundary.",
        concept_ids=[concept.id],
    )


def _refine_definition(
    concept: Concept, update: dict[str, Any], evidence: list[Any]
) -> str:
    revised = optional_text(update, "revised_definition")
    if revised is None or revised == concept.definition:
        return ""
    previous = concept.definition
    concept.definition_revisions.append(
        {
            "previous_definition": previous,
            "revised_definition": revised,
            "evidence": [asdict(item) for item in evidence],
        }
    )
    concept.definition = revised
    return f" Definition refined from {previous!r} to {revised!r}."


def _revise_category_placement(
    state: GroundedTheoryState, concept: Concept, update: dict[str, Any]
) -> str:
    if "parent_category_id" not in update:
        return ""
    parent_id = optional_text(update, "parent_category_id")
    if parent_id == concept.parent_category_id:
        return ""
    previous = concept.parent_category_id
    _set_parent_category(state, concept, parent_id)
    return f" Category placement changed from {previous!r} to {parent_id!r}."


def _set_parent_category(
    state: GroundedTheoryState, concept: Concept, parent_id: str | None
) -> None:
    if parent_id == concept.id:
        raise ValueError("a concept cannot be its own parent category")
    if parent_id is not None:
        parent = by_id(state.concepts, parent_id, "parent category")
        if parent.level != "category":
            raise ValueError("parent_category_id must name a category")
        _ensure_no_category_cycle(state, concept, parent)
    if concept.parent_category_id is not None:
        previous_parent = by_id(state.concepts, concept.parent_category_id, "parent category")
        previous_parent.child_concept_ids = [
            child_id for child_id in previous_parent.child_concept_ids if child_id != concept.id
        ]
    concept.parent_category_id = parent_id
    if parent_id is not None:
        parent = by_id(state.concepts, parent_id, "parent category")
        if concept.id not in parent.child_concept_ids:
            parent.child_concept_ids.append(concept.id)


def _ensure_no_category_cycle(
    state: GroundedTheoryState, concept: Concept, parent: Concept
) -> None:
    current: Concept | None = parent
    while current is not None:
        if current.id == concept.id:
            raise ValueError("parent_category_id would create a category cycle")
        current = (
            by_id(state.concepts, current.parent_category_id, "parent category")
            if current.parent_category_id is not None
            else None
        )
