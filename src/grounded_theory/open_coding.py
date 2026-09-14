"""Stage 1: constant-comparative open coding and concept memoing."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .models import Concept, GroundedTheoryState
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
    text,
)


def apply_open_coding(
    state: GroundedTheoryState, payload: dict[str, Any], action: dict[str, Any]
) -> dict[str, Any]:
    expected = expected_records(payload, action)
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
    }


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
