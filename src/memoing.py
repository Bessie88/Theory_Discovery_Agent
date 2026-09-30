"""Continuous analytic memoing and negative-case persistence."""

from __future__ import annotations

from typing import Any, Literal

from .models import Evidence, GroundedTheoryState, Memo, NegativeCase
from .validation import (
    by_id,
    dedupe_evidence,
    evidence_list_from_key,
    existing_ids,
    extend_unique,
    next_id,
    object_list,
    choice,
    string_list,
    text,
)


def add_analytic_memo(
    state: GroundedTheoryState,
    *,
    topic: str,
    observation: str,
    evidence: list[Evidence],
    comparisons: str,
    concept_ids: list[str] | None = None,
    relation_ids: list[str] | None = None,
    process_ids: list[str] | None = None,
) -> Memo:
    memo = Memo(
        id=next_id(state.memos, "memo"),
        topic=topic,
        analytic_observation=observation,
        supporting_evidence=dedupe_evidence(evidence),
        comparisons=comparisons,
        tentative_interpretation=observation,
        uncertainties="Requires continued comparison with subsequent records.",
        linked_concept_ids=list(concept_ids or []),
        linked_relation_ids=list(relation_ids or []),
        linked_process_ids=list(process_ids or []),
    )
    state.memos.append(memo)
    return memo


def add_negative_case(
    state: GroundedTheoryState,
    target_type: Literal["concept", "relationship", "process"],
    target_id: str,
    description: str,
    evidence: list[Evidence],
) -> None:
    state.negative_cases.append(
        NegativeCase(
            id=next_id(state.negative_cases, "negative_case"),
            target_type=target_type,
            target_id=target_id,
            description=description,
            evidence=dedupe_evidence(evidence),
        )
    )


def apply_memo_updates(
    state: GroundedTheoryState,
    updates: list[dict[str, Any]],
    *,
    allowed_record_ids: set[str] | None,
) -> None:
    for item in updates:
        decision = choice(item, "decision", {"NEW", "UPDATE"})
        evidence = evidence_list_from_key(
            item, "supporting_evidence", state, allowed_record_ids
        )
        negative = evidence_list_from_key(
            item, "negative_or_contradictory_cases", state, allowed_record_ids
        )
        if not evidence and not negative:
            raise ValueError(
                "every memo needs supporting_evidence or negative_or_contradictory_cases"
            )
        kwargs = {
            "topic": text(item, "topic"),
            "analytic_observation": text(item, "analytic_observation"),
            "supporting_evidence": evidence,
            "comparisons": text(item, "comparisons"),
            "tentative_interpretation": text(item, "tentative_interpretation"),
            "negative_or_contradictory_cases": negative,
            "uncertainties": text(item, "uncertainties"),
            "questions_for_further_analysis": string_list(
                item, "questions_for_further_analysis", allow_empty=True
            ),
            "linked_concept_ids": existing_ids(
                state.concepts,
                string_list(item, "linked_concept_ids", allow_empty=True),
                "concept",
            ),
            "linked_relation_ids": existing_ids(
                state.relationships,
                string_list(item, "linked_relation_ids", allow_empty=True),
                "relationship",
            ),
            "linked_process_ids": existing_ids(
                state.processes,
                string_list(item, "linked_process_ids", allow_empty=True),
                "process",
            ),
        }
        if decision == "NEW":
            memo = Memo(id=next_id(state.memos, "memo"), **kwargs)
            state.memos.append(memo)
        else:
            memo = by_id(state.memos, text(item, "memo_id"), "memo")
            for key, value in kwargs.items():
                setattr(memo, key, value)
        for relation_id in memo.linked_relation_ids:
            relation = by_id(state.relationships, relation_id, "relationship")
            extend_unique(relation.memo_ids, [memo.id])
