"""Stage 2: relation/process comparison and pre-commit grounding review."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .memoing import add_analytic_memo, add_negative_case, apply_memo_updates
from .models import GroundedTheoryState, Process, Relationship
from .open_coding import apply_concept_updates
from .validation import (
    by_id,
    choice,
    comparison,
    dedupe_evidence,
    evidence_list,
    evidence_list_from_key,
    existing_ids,
    expected_records,
    extend_evidence,
    extend_unique,
    next_id,
    object_list,
    optional_choice,
    optional_text,
    string_list,
    text,
)


STATUS_VALUES = {"well_grounded", "tentative", "insufficient_evidence"}
GROUNDING_KINDS = {
    "explicitly_expressed",
    "repeated_comparison",
    "tentative_theoretical_inference",
}


def apply_relational_analysis(
    state: GroundedTheoryState, payload: dict[str, Any], action: dict[str, Any]
) -> dict[str, Any]:
    expected = expected_records(payload, action)
    concept_updates = object_list(payload, "concept_updates", allow_empty=True)
    relation_updates = object_list(payload, "relationship_updates", allow_empty=True)
    process_updates = object_list(payload, "process_updates", allow_empty=True)
    memo_updates = object_list(payload, "memo_updates", allow_empty=True)
    concepts_created = apply_concept_updates(
        state, concept_updates, allowed_record_ids=set(expected)
    )
    relations_before, processes_before = len(state.relationships), len(state.processes)
    for update in relation_updates:
        apply_relationship_update(state, update)
    for update in process_updates:
        apply_process_update(state, update)
    apply_memo_updates(state, memo_updates, allowed_record_ids=None)
    state.relationally_analyzed_record_ids.extend(expected)
    return {
        "record_ids": expected,
        "concepts_created": concepts_created,
        "concept_updates": len(concept_updates),
        "relationships_created": len(state.relationships) - relations_before,
        "processes_created": len(state.processes) - processes_before,
        "relationship_updates": len(relation_updates),
        "process_updates": len(process_updates),
        "memo_updates": len(memo_updates),
    }


def apply_relationship_update(state: GroundedTheoryState, update: dict[str, Any]) -> None:
    decision = comparison(update)
    evidence = evidence_list(update, state)
    grounding_kind, explanation = relationship_grounding(update, evidence)
    if decision == "NEW":
        _create_relationship(state, update, evidence, grounding_kind, explanation)
        return
    relation = by_id(state.relationships, text(update, "existing_relation_id"), "relationship")
    updated_status = optional_choice(update, "status", STATUS_VALUES)
    if updated_status is not None:
        relation.status = updated_status  # type: ignore[assignment]
    if decision == "SAME":
        extend_evidence(relation.evidence, evidence)
    elif decision == "VARIATION":
        variation = text(update, "variation")
        relation.variations.append(
            {
                "description": variation,
                "evidence": [asdict(item) for item in evidence],
                "grounding_kind": grounding_kind,
                "grounding_explanation": explanation,
            }
        )
        extend_evidence(relation.evidence, evidence)
        memo = add_analytic_memo(
            state,
            topic=f"Relationship variation: {relation.id}",
            observation=variation,
            evidence=evidence,
            comparisons=explanation,
            relation_ids=[relation.id],
        )
        extend_unique(relation.memo_ids, [memo.id])
    else:
        description = text(update, "contradiction")
        extend_evidence(relation.negative_cases, evidence)
        add_negative_case(state, "relationship", relation.id, description, evidence)
        memo = add_analytic_memo(
            state,
            topic=f"Relationship challenge: {relation.id}",
            observation=description,
            evidence=evidence,
            comparisons=explanation,
            relation_ids=[relation.id],
        )
        extend_unique(relation.memo_ids, [memo.id])


def _create_relationship(
    state: GroundedTheoryState,
    update: dict[str, Any],
    evidence: list[Any],
    grounding_kind: str,
    explanation: str,
) -> None:
    if optional_text(update, "existing_relation_id"):
        raise ValueError("NEW relationship must not name existing_relation_id")
    source_id, target_id = text(update, "source_concept_id"), text(update, "target_concept_id")
    by_id(state.concepts, source_id, "source concept")
    by_id(state.concepts, target_id, "target concept")
    if source_id == target_id:
        raise ValueError("a relationship must connect two distinct concepts")
    relation = Relationship(
        id=next_id(state.relationships, "relation"),
        source_concept_id=source_id,
        relationship=text(update, "relationship"),
        target_concept_id=target_id,
        evidence=dedupe_evidence(evidence),
        comparative_basis=text(update, "comparative_basis"),
        grounding_kind=grounding_kind,  # type: ignore[arg-type]
        grounding_explanation=explanation,
        status=choice(update, "status", STATUS_VALUES, default="tentative"),  # type: ignore[arg-type]
        memo_ids=existing_ids(
            state.memos, string_list(update, "memo_ids", allow_empty=True), "memo"
        ),
    )
    state.relationships.append(relation)
    memo = add_analytic_memo(
        state,
        topic=f"Proposed relationship: {relation.id}",
        observation=f"{relation.source_concept_id} {relation.relationship} {relation.target_concept_id}.",
        evidence=evidence,
        comparisons=relation.comparative_basis,
        relation_ids=[relation.id],
    )
    extend_unique(relation.memo_ids, [memo.id])


def apply_process_update(state: GroundedTheoryState, update: dict[str, Any]) -> None:
    decision = comparison(update)
    relation_ids = existing_ids(
        state.relationships,
        string_list(update, "supporting_relation_ids", allow_empty=True),
        "relationship",
    )
    record_ids = existing_ids(
        state.records,
        string_list(update, "supporting_record_ids", allow_empty=True),
        "record",
    )
    negative = evidence_list_from_key(update, "negative_cases", state)
    relation_evidence = dedupe_evidence(
        [
            evidence
            for relation in state.relationships
            if relation.id in relation_ids
            for evidence in relation.evidence
        ]
    )
    memo_evidence = dedupe_evidence([*relation_evidence, *negative])
    if decision == "NEW":
        _create_process(state, update, relation_ids, record_ids, negative, memo_evidence)
        return
    process = by_id(state.processes, text(update, "existing_process_id"), "process")
    updated_status = optional_choice(update, "status", STATUS_VALUES)
    if updated_status is not None:
        process.status = updated_status  # type: ignore[assignment]
    if decision == "SAME":
        extend_unique(process.supporting_relation_ids, relation_ids)
        extend_unique(process.supporting_record_ids, record_ids)
    elif decision == "VARIATION":
        variation = text(update, "variation")
        process.alternative_pathways.append(variation)
        extend_unique(process.supporting_relation_ids, relation_ids)
        extend_unique(process.supporting_record_ids, record_ids)
        add_analytic_memo(
            state,
            topic=f"Process variation: {process.label}",
            observation=variation,
            evidence=memo_evidence,
            comparisons="New evidence qualifies an existing process pathway.",
            process_ids=[process.id],
        )
    else:
        description = text(update, "contradiction")
        extend_evidence(process.negative_cases, negative)
        add_negative_case(state, "process", process.id, description, negative)
        add_analytic_memo(
            state,
            topic=f"Process challenge: {process.label}",
            observation=description,
            evidence=memo_evidence,
            comparisons="A negative case challenges the process representation.",
            process_ids=[process.id],
        )


def _create_process(
    state: GroundedTheoryState,
    update: dict[str, Any],
    relation_ids: list[str],
    record_ids: list[str],
    negative: list[Any],
    memo_evidence: list[Any],
) -> None:
    if optional_text(update, "existing_process_id"):
        raise ValueError("NEW process must not name existing_process_id")
    if not relation_ids or not record_ids:
        raise ValueError("a new process needs supporting relations and record IDs")
    process = Process(
        id=next_id(state.processes, "process"),
        label=text(update, "label"),
        description=text(update, "description"),
        conditions=string_list(update, "conditions", allow_empty=True),
        actions_interactions=string_list(update, "actions_interactions", allow_empty=True),
        consequences=string_list(update, "consequences", allow_empty=True),
        subsequent_changes=string_list(update, "subsequent_changes", allow_empty=True),
        alternative_pathways=string_list(update, "alternative_pathways", allow_empty=True),
        supporting_relation_ids=relation_ids,
        supporting_record_ids=record_ids,
        negative_cases=dedupe_evidence(negative),
        status=choice(update, "status", STATUS_VALUES, default="tentative"),  # type: ignore[arg-type]
    )
    state.processes.append(process)
    add_analytic_memo(
        state,
        topic=f"Emerging process: {process.label}",
        observation=process.description,
        evidence=memo_evidence,
        comparisons="The process links cited relationships and records without creating a transitive direct relation.",
        process_ids=[process.id],
    )


def relationship_grounding(update: dict[str, Any], evidence: list[Any]) -> tuple[str, str]:
    kind = choice(update, "grounding_kind", GROUNDING_KINDS)
    explanation = text(update, "grounding_explanation")
    record_count = len({item.record_id for item in evidence})
    if kind == "repeated_comparison" and record_count < 2:
        raise ValueError("repeated_comparison needs evidence from at least two records")
    if kind == "tentative_theoretical_inference":
        if choice(update, "status", STATUS_VALUES, default="tentative") != "tentative":
            raise ValueError("tentative_theoretical_inference must use tentative status")
        if record_count < 2:
            raise ValueError(
                "tentative_theoretical_inference needs comparative evidence from at least two records"
            )
    return kind, explanation


def apply_relational_grounding_validation(
    state: GroundedTheoryState, payload: dict[str, Any]
) -> dict[str, Any]:
    if state.pending_relational_payload is None:
        raise ValueError("there is no pending relational submission to validate")
    verdict = choice(payload, "verdict", {"PASS", "FAIL"})
    issues = string_list(payload, "issues", allow_empty=True)
    if verdict == "PASS" and issues:
        raise ValueError("a PASS grounding review must not include issues")
    if verdict == "FAIL" and not issues:
        raise ValueError("a FAIL grounding review must include specific issues")
    if verdict == "PASS":
        event = apply_relational_analysis(
            state,
            state.pending_relational_payload,
            {"record_ids": list(state.pending_relational_record_ids)},
        )
        state.pending_relational_payload = None
        state.pending_relational_record_ids = []
        state.relational_validation_feedback = []
        state.relational_validation_attempts = 0
        return {"verdict": "PASS", "committed": event}
    rejected_record_ids = list(state.pending_relational_record_ids)
    state.pending_relational_payload = None
    state.pending_relational_record_ids = []
    state.relational_validation_feedback = issues
    state.relational_validation_attempts += 1
    if state.relational_validation_attempts < 2:
        return {
            "verdict": "FAIL",
            "issues": issues,
            "attempt": state.relational_validation_attempts,
            "blocked": False,
        }

    # A second independent rejection is evidence that this candidate batch
    # should not enter the inventory.  Do not let one unsupported proposal
    # prevent analysis of the remaining corpus, and do not commit any other
    # staged objects from this all-or-nothing submission.
    known = set(state.relationally_analyzed_record_ids)
    state.relationally_analyzed_record_ids.extend(
        record_id for record_id in rejected_record_ids if record_id not in known
    )
    state.relational_validation_feedback = []
    state.relational_validation_attempts = 0
    state.relational_validation_blocked = False
    return {
        "verdict": "FAIL",
        "issues": issues,
        "attempt": 2,
        "blocked": False,
        "discarded_candidate_batch": {
            "record_ids": rejected_record_ids,
            "reason": "The same relational batch failed independent grounding review twice; its uncommitted candidate updates were discarded.",
        },
    }
