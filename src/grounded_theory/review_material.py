"""Mechanical corpus retrieval for a single relational grounding review."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .models import Evidence, GroundedTheoryState, Relationship
from .validation import evidence_list, object_list, string_list


def build_relational_review_material(
    state: GroundedTheoryState, candidate: dict[str, Any]
) -> dict[str, Any]:
    """Return claim-specific cohorts and one de-duplicated original-record pool."""
    relation_updates = object_list(candidate, "relationship_updates", allow_empty=True)
    process_updates = object_list(candidate, "process_updates", allow_empty=True)
    candidate_concepts = _candidate_concepts(state, candidate)
    claims: list[dict[str, Any]] = []
    candidate_claims: dict[str, dict[str, Any]] = {}

    for update_index, update in enumerate(relation_updates):
        claim, relation = _relationship_claim(state, update, candidate_claims)
        supporting = _evidence_record_ids(evidence_list(update, state))
        contradictory = _negative_record_ids(state, relation)
        if update.get("comparison") == "CONTRADICTION":
            contradictory.update(supporting)
        claims.append(
            _claim_packet(
                claim_id=f"relationship:{update_index}",
                claim_type="relationship",
                relationship_update_index=update_index,
                claim=claim,
                supporting=supporting | _evidence_record_ids(relation.evidence if relation else []),
                contradictory=contradictory,
                state=state,
                candidate_concepts=candidate_concepts,
            )
        )

    for process_index, process in enumerate(process_updates):
        for edge_index, edge in enumerate(object_list(process, "edges", allow_empty=True)):
            claim = {
                "source_concept_id": edge.get("source_concept_id"),
                "relationship": edge.get("relationship"),
                "target_concept_id": edge.get("target_concept_id"),
            }
            supporting, contradictory = _edge_evidence(state, relation_updates, edge)
            claims.append(
                _claim_packet(
                    claim_id=f"process:{process_index}:edge:{edge_index}",
                    claim_type="process_edge",
                    process_update_index=process_index,
                    edge_index=edge_index,
                    claim=claim,
                    supporting=supporting,
                    contradictory=contradictory,
                    state=state,
                    candidate_concepts=candidate_concepts,
                )
            )

    requested_ids = {
        record_id
        for claim in claims
        for cohort in claim["evidence_sets"].values()
        for record_id in cohort
    }
    return {
        "review_claims": claims,
        "review_records": _review_records(state, requested_ids, claims),
        "retrieval_rule": (
            "Cohorts are derived mechanically from all stored concept and relationship "
            "evidence. Source/target-only membership means the concept was coded in that "
            "record while the other was not; it is potential counterevidence, not a "
            "semantic conclusion by Python."
        ),
    }


def next_relational_review_material(
    state: GroundedTheoryState,
    candidate: dict[str, Any],
    completed_reviews: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Return one unreviewed claim and exactly its original-record cohorts."""
    material = build_relational_review_material(state, candidate)
    completed_ids = {item.get("claim_id") for item in completed_reviews}
    if any(not isinstance(claim_id, str) for claim_id in completed_ids):
        raise ValueError("stored relational review results must name their claim_id")
    if len(completed_ids) != len(completed_reviews):
        raise ValueError("stored relational review results contain duplicate claims")
    for claim in material["review_claims"]:
        if claim["claim_id"] in completed_ids:
            continue
        record_ids = {
            record_id
            for cohort in claim["evidence_sets"].values()
            for record_id in cohort
        }
        return {
            "review_claim": claim,
            "review_records": [
                record for record in material["review_records"] if record["id"] in record_ids
            ],
            "retrieval_rule": material["retrieval_rule"],
        }
    return None


def _candidate_concepts(
    state: GroundedTheoryState, candidate: dict[str, Any]
) -> dict[str, dict[str, str]]:
    """Expose only new candidate concept labels/definitions needed for a claim.

    This mirrors the deterministic next-ID allocation used by Stage 2.  It does
    not interpret the labels; it lets a reviewer read a relation involving a
    concept created in the same staged submission without receiving the whole
    first analyst's payload.
    """
    used = {concept.id for concept in state.concepts}
    contexts: dict[str, dict[str, str]] = {}
    for update in object_list(candidate, "concept_updates", allow_empty=True):
        if update.get("comparison") != "NEW":
            continue
        concept_id = _next_candidate_concept_id(used)
        used.add(concept_id)
        label, definition = update.get("label"), update.get("definition")
        contexts[concept_id] = {
            "id": concept_id,
            "label": label if isinstance(label, str) else "new candidate concept",
            "definition": definition if isinstance(definition, str) else "",
        }
    return contexts


def _next_candidate_concept_id(used: set[str]) -> str:
    index = 1
    while f"concept_{index:03d}" in used:
        index += 1
    return f"concept_{index:03d}"


def _relationship_claim(
    state: GroundedTheoryState,
    update: dict[str, Any],
    candidate_claims: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], Relationship | None]:
    if update.get("comparison") == "NEW":
        claim = {
            "source_concept_id": update.get("source_concept_id"),
            "relationship": update.get("relationship"),
            "target_concept_id": update.get("target_concept_id"),
        }
        candidate_claims[_next_candidate_relation_id(state, candidate_claims)] = claim
        return claim, None
    relation_id = update.get("existing_relation_id")
    relation = next((item for item in state.relationships if item.id == relation_id), None)
    if relation is None:
        claim = candidate_claims.get(str(relation_id))
        if claim is None:
            raise ValueError(f"candidate references unknown relationship: {relation_id}")
        return claim, None
    return (
        {
            "source_concept_id": relation.source_concept_id,
            "relationship": relation.relationship,
            "target_concept_id": relation.target_concept_id,
        },
        relation,
    )


def _next_candidate_relation_id(
    state: GroundedTheoryState, candidate_claims: dict[str, dict[str, Any]]
) -> str:
    used = {relation.id for relation in state.relationships} | set(candidate_claims)
    index = 1
    while f"relation_{index:03d}" in used:
        index += 1
    return f"relation_{index:03d}"


def _edge_evidence(
    state: GroundedTheoryState,
    relation_updates: list[dict[str, Any]],
    edge: dict[str, Any],
) -> tuple[set[str], set[str]]:
    supporting: set[str] = set()
    contradictory: set[str] = set()
    for relation_id in string_list(edge, "supporting_relation_ids", allow_empty=True):
        relation = next(item for item in state.relationships if item.id == relation_id)
        supporting.update(_evidence_record_ids(relation.evidence))
        contradictory.update(_negative_record_ids(state, relation))
    indexes = edge.get("supporting_relationship_update_indexes", [])
    if not isinstance(indexes, list) or any(type(index) is not int for index in indexes):
        raise ValueError("supporting_relationship_update_indexes must be an array of integers")
    for index in indexes:
        if index < 0 or index >= len(relation_updates):
            raise ValueError(f"relationship update index is out of range: {index}")
        update = relation_updates[index]
        evidence = _evidence_record_ids(evidence_list(update, state))
        if update.get("comparison") == "CONTRADICTION":
            contradictory.update(evidence)
        else:
            supporting.update(evidence)
    return supporting, contradictory


def _claim_packet(
    *,
    claim_id: str,
    claim_type: str,
    claim: dict[str, Any],
    supporting: set[str],
    contradictory: set[str],
    state: GroundedTheoryState,
    candidate_concepts: dict[str, dict[str, str]],
    **indexes: int,
) -> dict[str, Any]:
    source_ids = _concept_record_ids(state, claim.get("source_concept_id"))
    target_ids = _concept_record_ids(state, claim.get("target_concept_id"))
    contradictory.update(
        _concept_challenge_record_ids(
            state, claim.get("source_concept_id"), claim.get("target_concept_id")
        )
    )
    return {
        "claim_id": claim_id,
        "claim_type": claim_type,
        **indexes,
        "claim": {
            **claim,
            "source_concept": _concept_context(
                state, claim.get("source_concept_id"), candidate_concepts
            ),
            "target_concept": _concept_context(
                state, claim.get("target_concept_id"), candidate_concepts
            ),
        },
        "evidence_sets": {
            "supporting_records": sorted(supporting),
            "source_without_target_records": sorted(source_ids - target_ids),
            "target_without_source_records": sorted(target_ids - source_ids),
            "explicitly_contradictory_records": sorted(contradictory),
        },
        "cohort_counts": {
            "supporting_records": len(supporting),
            "source_without_target_records": len(source_ids - target_ids),
            "target_without_source_records": len(target_ids - source_ids),
            "explicitly_contradictory_records": len(contradictory),
        },
    }


def _concept_record_ids(state: GroundedTheoryState, concept_id: Any) -> set[str]:
    if not isinstance(concept_id, str):
        return set()
    concept = next((item for item in state.concepts if item.id == concept_id), None)
    return _evidence_record_ids(concept.evidence if concept else [])


def _concept_context(
    state: GroundedTheoryState,
    concept_id: Any,
    candidate_concepts: dict[str, dict[str, str]],
) -> dict[str, str] | None:
    if not isinstance(concept_id, str):
        return None
    concept = next((item for item in state.concepts if item.id == concept_id), None)
    if concept is None:
        return candidate_concepts.get(
            concept_id,
            {"id": concept_id, "label": "new candidate concept", "definition": ""},
        )
    return {"id": concept.id, "label": concept.label, "definition": concept.definition}


def _negative_record_ids(state: GroundedTheoryState, relation: Relationship | None) -> set[str]:
    if relation is None:
        return set()
    records = _evidence_record_ids(relation.negative_cases)
    for negative_case in state.negative_cases:
        if negative_case.target_type == "relationship" and negative_case.target_id == relation.id:
            records.update(_evidence_record_ids(negative_case.evidence))
    return records


def _concept_challenge_record_ids(
    state: GroundedTheoryState, *concept_ids: Any
) -> set[str]:
    """Retrieve recorded concept boundaries without deciding their relevance."""
    ids = {item for item in concept_ids if isinstance(item, str)}
    records: set[str] = set()
    for concept in state.concepts:
        if concept.id in ids:
            records.update(_evidence_record_ids(concept.negative_or_boundary_cases))
    for negative_case in state.negative_cases:
        if negative_case.target_type == "concept" and negative_case.target_id in ids:
            records.update(_evidence_record_ids(negative_case.evidence))
    return records


def _evidence_record_ids(evidence: list[Evidence]) -> set[str]:
    return {item.record_id for item in evidence}


def _review_records(
    state: GroundedTheoryState, record_ids: set[str], claims: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    records = {record.id: record for record in state.records}
    concept_spans: dict[str, dict[str, list[str]]] = {}
    for claim in claims:
        for field in ("source_concept_id", "target_concept_id"):
            concept_id = claim["claim"].get(field)
            if not isinstance(concept_id, str):
                continue
            concept = next((item for item in state.concepts if item.id == concept_id), None)
            if concept is None:
                continue
            for evidence in concept.evidence:
                if evidence.record_id in record_ids:
                    concept_spans.setdefault(evidence.record_id, {}).setdefault(
                        concept_id, []
                    ).append(evidence.text_span)
    return [
        {
            **asdict(records[record_id]),
            "coded_concept_evidence": concept_spans.get(record_id, {}),
        }
        for record_id in sorted(record_ids)
    ]
