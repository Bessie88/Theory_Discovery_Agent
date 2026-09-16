"""Deterministic counterevidence retrieval for Stage 2 review packets.

This module deliberately does not decide whether a relationship is true.  It
uses the record-to-concept evidence index to make the review model confront
both the proposal's cited records and records that challenge its coverage.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .models import GroundedTheoryState
from .review_material import build_relational_review_material, next_relational_review_material
from .validation import choice, object_list, string_list, text


REVIEW_DECISIONS = {"RETAIN", "NARROW", "REJECT"}
REVIEW_ASSESSMENTS = {
    "explicitly_expressed",
    "repeated_comparison",
    "tentative_theoretical_inference",
    "unsupported",
}
REVIEW_STATUSES = {"well_grounded", "tentative", "insufficient_evidence"}


def normalize_claim_review(
    payload: dict[str, Any], claim: dict[str, Any]
) -> dict[str, Any]:
    """Validate one model decision and attach its deterministic claim index."""
    if text(payload, "claim_id") != claim.get("claim_id"):
        raise ValueError("review result claim_id does not match the pending review claim")
    normalized = deepcopy(payload)
    claim_type = claim.get("claim_type")
    if claim_type == "relationship":
        normalized["relationship_update_index"] = claim["relationship_update_index"]
        _validate_review(normalized, "relationship")
    elif claim_type == "process_edge":
        normalized["process_update_index"] = claim["process_update_index"]
        normalized["edge_index"] = claim["edge_index"]
        _validate_review(normalized, "process edge")
    else:
        raise ValueError("pending review claim has an unsupported claim_type")
    return normalized


def apply_review_decisions(
    state: GroundedTheoryState, candidate: dict[str, Any], review: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply a completed independent review without letting it add evidence.

    The reviewer can retain a claim, narrow its wording/conditions, or reject
    it.  This returns only the candidate updates that may proceed to the
    ordinary deterministic reference and verbatim-evidence checks.
    """
    if review.get("review_status") != "COMPLETE":
        raise ValueError("review_status must be COMPLETE")
    relation_updates = object_list(candidate, "relationship_updates", allow_empty=True)
    process_updates = object_list(candidate, "process_updates", allow_empty=True)
    relation_reviews = _indexed_reviews(
        object_list(review, "relationship_reviews", allow_empty=True),
        "relationship_update_index",
        set(range(len(relation_updates))),
        "relationship",
    )
    expected_edges = {
        (process_index, edge_index)
        for process_index, process in enumerate(process_updates)
        for edge_index, _ in enumerate(object_list(process, "edges", allow_empty=True))
    }
    edge_reviews = _edge_reviews(
        object_list(review, "process_edge_reviews", allow_empty=True), expected_edges
    )
    candidate_new_relation_ids = _candidate_new_relation_ids(state, relation_updates)
    candidate_new_indexes = {
        relation_id: index for index, relation_id in candidate_new_relation_ids.items()
    }

    filtered = deepcopy(candidate)
    retained_relation_updates: list[dict[str, Any]] = []
    retained_relation_original_indexes: list[int] = []
    rejected_relation_indexes = {
        index
        for index, decision in relation_reviews.items()
        if _decision(decision) == "REJECT"
    }
    dependent_rejections: set[int] = set()
    changed = True
    while changed:
        changed = False
        for index, update in enumerate(relation_updates):
            relation_id = update.get("existing_relation_id")
            parent_index = candidate_new_indexes.get(relation_id)
            if parent_index in rejected_relation_indexes and index not in rejected_relation_indexes:
                rejected_relation_indexes.add(index)
                dependent_rejections.add(index)
                changed = True
    retained_relation_index_map: dict[int, int] = {}
    narrowed_phrases_by_update: dict[int, str] = {}
    narrowed_phrases_by_relation_id: dict[str, str] = {}
    for index, update in enumerate(relation_updates):
        decision = relation_reviews[index]
        if index in rejected_relation_indexes:
            continue
        amended = deepcopy(update)
        _apply_relationship_review(amended, decision)
        if _decision(decision) == "NARROW":
            revised_phrase = text(decision, "revised_relationship")
            narrowed_phrases_by_update[index] = revised_phrase
            relation_id = update.get("existing_relation_id")
            if isinstance(relation_id, str):
                narrowed_phrases_by_relation_id[relation_id] = revised_phrase
        retained_relation_index_map[index] = len(retained_relation_updates)
        retained_relation_updates.append(amended)
        retained_relation_original_indexes.append(index)
    _rebase_candidate_relation_references(
        state,
        retained_relation_updates,
        retained_relation_original_indexes,
        candidate_new_relation_ids,
    )
    filtered["relationship_updates"] = retained_relation_updates

    retained_process_updates: list[dict[str, Any]] = []
    rejected_process_indexes: set[int] = set()
    for process_index, process in enumerate(process_updates):
        amended = deepcopy(process)
        reject_process = False
        edge_statuses: list[str] = []
        for edge_index, edge in enumerate(object_list(amended, "edges", allow_empty=True)):
            decision = edge_reviews[(process_index, edge_index)]
            if _decision(decision) == "REJECT":
                reject_process = True
                break
            update_indexes = edge.get("supporting_relationship_update_indexes", [])
            if any(index in rejected_relation_indexes for index in update_indexes):
                reject_process = True
                break
            edge["supporting_relationship_update_indexes"] = [
                retained_relation_index_map[index] for index in update_indexes
            ]
            narrowed_phrases = {
                narrowed_phrases_by_update[index]
                for index in update_indexes
                if index in narrowed_phrases_by_update
            }
            narrowed_phrases.update(
                narrowed_phrases_by_relation_id[relation_id]
                for relation_id in edge.get("supporting_relation_ids", [])
                if relation_id in narrowed_phrases_by_relation_id
            )
            if len(narrowed_phrases) > 1:
                raise ValueError("one process edge cannot retain conflicting narrowed phrases")
            if narrowed_phrases:
                edge["relationship"] = narrowed_phrases.pop()
            if _decision(decision) == "NARROW":
                edge["conditions"] = _conditions(decision, required=True)
            edge_statuses.append(_assessment_status(decision))
        if reject_process:
            rejected_process_indexes.add(process_index)
            continue
        if any(status != "well_grounded" for status in edge_statuses):
            amended["status"] = "tentative"
        retained_process_updates.append(amended)
    filtered["process_updates"] = retained_process_updates

    issues = string_list(review, "issues", allow_empty=True)
    return filtered, {
        "issues": issues,
        "relationship_reviews": [relation_reviews[index] for index in sorted(relation_reviews)],
        "process_edge_reviews": [
            edge_reviews[index] for index in sorted(edge_reviews)
        ],
        "rejected_relationship_update_indexes": sorted(rejected_relation_indexes),
        "discarded_dependent_relationship_update_indexes": sorted(dependent_rejections),
        "rejected_process_update_indexes": sorted(rejected_process_indexes),
    }


def _indexed_reviews(
    reviews: list[dict[str, Any]], key: str, expected: set[int], name: str
) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for review in reviews:
        index = review.get(key)
        if type(index) is not int or index not in expected or index in indexed:
            raise ValueError(f"{name} reviews must name each candidate index exactly once")
        _validate_review(review, name)
        indexed[index] = review
    if set(indexed) != expected:
        raise ValueError(f"{name} reviews must cover every candidate exactly once")
    return indexed


def _candidate_new_relation_ids(
    state: GroundedTheoryState, updates: list[dict[str, Any]]
) -> dict[int, str]:
    used = {relation.id for relation in state.relationships}
    result: dict[int, str] = {}
    for index, update in enumerate(updates):
        if update.get("comparison") != "NEW":
            continue
        relation_id = _next_relation_id(used)
        used.add(relation_id)
        result[index] = relation_id
    return result


def _rebase_candidate_relation_references(
    state: GroundedTheoryState,
    retained_updates: list[dict[str, Any]],
    original_indexes: list[int],
    original_new_relation_ids: dict[int, str],
) -> None:
    """Keep SAME/VARIATION updates valid if an earlier NEW claim was rejected."""
    original_ids = set(original_new_relation_ids.values())
    actual_ids_by_original: dict[str, str] = {}
    used = {relation.id for relation in state.relationships}
    for original_index, update in zip(original_indexes, retained_updates):
        if update.get("comparison") != "NEW":
            continue
        original_id = original_new_relation_ids[original_index]
        actual_id = _next_relation_id(used)
        used.add(actual_id)
        actual_ids_by_original[original_id] = actual_id
    for update in retained_updates:
        relation_id = update.get("existing_relation_id")
        if relation_id in original_ids:
            update["existing_relation_id"] = actual_ids_by_original[relation_id]


def _next_relation_id(used: set[str]) -> str:
    index = 1
    while f"relation_{index:03d}" in used:
        index += 1
    return f"relation_{index:03d}"


def _edge_reviews(
    reviews: list[dict[str, Any]], expected: set[tuple[int, int]]
) -> dict[tuple[int, int], dict[str, Any]]:
    indexed: dict[tuple[int, int], dict[str, Any]] = {}
    for review in reviews:
        process_index, edge_index = review.get("process_update_index"), review.get("edge_index")
        key = (process_index, edge_index)
        if (
            type(process_index) is not int
            or type(edge_index) is not int
            or key not in expected
            or key in indexed
        ):
            raise ValueError("process edge reviews must name each candidate edge exactly once")
        _validate_review(review, "process edge")
        indexed[key] = review
    if set(indexed) != expected:
        raise ValueError("process edge reviews must cover every candidate edge exactly once")
    return indexed


def _validate_review(review: dict[str, Any], name: str) -> None:
    decision = _decision(review)
    assessment = choice(review, "assessment", REVIEW_ASSESSMENTS)
    status = _assessment_status(review)
    text(review, "rationale")
    if decision == "REJECT":
        if assessment != "unsupported" or status != "insufficient_evidence":
            raise ValueError(f"a rejected {name} must be unsupported and insufficient_evidence")
        return
    if assessment == "unsupported":
        raise ValueError(f"a retained {name} cannot be assessed as unsupported")
    if decision == "NARROW":
        _conditions(review, required=True)
        if name == "relationship":
            text(review, "revised_relationship")


def _decision(review: dict[str, Any]) -> str:
    return choice(review, "decision", REVIEW_DECISIONS)


def _assessment_status(review: dict[str, Any]) -> str:
    return choice(review, "status", REVIEW_STATUSES)


def _conditions(review: dict[str, Any], *, required: bool) -> list[str]:
    conditions = string_list(review, "conditions", allow_empty=True)
    if required and not conditions:
        raise ValueError("a narrowed claim must name at least one condition")
    return conditions


def _apply_relationship_review(update: dict[str, Any], review: dict[str, Any]) -> None:
    update["grounding_kind"] = choice(
        review,
        "assessment",
        REVIEW_ASSESSMENTS - {"unsupported"},
    )
    update["status"] = _assessment_status(review)
    update["grounding_explanation"] = text(review, "rationale")
    if _decision(review) != "NARROW":
        return
    update["conditions"] = _conditions(review, required=True)
    if update.get("comparison") == "NEW":
        update["relationship"] = text(review, "revised_relationship")
    else:
        update["revised_relationship"] = text(review, "revised_relationship")
