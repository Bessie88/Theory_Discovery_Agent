"""Prompt loading and deterministic output schemas for model task packets."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"


def load_prompt(name: str) -> str:
    """Load the common stage prompt plus an optional dataset-specific overlay.

    The overlay changes task framing only; all output schemas and deterministic
    validation remain common to every study.  A missing overlay is deliberately
    a no-op so existing projects keep their current prompt behavior.
    """

    prompt = (PROMPTS_DIR / name).read_text(encoding="utf-8").strip()
    overlay_root = os.environ.get("GT_PROMPT_OVERLAY_DIR")
    if not overlay_root:
        return prompt
    overlay = Path(overlay_root) / name
    if not overlay.is_file():
        return prompt
    return f"{prompt}\n\n{overlay.read_text(encoding='utf-8').strip()}"


def memo_update_schema() -> dict[str, Any]:
    """Describe every field enforced by ``apply_memo_updates`` for agents."""
    return {
        "decision": "NEW|UPDATE",
        "memo_id": "required for UPDATE",
        "topic": "required",
        "analytic_observation": "required",
        "supporting_evidence": [{"record_id": "record ID", "quote": "short English quote", "note": "optional analytic note"}],
        "comparisons": "required",
        "tentative_interpretation": "required",
        "negative_or_contradictory_cases": [
            {"record_id": "record ID", "quote": "short English quote", "note": "optional analytic note"}
        ],
        "uncertainties": "required",
        "questions_for_further_analysis": [],
        "linked_concept_ids": [],
        "linked_relation_ids": [],
        "linked_process_ids": [],
    }


def open_coding_output_schema(record_ids: list[str]) -> dict[str, Any]:
    return {
        "processed_record_ids": record_ids,
        "record_judgments": [{
            "record_id": "requested record ID, in the supplied order",
            "disposition": (
                "SUPPORTS_EXISTING|VARIATION|BOUNDARY|POSSIBLE_NEW|"
                "NO_RELEVANT_MECHANISM"
            ),
            "concept_id": "required only for SUPPORTS_EXISTING, VARIATION, or BOUNDARY",
            "candidate_label": "required only for POSSIBLE_NEW",
            "evidence": {
                "record_id": "same requested record ID",
                "quote": "one short exact quote; required except NO_RELEVANT_MECHANISM",
                "note": "optional",
            },
            "rationale": "brief primary coding judgment; at most 24 words",
        }],
        "concept_updates": [
            {
                "comparison": "SAME|VARIATION|NEW|CONTRADICTION",
                "existing_concept_id": "required except NEW",
                "label": "required for NEW",
                "definition": "required for NEW",
                "level": "concept|category; optional for NEW, default concept",
                "parent_category_id": "optional category ID; null removes an existing concept from its category",
                "evidence": [
                    {
                        "record_id": "requested record ID",
                        "quote": "short English quote", "note": "optional analytic note",
                        "note": "optional",
                    }
                ],
                "variation": "required for VARIATION",
                "contradiction": "required for CONTRADICTION",
                "revised_definition": "optional corpus-specific refinement for an existing concept",
            }
        ],
        "memo_updates": [memo_update_schema()],
    }


def open_coding_coverage_audit_output_schema(record_ids: list[str]) -> dict[str, Any]:
    """One primary, durable open-coding judgment for every supplied record."""
    return {
        "processed_record_ids": record_ids,
        "record_judgments": [{
            "record_id": "requested record ID, in the supplied order",
            "disposition": (
                "SUPPORTS_EXISTING|VARIATION|BOUNDARY|POSSIBLE_NEW|"
                "NO_RELEVANT_MECHANISM"
            ),
            "concept_id": "required only for SUPPORTS_EXISTING, VARIATION, or BOUNDARY",
            "candidate_label": "required only for POSSIBLE_NEW",
            "evidence": {
                "record_id": "same requested record ID",
                "quote": "one short exact quote; required except NO_RELEVANT_MECHANISM",
                "note": "optional",
            },
            "rationale": "brief primary coding judgment; at most 24 words",
        }],
    }


def relational_output_schema(record_ids: list[str]) -> dict[str, Any]:
    """Return the compact, valid Stage-2 output shape.

    The stage prompt and strict response contract carry the analytic rules.  A
    terse exemplar leaves the input headroom for source evidence while making
    the response fields and their valid shapes unambiguous.
    """
    evidence = {"record_id": "record ID", "quote": "short quote", "note": "optional"}
    return {
        "processed_record_ids": record_ids,
        "deferred_records": [{"record_id": "requested record ID", "reason": "brief reason"}],
        "concept_updates": [{
            "comparison": "SAME|VARIATION|NEW|CONTRADICTION",
            "existing_concept_id": "unless NEW", "label": "for NEW", "definition": "for NEW",
            "level": "concept|category; optional", "parent_category_id": "optional",
            "evidence": [evidence], "variation": "for VARIATION",
            "contradiction": "for CONTRADICTION", "revised_definition": "optional",
        }],
        "relationship_updates": [{
            "comparison": "SAME|VARIATION|NEW|CONTRADICTION",
            "existing_relation_id": "unless NEW", "source_concept_id": "for NEW",
            "relationship": "for NEW", "target_concept_id": "for NEW", "evidence": [evidence],
            "comparative_basis": "for NEW; not co-occurrence alone",
            "grounding_kind": "explicitly_expressed|repeated_comparison|tentative_theoretical_inference",
            "grounding_explanation": "why evidence supports kind",
            "status": "well_grounded|tentative|insufficient_evidence", "conditions": [],
            "memo_ids": [],
            "variation": "for VARIATION", "contradiction": "for CONTRADICTION",
        }],
        "process_updates": [{
            "comparison": "SAME|VARIATION|NEW|CONTRADICTION",
            "existing_process_id": "unless NEW", "label": "for NEW", "description": "for NEW",
            "conditions": [], "actions_interactions": [], "consequences": [],
            "subsequent_changes": [], "alternative_pathways": [],
            "edges": [{
                "source_concept_id": "concept ID", "relationship": "direct relation phrase",
                "target_concept_id": "concept ID", "evidence": [evidence],
                "supporting_relation_ids": [], "supporting_relationship_update_indexes": [], "conditions": [],
            }],
            "supporting_record_ids": ["record ID"], "negative_cases": [evidence],
            "status": "well_grounded|tentative|insufficient_evidence",
            "variation": "for VARIATION", "contradiction": "for CONTRADICTION",
        }],
        "memo_updates": [{
            "decision": "NEW|UPDATE", "memo_id": "for UPDATE", "topic": "required",
            "analytic_observation": "required", "supporting_evidence": [evidence],
            "comparisons": "required", "tentative_interpretation": "required",
            "negative_or_contradictory_cases": [evidence], "uncertainties": "required",
            "questions_for_further_analysis": [], "linked_concept_ids": [],
            "linked_relation_ids": [], "linked_process_ids": [],
        }],
    }


def relational_grounding_review_output_schema() -> dict[str, Any]:
    """One reviewer transaction decides exactly one traceable claim."""
    return {
        "claim_id": "exactly the supplied review_claim.claim_id",
        "decision": "RETAIN|NARROW|REJECT",
        "assessment": "explicitly_expressed|repeated_comparison|tentative_theoretical_inference|unsupported",
        "status": "well_grounded|tentative|insufficient_evidence",
        "revised_relationship": "required only when review_claim.claim_type is relationship and decision is NARROW",
        "conditions": ["required for NARROW; otherwise []"],
        "rationale": "why this claim's supporting and counterevidence warrants the decision",
    }


def process_edge_grounding_review_output_schema() -> dict[str, Any]:
    """One independently retrieved-evidence decision for an unlinked process edge."""
    return {
        "claim_id": "exactly the supplied review_claim.claim_id",
        "edge_status": "supported|conditional|tentative|rejected",
        "supporting_evidence": [
            {"record_id": "record ID from review_records", "quote": "short English quote", "note": "optional analytic note"}
        ],
        "boundary_conditions": [],
        "negative_evidence": [
            {"record_id": "record ID from review_records", "quote": "short English quote", "note": "optional analytic note"}
        ],
        "rationale": "why the retrieved all-match evidence and boundary cohorts support, qualify, leave tentative, or reject this edge",
    }


def integration_output_schema() -> dict[str, Any]:
    return {
        "integration": {
            "status": "integrated|no_adequately_grounded_core_category",
            "core_category_id": "concept ID or null",
            "account": "evidence-grounded explanatory account",
            "propositions": [
                {
                    "statement": "grounded proposition",
                    "status": "well_grounded|tentative|unresolved",
                    "process_ids": [],
                    "relation_ids": [],
                    "evidence": [],
                }
            ],
            "alternative_pathways": [],
            "negative_case_ids": [],
            "unresolved_questions": [],
        },
        "theoretical_sampling_needs": [
            {
                "target": "underdeveloped concept, relationship, or process",
                "reason": "why the current corpus leaves it unresolved",
                "evidence_needed": "what additional evidence would clarify it",
                "evidence": [{"record_id": "record ID", "quote": "short English quote", "note": "optional analytic note"}],
                "status": "unresolved",
            }
        ],
        "memo_updates": [memo_update_schema()],
    }
