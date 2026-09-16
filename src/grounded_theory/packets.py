"""Prompt loading and deterministic output schemas for model task packets."""

from __future__ import annotations

from pathlib import Path
from typing import Any


PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"


def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8").strip()


def memo_update_schema() -> dict[str, Any]:
    """Describe every field enforced by ``apply_memo_updates`` for agents."""
    return {
        "decision": "NEW|UPDATE",
        "memo_id": "required for UPDATE",
        "topic": "required",
        "analytic_observation": "required",
        "supporting_evidence": [{"record_id": "record ID", "text_span": "verbatim quote"}],
        "comparisons": "required",
        "tentative_interpretation": "required",
        "negative_or_contradictory_cases": [
            {"record_id": "record ID", "text_span": "verbatim quote"}
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
                        "text_span": "verbatim quote",
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


def relational_output_schema(record_ids: list[str]) -> dict[str, Any]:
    return {
        "processed_record_ids": record_ids,
        "concept_updates": [
            {
                "comparison": "SAME|VARIATION|NEW|CONTRADICTION",
                "existing_concept_id": "required except NEW",
                "label": "required for NEW",
                "definition": "required for NEW",
                "level": "concept|category; optional for NEW, default concept",
                "parent_category_id": "optional category ID; null removes an existing concept from its category",
                "evidence": [{"record_id": "requested record ID", "text_span": "verbatim quote"}],
                "variation": "required for VARIATION",
                "contradiction": "required for CONTRADICTION",
                "revised_definition": "optional evidence-grounded revision of an existing concept or category",
            }
        ],
        "relationship_updates": [
            {
                "comparison": "SAME|VARIATION|NEW|CONTRADICTION",
                "existing_relation_id": "required except NEW",
                "source_concept_id": "required for NEW",
                "relationship": "natural-language relation phrase, required for NEW",
                "target_concept_id": "required for NEW",
                "evidence": [{"record_id": "record ID", "text_span": "verbatim quote"}],
                "comparative_basis": "required for NEW; never state co-occurrence alone",
                "grounding_kind": "explicitly_expressed|repeated_comparison|tentative_theoretical_inference",
                "grounding_explanation": "why the cited records support this grounding kind",
                "status": "well_grounded|tentative|insufficient_evidence",
                "variation": "required for VARIATION",
                "contradiction": "required for CONTRADICTION",
            }
        ],
        "process_updates": [
            {
                "comparison": "SAME|VARIATION|NEW|CONTRADICTION",
                "existing_process_id": "required except NEW",
                "label": "required for NEW",
                "description": "required for NEW",
                "conditions": [],
                "actions_interactions": [],
                "consequences": [],
                "subsequent_changes": [],
                "alternative_pathways": [],
                "edges": [
                    {
                        "source_concept_id": "concept ID",
                        "relationship": "the direct relationship phrase for this arrow",
                        "target_concept_id": "concept ID",
                        "supporting_relation_ids": ["existing direct relationship ID"],
                        "supporting_relationship_update_indexes": [
                            "zero-based index of a relationship_updates item in this submission"
                        ],
                        "conditions": [],
                    }
                ],
                "supporting_record_ids": ["record ID"],
                "negative_cases": [{"record_id": "record ID", "text_span": "verbatim quote"}],
                "status": "well_grounded|tentative|insufficient_evidence",
                "variation": "required for VARIATION",
                "contradiction": "required for CONTRADICTION",
            }
        ],
        "memo_updates": [memo_update_schema()],
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
                "evidence": [{"record_id": "record ID", "text_span": "verbatim quote"}],
                "status": "unresolved",
            }
        ],
        "memo_updates": [memo_update_schema()],
    }
