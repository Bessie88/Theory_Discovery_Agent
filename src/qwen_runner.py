"""Bounded, validator-backed execution of one Grounded Theory model packet."""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .persistence import state_from_dict, write_json
from .context_management import (
    CONTEXT_AUDIT_KEY,
    audit_context,
    compact_text,
    compact_concept_index,
    context_receipt_fields,
    detailed_concept_view,
    novelty_candidates,
    public_packet as context_public_packet,
)
from .hierarchical_integration import (
    EVIDENCE_REQUEST_ACTION,
    EVIDENCE_REQUEST_SCHEMA,
    LOCAL_MEMO_ACTION,
    LOCAL_MEMO_SCHEMA,
    REDUCED_MEMO_SCHEMA,
    REDUCE_MEMO_ACTION,
    execute_hierarchical_integration,
)
from .packets import (
    load_prompt,
    process_edge_grounding_review_output_schema,
    relational_grounding_review_output_schema,
)
from .project import GroundedTheoryProject
from .prompt_adaptation import (
    CRITIC_ACTION,
    EDITOR_ACTION,
    FAILURE_EXTRACTION_ACTION,
    PAIRWISE_ACTION,
    PAIRWISE_DIMENSIONS,
    VALIDATION_ACTION,
    blind_candidate_label,
    canonical_hash,
    normalize_pairwise,
    pairwise_schema,
)
from .review import normalize_claim_review
from .review_material import build_relational_review_material
from .token_budget import packet_token_estimate
from .thinking_adapter import thinking_model_alias
from .validation import align_evidence_quote, relational_record_disposition


DEFAULT_OUTPUT_TOKENS = {
    "open_coding": 6144,
    "open_coding_coverage_audit": 3072,
    "relational_process_analysis": 6144,
    "validate_relational_grounding": 4096,
    "theoretical_integration": 8192,
    "resolve_concept_novelty": 2048,
    FAILURE_EXTRACTION_ACTION: 2048,
    CRITIC_ACTION: 4096,
    EDITOR_ACTION: 2048,
    PAIRWISE_ACTION: 4096,
    LOCAL_MEMO_ACTION: 4096,
    REDUCE_MEMO_ACTION: 5000,
    EVIDENCE_REQUEST_ACTION: 2048,
}

# Visible-output targets remain research-design caps. The reduce worker alone
# receives a small runtime allowance so it can close a valid JSON response at
# the target boundary without changing that target.
DEFAULT_OUTPUT_RUNTIME_HEADROOM_TOKENS = 50

# These are hard ceilings for tokens generated inside Qwen's thinking block.
# They deliberately leave enough of each total output allowance for the
# visible JSON response. ``None`` disables Qwen thinking for a stage; use a
# positive cap only where deliberate reasoning is required. Override the
# complete mapping with
# GT_THINKING_TOKEN_BUDGETS when starting the Slurm runner.
DEFAULT_THINKING_TOKENS = {
    "open_coding": None,
    "open_coding_coverage_audit": None,
    "relational_process_analysis": 1000,
    "validate_relational_grounding": None,
    "theoretical_integration": 1000,
    "resolve_concept_novelty": None,
    FAILURE_EXTRACTION_ACTION: None,
    CRITIC_ACTION: None,
    EDITOR_ACTION: None,
    PAIRWISE_ACTION: None,
    LOCAL_MEMO_ACTION: None,
    REDUCE_MEMO_ACTION: None,
    EVIDENCE_REQUEST_ACTION: None,
}

# The adapter is instructed with the research-design target.  SGLang can
# occasionally report one or a few more reasoning tokens at a stop boundary,
# so the client separately reserves and accepts this small runtime allowance.
DEFAULT_THINKING_RUNTIME_HEADROOM_TOKENS = 50

# These limits reserve each action's bounded output budget inside the local
# Qwen 32k context window.  They are a guardrail: no packet is silently
# shortened to fit, because omitted counterevidence would change the method.
DEFAULT_INPUT_TOKENS = {
    "open_coding": 24000,
    "open_coding_coverage_audit": 24000,
    "relational_process_analysis": 24000,
    "validate_relational_grounding": 27000,
    "theoretical_integration": 22000,
    "resolve_concept_novelty": 12000,
    FAILURE_EXTRACTION_ACTION: 18000,
    CRITIC_ACTION: 18000,
    EDITOR_ACTION: 14000,
    PAIRWISE_ACTION: 22000,
    LOCAL_MEMO_ACTION: 18000,
    REDUCE_MEMO_ACTION: 18000,
    EVIDENCE_REQUEST_ACTION: 22000,
}

MODEL_CONTEXT_WINDOW_TOKENS = 32_768
# Validator feedback is useful for repair, but it is not source evidence.  A
# bounded copy prevents a pathological validation message from growing a retry
# beyond the request that was originally approved for the model context.
MAX_REPAIR_FEEDBACK_CHARS = 1_200

REVIEW_BATCH_INPUT_TOKENS = 27000
# Keep the study's relation-review target at 27,000 input tokens.  A tiny
# acceptance margin absorbs tokenizer accounting at packet-finalization
# boundaries; it is not a general context-window expansion.
RELATIONAL_REVIEW_INPUT_ACCEPTANCE_HEADROOM_TOKENS = 50
RELATIONAL_REVIEW_INPUT_ACCEPTANCE_TOKENS = (
    REVIEW_BATCH_INPUT_TOKENS + RELATIONAL_REVIEW_INPUT_ACCEPTANCE_HEADROOM_TOKENS
)
MAX_REVIEW_PARALLELISM = 4


OUTPUT_JSON_SCHEMAS: dict[str, dict[str, Any]] = {
    "open_coding": {
        "type": "object",
        "required": ["processed_record_ids", "concept_updates", "memo_updates"],
        "properties": {
            "processed_record_ids": {"type": "array", "items": {"type": "string"}},
            "record_judgments": {"type": "array", "items": {"type": "object"}},
            "concept_updates": {"type": "array", "items": {"type": "object"}},
            "memo_updates": {"type": "array", "items": {"type": "object"}},
        },
    },
    "open_coding_coverage_audit": {
        "type": "object",
        "required": ["processed_record_ids", "record_judgments"],
        "properties": {
            "processed_record_ids": {"type": "array", "items": {"type": "string"}},
            "record_judgments": {"type": "array", "items": {"type": "object"}},
        },
    },
    "relational_process_analysis": {
        "type": "object",
        "required": [
            "processed_record_ids", "concept_updates", "relationship_updates",
            "process_updates", "memo_updates",
        ],
        "properties": {
            "processed_record_ids": {"type": "array", "items": {"type": "string"}},
            "deferred_records": {"type": "array", "items": {"type": "object"}},
            "concept_updates": {"type": "array", "items": {"type": "object"}},
            "relationship_updates": {"type": "array", "items": {"type": "object"}},
            "process_updates": {"type": "array", "items": {"type": "object"}},
            "memo_updates": {"type": "array", "items": {"type": "object"}},
        },
    },
    "validate_relational_grounding": {
        "type": "object",
        "required": ["claim_id", "decision", "assessment", "status", "rationale", "conditions"],
        "properties": {
            "claim_id": {"type": "string"},
            "decision": {"type": "string"},
            "assessment": {"type": "string"},
            "status": {"type": "string"},
            "rationale": {"type": "string"},
            "conditions": {"type": "array", "items": {"type": "string"}},
            "revised_relationship": {"type": "string"},
        },
    },
    "theoretical_integration": {
        "type": "object",
        "required": ["integration", "theoretical_sampling_needs", "memo_updates"],
        "properties": {
            "integration": {
                "type": "object",
                "required": [
                    "status", "core_category_id", "account", "propositions",
                    "alternative_pathways", "negative_case_ids", "unresolved_questions",
                ],
                "properties": {
                    "status": {"type": "string"},
                    "core_category_id": {"type": "string_or_null"},
                    "account": {"type": "string"},
                    "propositions": {"type": "array", "items": {"type": "object"}},
                    "alternative_pathways": {"type": "array", "items": {"type": "string"}},
                    "negative_case_ids": {"type": "array", "items": {"type": "string"}},
                    "unresolved_questions": {"type": "array", "items": {"type": "string"}},
                },
            },
            "theoretical_sampling_needs": {"type": "array", "items": {"type": "object"}},
            "memo_updates": {"type": "array", "items": {"type": "object"}},
        },
    },
    "resolve_concept_novelty": {
        "type": "object",
        "required": ["resolutions"],
        "properties": {
            "resolutions": {"type": "array", "items": {"type": "object"}},
        },
    },
    FAILURE_EXTRACTION_ACTION: {"type": "object"},
    CRITIC_ACTION: {"type": "object"},
    EDITOR_ACTION: {"type": "object"},
    PAIRWISE_ACTION: {"type": "object"},
    LOCAL_MEMO_ACTION: LOCAL_MEMO_SCHEMA,
    REDUCE_MEMO_ACTION: REDUCED_MEMO_SCHEMA,
    EVIDENCE_REQUEST_ACTION: EVIDENCE_REQUEST_SCHEMA,
}


PROCESS_EDGE_REVIEW_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "claim_id",
        "edge_status",
        "supporting_evidence",
        "boundary_conditions",
        "negative_evidence",
        "rationale",
    ],
    "properties": {
        "claim_id": {"type": "string"},
        "edge_status": {"type": "string"},
        "supporting_evidence": {"type": "array", "items": {"type": "object"}},
        "boundary_conditions": {"type": "array", "items": {"type": "string"}},
        "negative_evidence": {"type": "array", "items": {"type": "object"}},
        "rationale": {"type": "string"},
    },
}


# Stage 1 is deliberately a compact constant-comparison pass. These limits
# prevent an excessively verbose completion from consuming the response budget
# for a small record batch. They run before any state mutation.
MAX_OPEN_CODING_CONCEPT_UPDATES = 6
MAX_OPEN_CODING_EVIDENCE_PER_UPDATE = 2
MAX_OPEN_CODING_MEMO_UPDATES = 1
MAX_OPEN_CODING_MEMO_SUPPORTING_EVIDENCE = 2
MAX_OPEN_CODING_MEMO_NEGATIVE_EVIDENCE = 1
MAX_OPEN_CODING_MEMO_QUESTIONS = 2
MAX_RELATIONAL_CONCEPT_UPDATES = 2
MAX_RELATIONAL_RELATIONSHIP_UPDATES = 3
MAX_RELATIONAL_PROCESS_UPDATES = 1
MAX_RELATIONAL_EVIDENCE_PER_UPDATE = 2
MAX_RELATIONAL_PROCESS_EDGES = 3
MAX_RELATIONAL_MEMO_UPDATES = 2
MAX_RELATIONAL_REVIEW_RATIONALE_WORDS = 120


def _schema_text(limit: int) -> dict[str, Any]:
    """One non-empty, XGrammar-compatible bounded output string.

    The deployed XGrammar release rejects a schema that combines ``pattern``
    and ``minLength``/``maxLength`` on the same string. Length bounds are
    natively enforced and provide the needed response-size guardrail.
    """
    return {
        "type": "string",
        "minLength": 1,
        "maxLength": limit,
    }


def _schema_text_list(max_items: int, text_limit: int) -> dict[str, Any]:
    return {"type": "array", "maxItems": max_items, "items": _schema_text(text_limit)}


def _schema_evidence() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "record_id": _schema_text(80),
            "quote": _schema_text(180),
            "note": _schema_text(96),
        },
        "required": ["record_id", "quote"],
        "additionalProperties": False,
    }


def _schema_evidence_list(max_items: int) -> dict[str, Any]:
    return {"type": "array", "maxItems": max_items, "items": _schema_evidence()}


def relational_structured_output_schema() -> dict[str, Any]:
    """Hard grammar for a bounded, complete Stage-2 transaction.

    Prompt caps alone leave a model free to write a huge yet valid JSON object.
    This schema is sent to SGLang/XGrammar, so decoding itself cannot add
    arbitrary keys or unbounded update lists and prose fields.
    """
    # ``maxItems`` alone bounded the number of discoveries but still let a
    # response fill every optional field in every discovery.  These mutually
    # exclusive comparison branches encode the *minimal valid payload* for
    # each analytic decision.  Evidence and full-corpus independent review are
    # retained; redundant state-edit fields are deliberately unavailable.
    identifier = _schema_text(48)
    short_text = _schema_text(72)
    analytic_text = _schema_text(160)
    relation_text = _schema_text(96)

    def evidence_items(maximum: int, *, minimum: int = 0) -> dict[str, Any]:
        return {
            "type": "array",
            "minItems": minimum,
            "maxItems": maximum,
            "items": {
                "type": "object",
                "properties": {
                    "record_id": identifier,
                    "quote": _schema_text(128),
                },
                "required": ["record_id", "quote"],
                "additionalProperties": False,
            },
        }

    def comparison_branch(
        decision: str, properties: dict[str, Any], required: list[str]
    ) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"comparison": {"enum": [decision]}, **properties},
            "required": ["comparison", *required],
            "additionalProperties": False,
        }

    raw_process_edge = {
        "type": "object",
        "properties": {
            "source_concept_id": identifier,
            "relationship": relation_text,
            "target_concept_id": identifier,
            "evidence": evidence_items(1, minimum=1),
            "supporting_relation_ids": _schema_text_list(2, 48),
            "supporting_relationship_update_indexes": {
                "type": "array", "maxItems": 2,
                "items": {"type": "integer", "minimum": 0, "maximum": 2},
            },
            "conditions": _schema_text_list(1, 72),
            "negative_evidence": evidence_items(1),
        },
        "required": ["source_concept_id", "relationship", "target_concept_id", "evidence"],
        "additionalProperties": False,
    }
    concept_update = {
        "oneOf": [
            comparison_branch("NEW", {
                "label": short_text, "definition": analytic_text,
                "evidence": evidence_items(1, minimum=1),
            }, ["label", "definition", "evidence"]),
            comparison_branch("SAME", {
                "existing_concept_id": identifier, "evidence": evidence_items(1, minimum=1),
            }, ["existing_concept_id", "evidence"]),
            comparison_branch("VARIATION", {
                "existing_concept_id": identifier, "evidence": evidence_items(1, minimum=1),
                "variation": analytic_text,
            }, ["existing_concept_id", "evidence", "variation"]),
            comparison_branch("CONTRADICTION", {
                "existing_concept_id": identifier, "evidence": evidence_items(1, minimum=1),
                "contradiction": analytic_text,
            }, ["existing_concept_id", "evidence", "contradiction"]),
        ],
    }
    relation_common = {
        "evidence": evidence_items(MAX_RELATIONAL_EVIDENCE_PER_UPDATE, minimum=1),
        "grounding_kind": {"enum": [
            "explicitly_expressed", "repeated_comparison", "tentative_theoretical_inference",
        ]},
        "grounding_explanation": analytic_text,
    }
    relationship_update = {
        "oneOf": [
            comparison_branch("NEW", {
                "source_concept_id": identifier, "relationship": relation_text,
                "target_concept_id": identifier, "comparative_basis": analytic_text,
                **relation_common,
            }, [
                "source_concept_id", "relationship", "target_concept_id", "comparative_basis",
                "evidence", "grounding_kind", "grounding_explanation",
            ]),
            comparison_branch("SAME", {
                "existing_relation_id": identifier, **relation_common,
            }, ["existing_relation_id", "evidence", "grounding_kind", "grounding_explanation"]),
            comparison_branch("VARIATION", {
                "existing_relation_id": identifier, "variation": analytic_text, **relation_common,
            }, [
                "existing_relation_id", "variation", "evidence", "grounding_kind",
                "grounding_explanation",
            ]),
            comparison_branch("CONTRADICTION", {
                "existing_relation_id": identifier, "contradiction": analytic_text, **relation_common,
            }, [
                "existing_relation_id", "contradiction", "evidence", "grounding_kind",
                "grounding_explanation",
            ]),
        ],
    }
    process_edges = {
        "type": "array", "minItems": 1, "maxItems": 2, "items": raw_process_edge,
    }
    process_update = {
        "oneOf": [
            comparison_branch("NEW", {
                "label": short_text, "description": analytic_text, "edges": process_edges,
            }, ["label", "description", "edges"]),
            comparison_branch("SAME", {
                "existing_process_id": identifier, "edges": process_edges,
            }, ["existing_process_id", "edges"]),
            comparison_branch("VARIATION", {
                "existing_process_id": identifier, "variation": analytic_text, "edges": process_edges,
            }, ["existing_process_id", "variation", "edges"]),
            comparison_branch("CONTRADICTION", {
                "existing_process_id": identifier, "contradiction": analytic_text, "edges": process_edges,
            }, ["existing_process_id", "contradiction", "edges"]),
        ],
    }
    memo_common = {
        "topic": short_text,
        "analytic_observation": analytic_text,
        "supporting_evidence": evidence_items(1, minimum=1),
        "comparisons": analytic_text,
        "tentative_interpretation": analytic_text,
        "negative_or_contradictory_cases": evidence_items(1),
        "uncertainties": short_text,
        "questions_for_further_analysis": _schema_text_list(1, 96),
        "linked_concept_ids": _schema_text_list(2, 48),
        "linked_relation_ids": _schema_text_list(2, 48),
        "linked_process_ids": _schema_text_list(1, 48),
    }
    memo_required = [
        "topic", "analytic_observation", "supporting_evidence", "comparisons",
        "tentative_interpretation", "negative_or_contradictory_cases", "uncertainties",
        "questions_for_further_analysis", "linked_concept_ids", "linked_relation_ids",
        "linked_process_ids",
    ]
    memo_update = {
        "oneOf": [
            {
                "type": "object",
                "properties": {"decision": {"enum": ["NEW"]}, **memo_common},
                "required": ["decision", *memo_required],
                "additionalProperties": False,
            },
            {
                "type": "object",
                "properties": {"decision": {"enum": ["UPDATE"]}, "memo_id": identifier, **memo_common},
                "required": ["decision", "memo_id", *memo_required],
                "additionalProperties": False,
            },
        ],
    }
    return {
        "type": "object",
        "properties": {
            "processed_record_ids": _schema_text_list(8, 48),
            "deferred_records": {
                "type": "array", "maxItems": 8,
                "items": {
                    "type": "object",
                    "properties": {"record_id": identifier, "reason": _schema_text(96)},
                    "required": ["record_id", "reason"],
                    "additionalProperties": False,
                },
            },
            "concept_updates": {"type": "array", "maxItems": MAX_RELATIONAL_CONCEPT_UPDATES, "items": concept_update},
            "relationship_updates": {"type": "array", "maxItems": MAX_RELATIONAL_RELATIONSHIP_UPDATES, "items": relationship_update},
            "process_updates": {"type": "array", "maxItems": MAX_RELATIONAL_PROCESS_UPDATES, "items": process_update},
            "memo_updates": {"type": "array", "maxItems": MAX_RELATIONAL_MEMO_UPDATES, "items": memo_update},
        },
        "required": [
            "processed_record_ids", "deferred_records", "concept_updates", "relationship_updates",
            "process_updates", "memo_updates",
        ],
        "additionalProperties": False,
    }


def response_format_for_action(action: str) -> dict[str, Any]:
    """Use XGrammar's real JSON Schema support for the output-risky Stage 2."""
    if action == "relational_process_analysis":
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "grounded_theory_relational_analysis",
                "strict": True,
                "schema": relational_structured_output_schema(),
            },
        }
    return {"type": "json_object"}


def _validate_text_word_limit(value: Any, *, limit: int, path: str) -> None:
    if isinstance(value, str) and len(value.split()) > limit:
        raise ValueError(f"{path} exceeds the {limit}-word compact-output limit")


def _validate_continuous_quote(value: Any, *, path: str) -> None:
    """Reject stitched quotations before evidence alignment reaches state logic."""
    if isinstance(value, str) and ("..." in value or "…" in value):
        raise ValueError(
            f"{path} must be one contiguous verbatim source excerpt; "
            "do not use an ellipsis to stitch phrases"
        )


def _validate_open_coding_output_volume(payload: dict[str, Any]) -> None:
    """Enforce the concise SoCRATES open-coding response contract."""
    updates = payload.get("concept_updates")
    if isinstance(updates, list):
        if len(updates) > MAX_OPEN_CODING_CONCEPT_UPDATES:
            raise ValueError(
                "open_coding concept_updates exceeds the "
                f"{MAX_OPEN_CODING_CONCEPT_UPDATES}-update compact-output limit"
            )
        for index, update in enumerate(updates):
            if not isinstance(update, dict):
                continue
            evidence = update.get("evidence")
            if isinstance(evidence, list):
                if len(evidence) > MAX_OPEN_CODING_EVIDENCE_PER_UPDATE:
                    raise ValueError(
                        f"$.concept_updates[{index}].evidence exceeds the "
                        f"{MAX_OPEN_CODING_EVIDENCE_PER_UPDATE}-item compact-output limit"
                    )
                for evidence_index, item in enumerate(evidence):
                    if isinstance(item, dict):
                        _validate_text_word_limit(
                            item.get("quote"), limit=24,
                            path=f"$.concept_updates[{index}].evidence[{evidence_index}].quote",
                        )
                        _validate_continuous_quote(
                            item.get("quote"),
                            path=f"$.concept_updates[{index}].evidence[{evidence_index}].quote",
                        )
                        _validate_text_word_limit(
                            item.get("note"), limit=16,
                            path=f"$.concept_updates[{index}].evidence[{evidence_index}].note",
                        )
            for key in ("definition", "revised_definition", "variation", "contradiction"):
                _validate_text_word_limit(
                    update.get(key), limit=40,
                    path=f"$.concept_updates[{index}].{key}",
                )
            _validate_text_word_limit(
                update.get("label"), limit=12,
                path=f"$.concept_updates[{index}].label",
            )

    memos = payload.get("memo_updates")
    if not isinstance(memos, list):
        return
    if len(memos) > MAX_OPEN_CODING_MEMO_UPDATES:
        raise ValueError(
            "open_coding memo_updates exceeds the "
            f"{MAX_OPEN_CODING_MEMO_UPDATES}-memo compact-output limit"
        )
    if not memos or not isinstance(memos[0], dict):
        return
    memo = memos[0]
    for key in ("analytic_observation", "comparisons", "tentative_interpretation", "uncertainties"):
        _validate_text_word_limit(memo.get(key), limit=50, path=f"$.memo_updates[0].{key}")
    _validate_text_word_limit(memo.get("topic"), limit=14, path="$.memo_updates[0].topic")
    for key, limit in (
        ("supporting_evidence", MAX_OPEN_CODING_MEMO_SUPPORTING_EVIDENCE),
        ("negative_or_contradictory_cases", MAX_OPEN_CODING_MEMO_NEGATIVE_EVIDENCE),
        ("questions_for_further_analysis", MAX_OPEN_CODING_MEMO_QUESTIONS),
    ):
        value = memo.get(key)
        if isinstance(value, list) and len(value) > limit:
            raise ValueError(f"$.memo_updates[0].{key} exceeds the {limit}-item compact-output limit")
        if isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, dict):
                        _validate_continuous_quote(
                            item.get("quote"), path=f"$.memo_updates[0].{key}[{index}].quote"
                        )


def _validate_open_coding_coverage_output(
    packet: dict[str, Any], payload: dict[str, Any]
) -> None:
    """Reject any result that lacks a one-to-one record-level judgment."""
    expected = packet.get("expected_output", {}).get("processed_record_ids")
    if not isinstance(expected, list) or any(not isinstance(item, str) for item in expected):
        raise ValueError("coverage audit packet lacks its requested record IDs")
    received = payload.get("processed_record_ids")
    if received != expected:
        raise ValueError("coverage audit processed_record_ids must exactly equal the requested batch")
    judgments = payload.get("record_judgments")
    if not isinstance(judgments, list):
        raise ValueError("coverage audit record_judgments must be an array")
    ids = [item.get("record_id") if isinstance(item, dict) else None for item in judgments]
    if ids != expected:
        raise ValueError("coverage audit needs one ordered judgment for every requested record")
    for index, item in enumerate(judgments):
        if not isinstance(item, dict):
            raise ValueError(f"$.record_judgments[{index}] must be an object")
        disposition = item.get("disposition")
        if disposition not in {
            "SUPPORTS_EXISTING", "VARIATION", "BOUNDARY", "POSSIBLE_NEW",
            "NO_RELEVANT_MECHANISM",
        }:
            raise ValueError(f"$.record_judgments[{index}].disposition is invalid")
        _validate_text_word_limit(
            item.get("rationale"), limit=24, path=f"$.record_judgments[{index}].rationale"
        )
        evidence = item.get("evidence")
        needs_evidence = disposition != "NO_RELEVANT_MECHANISM"
        if needs_evidence:
            if not isinstance(evidence, dict) or evidence.get("record_id") != item.get("record_id"):
                raise ValueError(f"$.record_judgments[{index}] requires same-record evidence")
            _validate_text_word_limit(
                evidence.get("quote"), limit=24,
                path=f"$.record_judgments[{index}].evidence.quote",
            )
            _validate_continuous_quote(
                evidence.get("quote"), path=f"$.record_judgments[{index}].evidence.quote"
            )
        elif evidence is not None:
            raise ValueError(f"$.record_judgments[{index}] must omit evidence")


def _validate_evidence_volume(value: Any, *, limit: int, path: str) -> None:
    if not isinstance(value, list):
        return
    if len(value) > limit:
        raise ValueError(f"{path} exceeds the {limit}-item compact-output limit")


def _validate_relational_output_volume(payload: dict[str, Any]) -> None:
    """Keep Stage 2 focused on substantive theory-state changes."""
    for key, limit in (
        ("concept_updates", MAX_RELATIONAL_CONCEPT_UPDATES),
        ("relationship_updates", MAX_RELATIONAL_RELATIONSHIP_UPDATES),
        ("process_updates", MAX_RELATIONAL_PROCESS_UPDATES),
        ("memo_updates", MAX_RELATIONAL_MEMO_UPDATES),
    ):
        value = payload.get(key)
        if isinstance(value, list) and len(value) > limit:
            raise ValueError(f"$.{key} exceeds the {limit}-item compact-output limit")

    deferred = payload.get("deferred_records")
    if isinstance(deferred, list):
        for index, item in enumerate(deferred):
            if not isinstance(item, dict):
                continue
            _validate_text_word_limit(
                item.get("reason"), limit=24, path=f"$.deferred_records[{index}].reason"
            )

    for index, update in enumerate(payload.get("concept_updates", [])):
        if not isinstance(update, dict):
            continue
        _validate_evidence_volume(
            update.get("evidence"), limit=MAX_RELATIONAL_EVIDENCE_PER_UPDATE,
            path=f"$.concept_updates[{index}].evidence",
        )
        for key in ("definition", "revised_definition", "variation", "contradiction"):
            _validate_text_word_limit(update.get(key), limit=40, path=f"$.concept_updates[{index}].{key}")

    for index, update in enumerate(payload.get("relationship_updates", [])):
        if not isinstance(update, dict):
            continue
        _validate_evidence_volume(
            update.get("evidence"), limit=MAX_RELATIONAL_EVIDENCE_PER_UPDATE,
            path=f"$.relationship_updates[{index}].evidence",
        )
        for key in ("relationship", "comparative_basis", "grounding_explanation", "variation", "contradiction"):
            _validate_text_word_limit(update.get(key), limit=50, path=f"$.relationship_updates[{index}].{key}")

    for index, process in enumerate(payload.get("process_updates", [])):
        if not isinstance(process, dict):
            continue
        _validate_text_word_limit(process.get("label"), limit=12, path=f"$.process_updates[{index}].label")
        _validate_text_word_limit(process.get("description"), limit=60, path=f"$.process_updates[{index}].description")
        edges = process.get("edges")
        if isinstance(edges, list) and len(edges) > MAX_RELATIONAL_PROCESS_EDGES:
            raise ValueError(
                f"$.process_updates[{index}].edges exceeds the "
                f"{MAX_RELATIONAL_PROCESS_EDGES}-item compact-output limit"
            )
        if isinstance(edges, list):
            for edge_index, edge in enumerate(edges):
                if isinstance(edge, dict):
                    _validate_evidence_volume(
                        edge.get("evidence"), limit=MAX_RELATIONAL_EVIDENCE_PER_UPDATE,
                        path=f"$.process_updates[{index}].edges[{edge_index}].evidence",
                    )
        _validate_evidence_volume(
            process.get("negative_cases"), limit=2,
            path=f"$.process_updates[{index}].negative_cases",
        )


def _validate_relational_record_disposition(packet: dict[str, Any], payload: dict[str, Any]) -> None:
    """Apply the same coverage/no-repeat gate before any candidate is staged."""
    records = packet.get("records", [])
    if not isinstance(records, list):
        raise ValueError("Stage-2 packet records must be an array")
    record_ids = [item.get("id") for item in records if isinstance(item, dict)]
    if len(record_ids) != len(records) or any(not isinstance(item, str) for item in record_ids):
        raise ValueError("Stage-2 packet records must each have a string ID")
    forced = packet.get("forced_resolution_record_ids", [])
    relational_record_disposition(
        payload,
        {"record_ids": record_ids, "forced_resolution_record_ids": forced},
    )


def _validate_relational_review_output_volume(payload: dict[str, Any]) -> None:
    """Keep grounding-review decisions bounded without rejecting usable evidence.

    Review evidence is revalidated against the persistent source corpus at
    commit time.  It must therefore remain traceable, but it need not obey the
    much tighter presentation limits used for ordinary agent updates.  Those
    limits were causing otherwise valid review groups to exhaust retries over
    a few extra words in a quote or note.
    """
    _validate_text_word_limit(
        payload.get("rationale"),
        limit=MAX_RELATIONAL_REVIEW_RATIONALE_WORDS,
        path="$.rationale",
    )
    conditions = payload.get("conditions")
    if isinstance(conditions, list):
        if len(conditions) > 4:
            raise ValueError("$.conditions exceeds the 4-item compact-output limit")
        for index, condition in enumerate(conditions):
            _validate_text_word_limit(condition, limit=30, path=f"$.conditions[{index}]")
    for path, value in (
        ("$.supporting_evidence", payload.get("supporting_evidence")),
        ("$.negative_evidence", payload.get("negative_evidence")),
    ):
        if isinstance(value, list) and len(value) > 2:
            raise ValueError(f"{path} exceeds the 2-item compact-output limit")


def validate_stage_json_schema(packet: dict[str, Any], payload: dict[str, Any]) -> None:
    """Hard JSON shape gate for every model stage, before state submission."""
    action = packet.get("action")
    schema = _output_schema_for_packet(packet)
    if schema is None:
        raise CompletionError(f"no output JSON schema for action: {action!r}")
    _validate_json_value(payload, schema, "$")
    if action == "open_coding":
        _validate_open_coding_output_volume(payload)
    elif action == "open_coding_coverage_audit":
        _validate_open_coding_coverage_output(packet, payload)
    elif action == "relational_process_analysis":
        _validate_relational_output_volume(payload)
        _validate_relational_record_disposition(packet, payload)
    elif action == "validate_relational_grounding":
        _validate_relational_review_output_volume(payload)


def _output_schema_for_packet(packet: dict[str, Any]) -> dict[str, Any] | None:
    action = packet.get("action")
    schema = OUTPUT_JSON_SCHEMAS.get(action) if isinstance(action, str) else None
    claim = packet.get("review_claim")
    if (
        action == "validate_relational_grounding"
        and isinstance(claim, dict)
        and claim.get("claim_type") == "process_edge"
    ):
        return PROCESS_EDGE_REVIEW_JSON_SCHEMA
    return schema


def _discard_stage_output_extra_fields(
    packet: dict[str, Any], candidate: Any
) -> tuple[Any, list[str]]:
    """Tolerate presentation-only top-level extras without changing findings.

    All fields required by the declared output schema still have to be present
    and valid.  This merely drops keys the model appended beyond that schema;
    evidence/ID provenance and every substantive state validator run unchanged.
    """
    if not isinstance(candidate, dict):
        return candidate, []
    schema = _output_schema_for_packet(packet)
    properties = schema.get("properties") if isinstance(schema, dict) else None
    if not isinstance(properties, dict):
        return candidate, []
    ignored = sorted(set(candidate) - set(properties))
    if not ignored:
        return candidate, []
    return {key: value for key, value in candidate.items() if key in properties}, ignored


def validate_tolerant_stage_json_schema(packet: dict[str, Any], payload: dict[str, Any]) -> None:
    """Validate an integration worker result after discarding harmless extras."""
    normalized, _ = _discard_stage_output_extra_fields(packet, payload)
    if normalized is not payload and isinstance(normalized, dict):
        payload.clear()
        payload.update(normalized)
    validate_stage_json_schema(packet, payload)


def _validate_json_value(value: Any, schema: dict[str, Any], path: str) -> None:
    expected_type = schema["type"]
    if expected_type == "object":
        if not isinstance(value, dict):
            raise ValueError(f"{path} must be a JSON object")
        if "properties" not in schema:
            return
        properties = schema["properties"]
        missing = [name for name in schema.get("required", []) if name not in value]
        if missing:
            raise ValueError(f"{path} is missing required fields: {missing}")
        extra = sorted(set(value) - set(properties))
        if extra:
            raise ValueError(f"{path} has unsupported fields: {extra}")
        for name, item_schema in properties.items():
            if name in value:
                _validate_json_value(value[name], item_schema, f"{path}.{name}")
        return
    if expected_type == "array":
        if not isinstance(value, list):
            raise ValueError(f"{path} must be a JSON array")
        item_schema = schema.get("items")
        if item_schema is not None:
            for index, item in enumerate(value):
                _validate_json_value(item, item_schema, f"{path}[{index}]")
        return
    if expected_type == "string":
        if not isinstance(value, str):
            raise ValueError(f"{path} must be a JSON string")
        return
    if expected_type == "string_or_null":
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{path} must be a JSON string or null")
        return
    raise RuntimeError(f"unsupported JSON schema type: {expected_type}")

SYSTEM_PROMPT = """You are a careful Grounded Theory analyst working on one isolated transaction.
The user packet is the complete and only analytic input. Follow its prompt, evidence rules,
and expected output exactly. Work silently, then return exactly one JSON object and no prose.
Use concise, plain English for all non-verbatim, reader-facing text. Prefer short, direct
wording; do not change required JSON keys, IDs, enum values, or evidence record IDs and quoted excerpts.
Never invent an evidence location, record ID, concept ID, relation ID, process ID, or causal link.
When the supplied records do not justify an update, use an empty update list rather than
inventing a finding. Do not claim saturation. This response is checked by a deterministic
validator before it can be committed."""


STRICT_JSON_CONTRACT = """STRICT JSON TYPE CONTRACT (required):
- Return exactly one JSON object, with no Markdown or prose before or after it.
- Follow expected_output: a text field is one non-empty string, a field shown as [] is an
  array, and an evidence item is an object. Do not substitute an array for a text field.
- Use only the listed enum values. Use [] for an allowed empty collection.
- Each evidence item uses {"record_id": "...", "quote": "...", "note": "..."}; note is optional.\n  quote is temporary and is not saved; the validator derives the stored source excerpt.\n- In every memo_updates item, comparisons, tentative_interpretation, and uncertainties are
  scalar strings. Evidence and linked-ID fields are arrays.
"""

ACTION_OUTPUT_CONTRACTS = {
    "open_coding": """For open_coding, the top-level fields are processed_record_ids,
record_judgments, concept_updates, and memo_updates; all four are arrays. There must be exactly
one ordered record_judgment for every supplied record. Each judgment is SUPPORTS_EXISTING,
VARIATION, BOUNDARY, POSSIBLE_NEW, or NO_RELEVANT_MECHANISM. The first four require one exact
same-record quote; existing-concept judgments name a concept ID; POSSIBLE_NEW names only a
short candidate label; NO_RELEVANT_MECHANISM has no quote and gives a short rationale. Each concept comparison is one of
SAME, VARIATION, NEW, or CONTRADICTION. Every concept evidence value is an array of evidence
objects; label and definition are scalar strings when NEW.""",
    "open_coding_coverage_audit": """For open_coding_coverage_audit, return
processed_record_ids and record_judgments only. There must be exactly one
ordered judgment for every supplied record. Each judgment is SUPPORTS_EXISTING,
VARIATION, BOUNDARY, POSSIBLE_NEW, or NO_RELEVANT_MECHANISM. The first four
need one exact same-record quote; existing-concept judgments name a concept ID;
POSSIBLE_NEW names only a short candidate label; NO_RELEVANT_MECHANISM has no
quote and gives a short rationale. Before returning, mechanically verify that
each judgment record_id equals its evidence record_id and that the quote is a
verbatim contiguous span of that exact record, never an adjacent turn.""",
    "relational_process_analysis": """For relational_process_analysis, the only top-level keys
are processed_record_ids, deferred_records, concept_updates, relationship_updates,
process_updates, and memo_updates; all are arrays. At most 2 concept updates, 3 relationship
updates, 1 process update, and 2 memo updates are allowed; these are maximums, not quotas, and
fewer findings are preferable when evidence is weak. Every supplied record must occur exactly
once in processed_record_ids or deferred_records. A deferred record is not evidence for an
emitted update. Never defer a forced-resolution ID. Use the minimal fields for the comparison:
NEW concepts use label, definition, evidence; SAME uses existing_concept_id and evidence;
VARIATION or CONTRADICTION additionally uses only its matching explanation field. NEW
relationships use source_concept_id, relationship, target_concept_id, comparative_basis,
evidence, grounding_kind, and grounding_explanation; existing relationships use their ID,
evidence, grounding_kind, grounding_explanation, plus variation or contradiction only when
applicable. NEW processes use label, description, and directly evidenced edges; existing
processes use their ID, edges, plus variation or contradiction only when applicable. Do not
emit unavailable fields, empty placeholder arrays, or all optional fields. Independent review
will add process-edge status and reviewer rationale after this candidate is accepted.""",
    "validate_relational_grounding": """For validate_relational_grounding, the top-level
fields are claim_id, decision, assessment, status, revised_relationship, conditions, and
rationale. claim_id, decision, assessment, status, rationale, and a supplied
revised_relationship are scalar strings; conditions is an array of strings. Use the exact
claim_id supplied in review_claim.""",
    "theoretical_integration": """For theoretical_integration, the top-level fields are
integration, theoretical_sampling_needs, and memo_updates. integration is one object;
propositions, alternative_pathways, negative_case_ids, unresolved_questions, sampling needs,
and memo_updates are arrays. account and proposition statements are scalar strings; evidence
is an array of evidence objects. Cite only record IDs in evidence_on_demand; every quote must be a short exact contiguous excerpt from the cited record's source_summary.text. If no exact excerpt is available, use [] rather than a paraphrase.""",
    "resolve_concept_novelty": """For resolve_concept_novelty, return only resolutions. Each
resolution names one candidate_index and decides genuinely_new, existing_equivalent,
property_or_dimension, or refine_existing. Similarity only triggers review; do not merge or
alter concepts in this independent transaction.""",
    FAILURE_EXTRACTION_ACTION: """For extract_stage2_methodological_failures, return only
batch_id and events. Each event names one supplied claim, one allowed failure_type, a positive
instance_count, a concise reason, and only supplied evidence_record_ids.""",
    CRITIC_ACTION: """For diagnose_stage2_prompt_strategy, return only failure_type,
prompt_related, diagnosis, evidence, recommended_change, and risk. The critic diagnoses but
does not edit a strategy or methodological rule.""",
    EDITOR_ACTION: """For edit_stage2_prompt_strategy, return exactly one allowlisted
strategy patch with parent_version, failure_addressed, operation, rule_id, old_text, new_text,
rationale, expected_effect, and possible_regression.""",
    PAIRWISE_ACTION: """For blind_pairwise_stage2_evaluation, return dimensions, overall,
and rationale. Every dimension uses an A, B, or TIE winner with a concise rationale.""",
    LOCAL_MEMO_ACTION: """For theoretical_integration_local_memo, return exactly the local
theory memo shape, including its task-aware handoff for final theoretical integration.
Every handoff finding and uncertainty needs supplied source IDs; retain negative cases and
tentative alternatives. Do not merge categories or claim causality not supported by supplied
evidence.""",
    REDUCE_MEMO_ACTION: """For theoretical_integration_reduce_memo, return only neighborhood_id,
source_ids with canonical memo IDs, and handoff summary statements with memo_references. Do not
repeat local-memo fields or evidence.""",
    EVIDENCE_REQUEST_ACTION: """For theoretical_integration_evidence_request, return only
evidence_requests: an array of up to twelve objects with id and reason.""",
}


def strict_output_contract(action: str, packet: dict[str, Any] | None = None) -> str:
    claim = packet.get("review_claim") if isinstance(packet, dict) else None
    if action == "validate_relational_grounding" and isinstance(claim, dict) and claim.get("claim_type") == "process_edge":
        return STRICT_JSON_CONTRACT + "\n" + (
            "For this process-edge review, the only top-level fields are claim_id, "
            "edge_status, supporting_evidence, boundary_conditions, negative_evidence, "
            "and rationale. edge_status is supported, conditional, tentative, or rejected. "
            "Both evidence fields are arrays of record_id/quote evidence objects; boundary_conditions is an array of strings."
        )
    return STRICT_JSON_CONTRACT + "\n" + ACTION_OUTPUT_CONTRACTS.get(
        action,
        "Follow the field names and types shown in expected_output.",
    )

class CompletionError(RuntimeError):
    """A local model request or its response could not produce a usable object."""


class OutputTokenLimitError(CompletionError):
    """Every bounded retry ended because the response allowance was exhausted."""


class StructuredOutputUnavailableError(CompletionError):
    """The local server rejected a required grammar-backed response format."""


class PacketTooLargeError(CompletionError):
    """The orchestrator must split or explicitly retain an oversized packet."""


class ExhaustedValidationError(CompletionError):
    """All bounded attempts failed local candidate validation, not transport."""


Completion = Callable[[dict[str, Any], str | None], dict[str, Any]]


def build_chat_request(
    packet: dict[str, Any],
    *,
    model: str,
    max_tokens: int,
    thinking_budget: int | None = None,
    repair_feedback: str | None = None,
) -> dict[str, Any]:
    """Build a stateless request; retries never inherit hidden agent context."""
    public_packet = context_public_packet(packet)
    public_packet.pop("grounded_theory_results_path", None)
    instruction = (
        "Complete the following Grounded Theory packet. Return only the complete JSON object "
        "that matches expected_output.\n\nPACKET:\n"
        + json.dumps(public_packet, ensure_ascii=False, separators=(",", ":"))
    )
    if repair_feedback is not None:
        bounded_feedback = repair_feedback[:MAX_REPAIR_FEEDBACK_CHARS]
        if len(repair_feedback) > MAX_REPAIR_FEEDBACK_CHARS:
            bounded_feedback = bounded_feedback.rstrip() + " …[feedback capped]"
        instruction += (
            "\n\nA previous candidate was rejected before any state changed. Regenerate the "
            "entire JSON object, correcting the reported issue.\nVALIDATOR FEEDBACK:\n"
            + bounded_feedback
        )
    instruction += "\n\n" + strict_output_contract(
        str(public_packet.get("action", "")), public_packet
    )

    request = {
        "model": model,
        "temperature": 0,
        "top_p": 1,
        "max_tokens": max_tokens,
        "response_format": response_format_for_action(str(public_packet.get("action", ""))),
        # A disabled stage uses Qwen's non-thinking template.  Stage 2's
        # grammar-enforced budget governs its deliberately brief reasoning;
        # use medium rather than Qwen's default xhigh effort before that cap
        # applies, preserving analytic quality without uncontrolled expansion.
        "chat_template_kwargs": (
            {"enable_thinking": True, "reasoning_effort": "medium"}
            if thinking_budget is not None
            else {"enable_thinking": False}
        ),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
        ],
    }
    if thinking_budget is not None:
        # The loopback adapter moves this into SGLang's supported
        # ``custom_params.thinking_budget`` transport after authenticating the
        # stage-selected alias.  Keep this client-side assertion for adapter
        # compatibility; it must not be relied on by the OpenAI chat endpoint.
        request["max_thinking_tokens"] = thinking_budget
    return request


def _render_qwen_chat_prompt(request_body: dict[str, Any]) -> str:
    """Render the exact no-tools Qwen chat template used by this runner.

    Every local request contains exactly one system message and one user
    message. Keeping this narrow mirror of the pinned Qwen tokenizer template
    lets the client count the actual prompt token IDs before it sends the HTTP
    request. A changed request shape fails closed instead of silently falling
    back to a character heuristic.
    """
    messages = request_body.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        raise CompletionError("strict Qwen context guard requires system and user messages")
    system, user = messages
    if not (
        isinstance(system, dict)
        and isinstance(user, dict)
        and system.get("role") == "system"
        and user.get("role") == "user"
        and isinstance(system.get("content"), str)
        and isinstance(user.get("content"), str)
    ):
        raise CompletionError("strict Qwen context guard received an unsupported chat message")
    template_kwargs = request_body.get("chat_template_kwargs")
    if not isinstance(template_kwargs, dict):
        raise CompletionError("strict Qwen context guard requires chat template settings")
    thinking_enabled = template_kwargs.get("enable_thinking")
    if thinking_enabled is True:
        if template_kwargs != {"enable_thinking": True, "reasoning_effort": "medium"}:
            raise CompletionError("strict Qwen context guard requires the pinned medium thinking template")
        generation_prefix = "<|im_start|>assistant\n<think>\n"
    elif thinking_enabled is False:
        if template_kwargs != {"enable_thinking": False}:
            raise CompletionError("strict Qwen context guard requires the pinned non-thinking template")
        generation_prefix = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    else:
        raise CompletionError("strict Qwen context guard requires an explicit thinking setting")
    return (
        "<|im_start|>system\n"
        + system["content"]
        + "<|im_end|>\n<|im_start|>user\n"
        + user["content"]
        + "<|im_end|>\n"
        + generation_prefix
    )


@lru_cache(maxsize=4)
def _qwen_tokenizer(tokenizer_file: str) -> Any:
    """Load the pinned tokenizer without importing the model or its weights."""
    try:
        from tokenizers import Tokenizer
    except ImportError as error:  # pragma: no cover - runner environment supplies tokenizers.
        raise CompletionError("strict Qwen context guard requires the tokenizers package") from error
    path = Path(tokenizer_file)
    if not path.is_file():
        raise CompletionError(f"strict Qwen context guard cannot find tokenizer file: {path}")
    try:
        return Tokenizer.from_file(str(path))
    except Exception as error:  # pragma: no cover - corrupt tokenizer is an operator error.
        raise CompletionError(f"strict Qwen context guard cannot load tokenizer file: {path}") from error


def qwen_request_input_tokens(request_body: dict[str, Any], tokenizer_file: str | Path) -> int:
    """Count the rendered Qwen request, including all chat-template tokens."""
    rendered = _render_qwen_chat_prompt(request_body)
    tokenizer = _qwen_tokenizer(str(Path(tokenizer_file)))
    return len(tokenizer.encode(rendered, add_special_tokens=False).ids)


def ensure_qwen_request_within_context(
    request_body: dict[str, Any],
    *,
    tokenizer_file: str | Path,
    context_window_tokens: int,
    output_tokens: int,
    thinking_tokens: int | None,
) -> int:
    """Fail before network I/O when this exact Qwen request cannot fit.

    The strict-thinking runtime can account for its reasoning allowance
    separately from visible completion tokens, so both reservations are held
    even on server versions where one happens to count inside the other.
    """
    if context_window_tokens <= 0 or output_tokens <= 0:
        raise ValueError("context and output token budgets must be positive")
    if thinking_tokens is not None and thinking_tokens <= 0:
        raise ValueError("thinking token budget must be positive when enabled")
    input_tokens = qwen_request_input_tokens(request_body, tokenizer_file)
    reserved_generation = output_tokens + (thinking_tokens or 0)
    total = input_tokens + reserved_generation
    if total > context_window_tokens:
        raise PacketTooLargeError(
            "Qwen request would use "
            f"{input_tokens} input + {output_tokens} visible-output + "
            f"{thinking_tokens or 0} thinking = {total} tokens "
            f"(context limit {context_window_tokens}); request was not sent"
        )
    return input_tokens


def parse_output_token_budgets(value: str | None) -> dict[str, int]:
    """Read optional per-action output-cap overrides for a local-Qwen run.

    The defaults remain in force for actions not named by the caller. This
    lets a study raise one demonstrated-too-small cap without accidentally
    changing every other bounded stage.
    """
    configured = dict(DEFAULT_OUTPUT_TOKENS)
    if value is None or not value.strip():
        return configured
    for item in value.split(","):
        action, separator, raw_limit = item.partition("=")
        action = action.strip()
        if not separator or not action or not raw_limit.strip():
            raise ValueError(
                "output-token budgets must be comma-separated action=token pairs"
            )
        if action not in configured:
            raise ValueError(f"unknown output-token budget action: {action!r}")
        try:
            limit = int(raw_limit.strip())
        except ValueError as error:
            raise ValueError(
                f"output-token budget for {action!r} must be an integer"
            ) from error
        if limit <= 0:
            raise ValueError(
                f"output-token budget for {action!r} must be a positive integer"
            )
        configured[action] = limit
    return configured


def parse_output_runtime_limits(
    value: str | None, output_tokens: dict[str, int]
) -> dict[str, int]:
    """Read accepted visible-output ceilings without changing output targets."""
    parsed = dict(output_tokens)
    if value is None or not value.strip():
        parsed[REDUCE_MEMO_ACTION] += DEFAULT_OUTPUT_RUNTIME_HEADROOM_TOKENS
        return parsed
    for item in value.split(","):
        action, separator, raw_limit = item.partition("=")
        action = action.strip()
        if not separator or not action or not raw_limit.strip():
            raise ValueError(
                "output runtime limits must be comma-separated action=token pairs"
            )
        if action not in parsed:
            raise ValueError(f"unknown output runtime-limit action: {action!r}")
        try:
            limit = int(raw_limit.strip())
        except ValueError as error:
            raise ValueError(
                f"output runtime limit for {action!r} must be an integer"
            ) from error
        if limit < output_tokens[action]:
            raise ValueError(
                f"output runtime limit for {action!r} must be at least its target"
            )
        parsed[action] = limit
    return parsed


def parse_thinking_token_budgets(
    value: str | None, output_tokens: dict[str, int] | None = None
) -> dict[str, int | None]:
    """Read a complete action=positive-cap or action=disabled runner mapping."""
    if value is None or not value.strip():
        return dict(DEFAULT_THINKING_TOKENS)
    parsed: dict[str, int | None] = {}
    for item in value.split(","):
        action, separator, raw_limit = item.partition("=")
        action = action.strip()
        if not separator or not action or not raw_limit.strip():
            raise ValueError(
                "thinking-token budgets must be comma-separated action=token pairs"
            )
        raw_limit = raw_limit.strip().lower()
        if raw_limit in {"disabled", "off"}:
            limit = None
        else:
            try:
                limit = int(raw_limit)
            except ValueError as error:
                raise ValueError(
                    f"thinking-token budget for {action!r} must be a positive integer or disabled"
                ) from error
        parsed[action] = limit
    return _validated_thinking_token_budgets(parsed, output_tokens)


def parse_thinking_runtime_limits(
    value: str | None,
    thinking_tokens: dict[str, int | None],
    output_tokens: dict[str, int] | None = None,
) -> dict[str, int | None]:
    """Read per-stage accepted thinking ceilings without changing request targets."""
    if value is None or not value.strip():
        parsed = {
            action: (None if target is None else target + DEFAULT_THINKING_RUNTIME_HEADROOM_TOKENS)
            for action, target in thinking_tokens.items()
        }
    else:
        parsed: dict[str, int | None] = {}
        for item in value.split(","):
            action, separator, raw_limit = item.partition("=")
            action = action.strip()
            if not separator or not action or not raw_limit.strip():
                raise ValueError(
                    "thinking runtime limits must be comma-separated action=token pairs"
                )
            raw_limit = raw_limit.strip().lower()
            if raw_limit in {"disabled", "off"}:
                parsed[action] = None
                continue
            try:
                parsed[action] = int(raw_limit)
            except ValueError as error:
                raise ValueError(
                    f"thinking runtime limit for {action!r} must be a positive integer or disabled"
                ) from error
    outputs = output_tokens or DEFAULT_OUTPUT_TOKENS
    if set(parsed) != set(thinking_tokens):
        raise ValueError("thinking runtime limits must name exactly the configured actions")
    for action, target in thinking_tokens.items():
        limit = parsed[action]
        if target is None:
            if limit is not None:
                raise ValueError(f"disabled thinking action {action!r} cannot have a runtime limit")
            continue
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < target:
            raise ValueError(
                f"thinking runtime limit for {action!r} must be an integer at least its target"
            )
        if limit >= outputs[action]:
            raise ValueError(
                f"thinking runtime limit for {action!r} must be below its total output budget"
            )
    return parsed


def _validated_thinking_token_budgets(
    budgets: dict[str, int | None] | None,
    output_tokens: dict[str, int] | None = None,
) -> dict[str, int | None]:
    configured = dict(DEFAULT_THINKING_TOKENS if budgets is None else budgets)
    outputs = output_tokens or DEFAULT_OUTPUT_TOKENS
    missing = sorted(set(outputs) - set(configured))
    extra = sorted(set(configured) - set(outputs))
    if missing or extra:
        details = []
        if missing:
            details.append("missing actions: " + ", ".join(missing))
        if extra:
            details.append("unknown actions: " + ", ".join(extra))
        raise ValueError("invalid thinking-token budgets (" + "; ".join(details) + ")")
    for action, limit in configured.items():
        if limit is None:
            continue
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError(
                f"thinking-token budget for {action!r} must be a positive integer or None to disable"
            )
        if limit >= outputs[action]:
            raise ValueError(
                f"thinking-token budget for {action!r} ({limit}) must be below its "
                f"total output budget ({outputs[action]})"
            )
    return configured


def ensure_packet_within_budget(
    packet: dict[str, Any], input_tokens: dict[str, int] | None = None
) -> None:
    action = packet.get("action")
    budget = (input_tokens or DEFAULT_INPUT_TOKENS).get(action)
    if budget is None:
        raise CompletionError(f"no input-token budget for action: {action!r}")
    accepted_budget = budget
    if action == "validate_relational_grounding":
        accepted_budget += RELATIONAL_REVIEW_INPUT_ACCEPTANCE_HEADROOM_TOKENS
    estimate = packet_token_estimate(context_public_packet(packet))
    if estimate > accepted_budget:
        limit_description = str(budget)
        if accepted_budget != budget:
            limit_description += f" target; {accepted_budget} hard acceptance limit"
        raise PacketTooLargeError(
            f"{action} packet estimates {estimate} input tokens (limit {limit_description}); "
            "no evidence was dropped and this uncommitted transaction must be split, "
            "reduced, or explicitly retained for intervention"
        )


def _input_token_budgets_for_output_tokens(output_tokens: dict[str, int]) -> dict[str, int]:
    """Keep the established input-plus-output safety envelope when a cap grows."""
    missing = sorted(set(DEFAULT_OUTPUT_TOKENS) - set(output_tokens))
    extra = sorted(set(output_tokens) - set(DEFAULT_OUTPUT_TOKENS))
    if missing or extra:
        raise ValueError("output-token budgets must name exactly the supported actions")
    limits: dict[str, int] = {}
    for action, default_input in DEFAULT_INPUT_TOKENS.items():
        configured_output = output_tokens[action]
        default_output = DEFAULT_OUTPUT_TOKENS[action]
        if configured_output <= default_output:
            limits[action] = default_input
            continue
        limits[action] = default_input - (configured_output - default_output)
        if limits[action] <= 0:
            raise ValueError(
                f"output-token budget for {action!r} leaves no protected input context"
            )
    return limits


def parse_completion(response: dict[str, Any]) -> dict[str, Any]:
    """Extract the one JSON object from an OpenAI-compatible chat response."""
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise CompletionError("local completion response has no choices")
    choice = choices[0]
    if choice.get("finish_reason") == "length":
        raise OutputTokenLimitError("local completion reached its bounded output-token limit")
    message = choice.get("message")
    if not isinstance(message, dict):
        raise CompletionError("local completion response has no message")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise CompletionError("local completion response has no JSON content")
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as error:
        raise CompletionError(f"local completion returned invalid JSON: {error.msg}") from error
    if not isinstance(payload, dict):
        raise CompletionError("local completion result must be one JSON object")
    return payload


class LocalQwenClient:
    """A small OpenAI-compatible client for the runner-owned local SGLang server."""

    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        timeout_seconds: int,
        output_tokens: dict[str, int] | None = None,
        output_runtime_limits: dict[str, int] | None = None,
        thinking_tokens: dict[str, int | None] | None = None,
        thinking_runtime_limits: dict[str, int | None] | None = None,
        tokenizer_file: str | Path | None = None,
        context_window_tokens: int | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.output_tokens = output_tokens or DEFAULT_OUTPUT_TOKENS
        self.output_runtime_limits = parse_output_runtime_limits(
            None if output_runtime_limits is None else ",".join(
                f"{action}={limit}" for action, limit in output_runtime_limits.items()
            ),
            self.output_tokens,
        )
        self.thinking_tokens = _validated_thinking_token_budgets(
            thinking_tokens, self.output_tokens
        )
        self.thinking_runtime_limits = parse_thinking_runtime_limits(
            None, self.thinking_tokens, self.output_tokens
        ) if thinking_runtime_limits is None else parse_thinking_runtime_limits(
            ",".join(
                f"{action}={'disabled' if limit is None else limit}"
                for action, limit in thinking_runtime_limits.items()
            ),
            self.thinking_tokens,
            self.output_tokens,
        )
        if tokenizer_file is None and context_window_tokens is not None:
            raise ValueError("context_window_tokens requires tokenizer_file")
        if tokenizer_file is not None:
            self.tokenizer_file = Path(tokenizer_file)
            self.context_window_tokens = (
                MODEL_CONTEXT_WINDOW_TOKENS
                if context_window_tokens is None
                else context_window_tokens
            )
            if self.context_window_tokens <= 0:
                raise ValueError("context_window_tokens must be positive")
        else:
            # Unit-test clients and custom, non-Qwen Completion callables may
            # deliberately omit the local transport. The Slurm entry point
            # always supplies both values and therefore cannot bypass the
            # exact preflight guard.
            self.tokenizer_file = None
            self.context_window_tokens = None

    def complete(self, packet: dict[str, Any], repair_feedback: str | None = None) -> dict[str, Any]:
        action = packet.get("action")
        target_tokens = self.output_tokens.get(action) if isinstance(action, str) else None
        max_tokens = self.output_runtime_limits.get(action) if isinstance(action, str) else None
        if target_tokens is None or max_tokens is None:
            raise CompletionError(f"no output-token budget for action: {action!r}")
        thinking_budget = self._thinking_budget(packet)
        thinking_runtime_limit = self._thinking_runtime_limit(packet)
        request_body = build_chat_request(
            packet,
            model=thinking_model_alias(self.model, thinking_budget),
            max_tokens=max_tokens,
            thinking_budget=thinking_budget,
            repair_feedback=repair_feedback,
        )
        exact_input_tokens: int | None = None
        if self.tokenizer_file is not None and self.context_window_tokens is not None:
            exact_input_tokens = ensure_qwen_request_within_context(
                request_body,
                tokenizer_file=self.tokenizer_file,
                context_window_tokens=self.context_window_tokens,
                output_tokens=max_tokens,
                thinking_tokens=thinking_runtime_limit,
            )
            self._record_preflight_budget(
                packet, exact_input_tokens, target_tokens, max_tokens,
                thinking_budget, thinking_runtime_limit
            )
        started_at = time.perf_counter()
        try:
            response = self._post(request_body)
        except CompletionError as error:
            # Older SGLang builds can lack JSON mode. The deterministic validator
            # remains the authoritative contract in that compatibility fallback.
            # Stage 2 is different: a bare JSON object is exactly what allowed
            # an unbounded response to recur, so never silently drop its schema.
            if request_body["response_format"].get("type") == "json_schema":
                raise StructuredOutputUnavailableError(
                    "local server rejected the required Stage-2 JSON Schema constraint: "
                    f"{error}"
                ) from error
            if "response_format" not in str(error):
                raise
            request_body.pop("response_format", None)
            response = self._post(request_body)
        if exact_input_tokens is not None:
            self._validate_observed_usage(
                response,
                expected_input_tokens=exact_input_tokens,
                output_tokens=max_tokens,
                thinking_tokens=thinking_runtime_limit,
                thinking_target=thinking_budget,
            )
        _record_local_call_metrics(packet, response, time.perf_counter() - started_at)
        return parse_completion(response)

    def _record_preflight_budget(
        self,
        packet: dict[str, Any],
        input_tokens: int,
        output_target_tokens: int,
        output_runtime_limit: int,
        thinking_target: int | None,
        thinking_runtime_limit: int | None,
    ) -> None:
        audit = packet.get(CONTEXT_AUDIT_KEY)
        if not isinstance(audit, dict):
            return
        audit["exact_qwen_input_tokens"] = input_tokens
        audit["context_window_tokens"] = self.context_window_tokens
        audit["model_context_window_tokens"] = self.context_window_tokens
        audit["output_token_budget"] = output_target_tokens
        audit["runtime_output_token_limit"] = output_runtime_limit
        audit["reserved_visible_output_tokens"] = output_runtime_limit
        audit["requested_thinking_tokens"] = thinking_target or 0
        audit["reserved_thinking_tokens"] = thinking_runtime_limit or 0
        audit["exact_qwen_total_reserved_tokens"] = (
            input_tokens + output_runtime_limit + (thinking_runtime_limit or 0)
        )

    @staticmethod
    def _validate_observed_usage(
        response: dict[str, Any],
        *,
        expected_input_tokens: int,
        output_tokens: int,
        thinking_tokens: int | None,
        thinking_target: int | None,
    ) -> None:
        """Reject a server/template mismatch before it can mutate state."""
        usage = response.get("usage")
        if not isinstance(usage, dict):
            raise CompletionError("strict Qwen context guard requires usage metrics from the server")
        observed_input = _usage_count(usage.get("prompt_tokens"))
        observed_output = _usage_count(usage.get("completion_tokens"))
        if observed_input is None or observed_output is None:
            raise CompletionError("strict Qwen context guard requires prompt and completion token metrics")
        if observed_input != expected_input_tokens:
            raise CompletionError(
                "strict Qwen context guard observed a tokenizer/template mismatch "
                f"({observed_input} server input tokens != {expected_input_tokens} preflight tokens)"
            )
        if observed_output > output_tokens:
            raise CompletionError(
                f"server exceeded the requested output budget ({observed_output} > {output_tokens})"
            )
        details = usage.get("completion_tokens_details")
        details = details if isinstance(details, dict) else {}
        observed_thinking = _usage_count(details.get("reasoning_tokens"))
        if observed_thinking is None:
            observed_thinking = _usage_count(usage.get("reasoning_tokens"))
        if thinking_tokens is not None:
            if observed_thinking is None:
                raise CompletionError("strict Qwen context guard requires reasoning-token metrics")
            if observed_thinking > thinking_tokens:
                raise CompletionError(
                    "server exceeded the accepted thinking runtime allowance "
                    f"({observed_thinking} > {thinking_tokens}; target {thinking_target})"
                )

    def _thinking_budget(self, packet: dict[str, Any]) -> int | None:
        action = packet.get("action")
        if not isinstance(action, str) or action not in self.thinking_tokens:
            raise CompletionError(f"no thinking-token budget for action: {action!r}")
        return self.thinking_tokens[action]

    def _thinking_runtime_limit(self, packet: dict[str, Any]) -> int | None:
        action = packet.get("action")
        if not isinstance(action, str) or action not in self.thinking_runtime_limits:
            raise CompletionError(f"no thinking runtime limit for action: {action!r}")
        return self.thinking_runtime_limits[action]

    def _post(self, request_body: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(request_body, ensure_ascii=False).encode("utf-8")
        request = Request(
            self.endpoint,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise CompletionError(f"local completion HTTP {error.code}: {detail}") from error
        except URLError as error:
            raise CompletionError(f"local completion connection error: {error.reason}") from error
        except TimeoutError as error:
            raise CompletionError("local completion request timed out") from error
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as error:
            raise CompletionError("local completion server returned non-JSON data") from error
        if not isinstance(value, dict):
            raise CompletionError("local completion server returned a non-object response")
        return value


def _usage_count(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _record_local_call_metrics(
    packet: dict[str, Any], response: dict[str, Any], wall_time_seconds: float
) -> None:
    """Attach observed local-call usage to the private task audit receipt."""
    audit = packet.get(CONTEXT_AUDIT_KEY)
    if not isinstance(audit, dict):
        return
    usage = response.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    completion_details = usage.get("completion_tokens_details")
    completion_details = completion_details if isinstance(completion_details, dict) else {}
    choices = response.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    thinking_tokens = _usage_count(completion_details.get("reasoning_tokens"))
    if thinking_tokens is None:
        thinking_tokens = _usage_count(usage.get("reasoning_tokens"))
    metric = {
        "wall_time_seconds": round(wall_time_seconds, 3),
        "input_tokens": _usage_count(usage.get("prompt_tokens")),
        "output_tokens": _usage_count(usage.get("completion_tokens")),
        "thinking_tokens": thinking_tokens,
        "finish_reason": choice.get("finish_reason") if isinstance(choice.get("finish_reason"), str) else None,
    }
    calls = audit.setdefault("llm_calls", [])
    if not isinstance(calls, list):
        calls = []
        audit["llm_calls"] = calls
    calls.append(metric)
    audit["llm_call_count"] = len(calls)
    audit["llm_wall_time_seconds"] = round(
        sum(item.get("wall_time_seconds", 0.0) for item in calls if isinstance(item, dict)), 3
    )


class PrimeQwenClient:
    """Run one disposable Prime transaction against the runner-owned Qwen server.

    Prime receives no tools, skills, context files, session, or prior messages.
    The run directory contains only its local model configuration; project JSON
    remains the sole persistent analytic memory.
    """

    provider_name = "grounded-theory-local-qwen"

    def __init__(
        self,
        *,
        prime_command: str | Path,
        prime_agent_dir: str | Path,
        cwd: str | Path,
        endpoint: str,
        model: str,
        timeout_seconds: int,
        output_tokens: dict[str, int] | None = None,
        input_tokens: dict[str, int] | None = None,
        thinking_tokens: dict[str, int | None] | None = None,
    ) -> None:
        self.prime_command = Path(prime_command)
        self.prime_agent_dir = Path(prime_agent_dir)
        self.cwd = Path(cwd)
        self.endpoint = endpoint
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.output_tokens = output_tokens or DEFAULT_OUTPUT_TOKENS
        self.input_tokens = input_tokens or DEFAULT_INPUT_TOKENS
        self.thinking_tokens = _validated_thinking_token_budgets(
            thinking_tokens, self.output_tokens
        )

    def complete(self, packet: dict[str, Any], repair_feedback: str | None = None) -> dict[str, Any]:
        ensure_packet_within_budget(packet, self.input_tokens)
        action = packet.get("action")
        max_tokens = self.output_tokens.get(action) if isinstance(action, str) else None
        if max_tokens is None:
            raise CompletionError(f"no output-token budget for action: {action!r}")
        thinking_budget = self._thinking_budget(packet)
        transaction_dir = self.prime_agent_dir / "transactions" / uuid.uuid4().hex
        provider_model = thinking_model_alias(self.model, thinking_budget)
        self._write_model_config(transaction_dir, max_tokens, provider_model)
        try:
            response = subprocess.run(
                self._command(
                    self._instruction(packet, repair_feedback),
                    provider_model,
                    transaction_dir / "prime-daemon.sock",
                ),
                cwd=self.cwd,
                env=self._environment(transaction_dir),
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except OSError as error:
            raise CompletionError(f"could not start Prime: {error}") from error
        except subprocess.TimeoutExpired as error:
            raise CompletionError("Prime transaction timed out") from error
        if response.returncode != 0:
            detail = response.stderr.strip() or response.stdout.strip() or "no diagnostic"
            raise CompletionError(f"Prime transaction failed ({response.returncode}): {detail}")
        return _parse_json_text(response.stdout)

    def _command(
        self, instruction: str, provider_model: str, daemon_socket: Path
    ) -> list[str]:
        command = [str(self.prime_command)]
        if self.prime_command.exists() and not os.access(self.prime_command, os.X_OK):
            command.insert(0, "bash")
        return [
            *command,
            "--print",
            "--mode",
            "text",
            "--no-session",
            "--no-tools",
            "--no-skills",
            "--no-extensions",
            "--no-context-files",
            # Prime otherwise connects to a shared default daemon, which can
            # retain a prior run's model configuration.  A per-transaction
            # socket makes the local configuration and adapter endpoint
            # unambiguous, while --offline prevents any remote provider setup.
            "--offline",
            "--daemon-socket",
            str(daemon_socket),
            "--cwd",
            str(self.cwd),
            "--provider",
            self.provider_name,
            "--model",
            provider_model,
            "--thinking",
            "medium",
            "--system-prompt",
            SYSTEM_PROMPT,
            instruction,
        ]

    def _environment(self, transaction_dir: Path) -> dict[str, str]:
        environment = os.environ.copy()
        environment["PRIME_AGENT_CODING_AGENT_DIR"] = str(transaction_dir)
        environment["PRIME_AGENT_SESSION_DIR"] = str(transaction_dir / "sessions")
        return environment

    def _write_model_config(
        self, transaction_dir: Path, max_tokens: int, provider_model: str
    ) -> None:
        write_json(
            transaction_dir / "models.json",
            {
                "providers": {
                    self.provider_name: {
                        "baseUrl": _prime_base_url(self.endpoint),
                        "api": "openai-completions",
                        "apiKey": "local",
                        "authHeader": False,
                        "compat": {
                            "supportsDeveloperRole": False,
                            "supportsReasoningEffort": True,
                        },
                        "models": [
                            {
                                "id": provider_model,
                                "name": self.model,
                                "reasoning": True,
                                "input": ["text"],
                                "contextWindow": 32768,
                                "maxTokens": max_tokens,
                                "cost": {
                                    "input": 0,
                                    "output": 0,
                                    "cacheRead": 0,
                                    "cacheWrite": 0,
                                },
                            }
                        ],
                    }
                }
            },
        )

    @staticmethod
    def _instruction(packet: dict[str, Any], repair_feedback: str | None) -> str:
        request = build_chat_request(
            packet,
            model="unused-by-prime",
            max_tokens=0,
            repair_feedback=repair_feedback,
        )
        return str(request["messages"][1]["content"])

    def _thinking_budget(self, packet: dict[str, Any]) -> int | None:
        action = packet.get("action")
        if not isinstance(action, str) or action not in self.thinking_tokens:
            raise CompletionError(f"no thinking-token budget for action: {action!r}")
        return self.thinking_tokens[action]


def _prime_base_url(endpoint: str) -> str:
    suffix = "/chat/completions"
    normalized = endpoint.rstrip("/")
    return normalized[: -len(suffix)] if normalized.endswith(suffix) else normalized


def _parse_json_text(output: str) -> dict[str, Any]:
    content = output.strip()
    if content.startswith("```json") and content.endswith("```"):
        content = content[len("```json") : -3].strip()
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as error:
        raise CompletionError(f"Prime returned invalid JSON: {error.msg}") from error
    if not isinstance(payload, dict):
        raise CompletionError("Prime result must be one JSON object")
    return payload


def _runtime_token_budgets(
    complete: Completion,
) -> tuple[dict[str, int], dict[str, int | None]]:
    """Use a client's configured limits, with deterministic test defaults."""
    owner = getattr(complete, "__self__", None)
    output_tokens = getattr(owner, "output_tokens", DEFAULT_OUTPUT_TOKENS)
    thinking_tokens = getattr(owner, "thinking_tokens", DEFAULT_THINKING_TOKENS)
    if not isinstance(output_tokens, dict) or not isinstance(thinking_tokens, dict):
        return dict(DEFAULT_OUTPUT_TOKENS), dict(DEFAULT_THINKING_TOKENS)
    return dict(output_tokens), _validated_thinking_token_budgets(
        thinking_tokens, output_tokens
    )


def _runtime_output_limits(
    complete: Completion, output_tokens: dict[str, int]
) -> dict[str, int]:
    """Read a bound client's accepted visible-output limits for audit use."""
    owner = getattr(complete, "__self__", None)
    limits = getattr(owner, "output_runtime_limits", None)
    if not isinstance(limits, dict):
        return dict(output_tokens)
    return parse_output_runtime_limits(
        ",".join(f"{action}={limit}" for action, limit in limits.items()),
        output_tokens,
    )


def _annotate_runtime_budget(
    packet: dict[str, Any],
    output_tokens: dict[str, int],
    thinking_tokens: dict[str, int | None],
    output_runtime_limits: dict[str, int] | None = None,
) -> None:
    """Persist the applied cap, never just an effort label, in private audit data."""
    audit = packet.get(CONTEXT_AUDIT_KEY)
    action = packet.get("action")
    if not isinstance(audit, dict) or not isinstance(action, str):
        return
    if action not in output_tokens or action not in thinking_tokens:
        return
    audit["output_token_budget"] = output_tokens[action]
    if output_runtime_limits is not None:
        audit["runtime_output_token_limit"] = output_runtime_limits[action]
    thinking_budget = thinking_tokens[action]
    audit["thinking_token_budget"] = thinking_budget
    if thinking_budget is not None:
        audit["thinking_mode"] = "strict"
        audit["thinking_budget_transport"] = "custom_params.thinking_budget"
        audit["model_configuration"] = (
            "Prime stateless; strict thinking enabled; "
            f"thinking token cap={thinking_budget}"
        )
    else:
        audit["thinking_mode"] = "disabled"
        audit["thinking_budget_transport"] = "chat_template_kwargs.enable_thinking=false"
        audit["model_configuration"] = "Prime stateless; Qwen thinking disabled"


def _runtime_audited_completion(complete: Completion) -> Completion:
    output_tokens, thinking_tokens = _runtime_token_budgets(complete)
    output_runtime_limits = _runtime_output_limits(complete, output_tokens)

    def annotated(packet: dict[str, Any], repair_feedback: str | None = None) -> dict[str, Any]:
        _annotate_runtime_budget(
            packet, output_tokens, thinking_tokens, output_runtime_limits
        )
        return complete(packet, repair_feedback)

    return annotated


def execute_stage(
    project: GroundedTheoryProject,
    complete: Completion,
    *,
    packet_path: str | Path,
    result_path: str | Path,
    attempts: int,
    review_parallelism: int = 1,
    integration_parallelism: int = 1,
    packet_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Complete one stage without ever treating a failed candidate as analysed."""
    if attempts < 1:
        raise ValueError("attempts must be positive")
    if review_parallelism < 1:
        raise ValueError("review_parallelism must be positive")
    if review_parallelism > MAX_REVIEW_PARALLELISM:
        raise ValueError(
            f"review_parallelism must not exceed {MAX_REVIEW_PARALLELISM}, "
            "the configured local Qwen request limit"
        )
    output_tokens, thinking_tokens = _runtime_token_budgets(complete)
    output_runtime_limits = _runtime_output_limits(complete, output_tokens)
    input_tokens = _input_token_budgets_for_output_tokens(output_tokens)
    complete = _runtime_audited_completion(complete)
    if packet_override is None:
        packet = project.task_packet()
    else:
        expected_action = project.next_action().get("action")
        if packet_override.get("action") != expected_action:
            raise ValueError(
                "packet_override action must match the project's next action "
                f"({packet_override.get('action')!r} != {expected_action!r})"
            )
        # A validation replay may freeze the context envelope, but must never
        # mutate that artifact while adding runtime budget/audit annotations.
        packet = deepcopy(packet_override)
    if packet.get("action") == VALIDATION_ACTION:
        return _execute_prompt_adaptation_validation(
            project,
            complete,
            packet=packet,
            packet_path=Path(packet_path),
            result_path=Path(result_path),
            attempts=attempts,
            review_parallelism=review_parallelism,
        )
    _annotate_runtime_budget(
        packet, output_tokens, thinking_tokens, output_runtime_limits
    )
    if packet.get("action") == "theoretical_integration":
        try:
            return execute_hierarchical_integration(
                project,
                complete,
                packet_path=Path(packet_path),
                result_path=Path(result_path),
                attempts=attempts,
                parallelism=integration_parallelism,
                validate=validate_tolerant_stage_json_schema,
                ensure_budget=lambda candidate: (
                    _annotate_runtime_budget(
                        candidate, output_tokens, thinking_tokens,
                        output_runtime_limits,
                    ),
                    ensure_packet_within_budget(candidate, input_tokens),
                )[1],
            )
        except PacketTooLargeError:
            raise
        except (RuntimeError, ValueError) as error:
            raise CompletionError(str(error)) from error
    if packet.get("action") == "validate_relational_grounding" and review_parallelism > 1:
        review_packets = _pending_relational_review_packets(project, packet)
        if len(review_packets) > 1:
            return _execute_parallel_relational_claim_reviews(
                project,
                review_packets,
                complete,
                packet_path=Path(packet_path),
                result_path=Path(result_path),
                attempts=attempts,
                parallelism=review_parallelism,
            )
    audit = packet.get(CONTEXT_AUDIT_KEY)
    if isinstance(audit, dict):
        audit["estimated_input_tokens"] = packet_token_estimate(context_public_packet(packet))
    write_json(Path(packet_path), packet)
    _record_context_audit(project, packet, status="prepared", packet_path=Path(packet_path))
    try:
        ensure_packet_within_budget(packet, input_tokens)
    except PacketTooLargeError:
        if packet.get("action") != "validate_relational_grounding":
            raise
        return _execute_batched_relational_review(
            project,
            packet,
            complete,
            packet_path=Path(packet_path),
            result_path=Path(result_path),
            attempts=attempts,
            parallelism=review_parallelism,
        )
    feedback: str | None = None
    failures: list[str] = []
    destination = Path(result_path)
    adaptation_snapshot = project.capture_prompt_adaptation_snapshot(packet)

    output_limited_only = True
    validation_failures_only = True
    for attempt in range(1, attempts + 1):
        try:
            candidate = complete(packet, feedback)
            raw_candidate = deepcopy(candidate)
            candidate, ignored_fields = _discard_stage_output_extra_fields(packet, candidate)
            if ignored_fields:
                write_json(
                    _raw_attempt_path(destination, attempt),
                    {
                        "candidate": raw_candidate,
                        "normalization": {"ignored_top_level_fields": ignored_fields},
                    },
                )
            evidence_quote_repairs = _repair_open_coding_evidence_quotes(
                project, packet, candidate
            )
            validate_stage_json_schema(packet, candidate)
            citation_repairs = _repair_exact_packet_evidence_citations(
                project, packet, candidate
            )
            evidence_feedback = _packet_evidence_alignment_feedback(
                project, packet, candidate
            )
            if evidence_feedback is not None:
                raise ValueError(evidence_feedback)
            novelty_feedback, expansion_ids = _open_coding_novelty_feedback(
                project,
                packet,
                candidate,
                complete,
                attempts=attempts,
                packet_path=Path(packet_path),
                result_path=Path(result_path),
            )
            if novelty_feedback is not None:
                _expand_open_packet_context(packet, project, expansion_ids)
                raise ValueError(novelty_feedback)
            write_json(_attempt_path(destination, attempt), candidate)
            accepted = project.submit_agent_result(candidate)
            if adaptation_snapshot is not None and packet.get("action") == "relational_process_analysis":
                project.attach_prompt_adaptation_snapshot(
                    adaptation_snapshot,
                    staged=accepted.get("accepted_action") == "relational_analysis_staged",
                )
        except OutputTokenLimitError as error:
            feedback = str(error)
            failures.append(f"attempt {attempt}: {feedback}")
            if attempt < attempts:
                time.sleep(min(attempt, 3))
            continue
        except StructuredOutputUnavailableError:
            # A grammar rejection cannot improve through candidate feedback.
            # Keep its dedicated runner exit code so the shell does not retry
            # the identical unsupported request.
            raise
        except (CompletionError, ValueError) as error:
            output_limited_only = False
            if isinstance(error, CompletionError):
                validation_failures_only = False
            feedback = str(error)
            failures.append(f"attempt {attempt}: {feedback}")
            if attempt < attempts:
                time.sleep(min(attempt, 3))
            continue
        write_json(destination, candidate)
        _record_context_audit(
            project,
            packet,
            status="accepted",
            packet_path=Path(packet_path),
            result_path=destination,
            extra={
                "accepted_updates": _accepted_update_counts(candidate),
                "evidence_citation_repairs": citation_repairs,
                "evidence_quote_repairs": evidence_quote_repairs,
            },
        )
        return {
            "attempt": attempt,
            "accepted": accepted,
            "action": packet["action"],
            "failures_before_success": failures,
            "evidence_citation_repairs": citation_repairs,
            "evidence_quote_repairs": evidence_quote_repairs,
        }
    message = "; ".join(failures)
    if output_limited_only:
        raise OutputTokenLimitError(message)
    if validation_failures_only:
        raise ExhaustedValidationError(message)
    raise CompletionError(message)


def _packet_evidence_alignment_feedback(
    project: GroundedTheoryProject,
    packet: dict[str, Any],
    candidate: dict[str, Any],
) -> str | None:
    """Give a repairable record-ID error before a state submission fails.

    Adjacent SoCRATES turns repeat one another's public language.  A model can
    therefore copy an exact quote while attaching it to the neighboring turn's
    record ID.  Exact, deterministic remaps are applied first by
    ``_repair_exact_packet_evidence_citations``.  This function handles all
    remaining cases by returning a precise repair request instead of allowing
    a malformed candidate to reach project state.
    """
    if packet.get("action") == "theoretical_integration":
        return _final_integration_evidence_alignment_feedback(project, packet, candidate)
    if packet.get("action") not in {
        "open_coding", "open_coding_coverage_audit", "relational_process_analysis",
    }:
        return None
    packet_records = packet.get("records")
    if not isinstance(packet_records, list):
        return None
    allowed_ids = [
        item.get("id") for item in packet_records
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    ]
    if not allowed_ids:
        return None
    source_by_id = {record.id: record.text for record in project.state.records}
    allowed_sources = {
        record_id: source_by_id[record_id]
        for record_id in allowed_ids if record_id in source_by_id
    }
    for path, evidence in _candidate_evidence_items(candidate):
        record_id = evidence.get("record_id")
        quote = evidence.get("quote")
        if not isinstance(record_id, str) or not isinstance(quote, str):
            continue
        source = allowed_sources.get(record_id)
        if source is None:
            continue
        try:
            align_evidence_quote(quote, source)
            continue
        except ValueError:
            pass
        matching_ids: list[str] = []
        for candidate_id, candidate_source in allowed_sources.items():
            if candidate_id == record_id:
                continue
            try:
                align_evidence_quote(quote, candidate_source)
                matching_ids.append(candidate_id)
            except ValueError:
                continue
        location = f"{path}.record_id"
        if matching_ids:
            if packet.get("action") == "open_coding_coverage_audit":
                return (
                    f"Coverage judgment evidence at {location} cites {record_id}, but its quote does not occur "
                    f"in that record and aligns to supplied record(s) {', '.join(matching_ids)}. "
                    f"The judgment record ID is fixed: keep {record_id} and replace the quote with one exact "
                    f"continuous quote from {record_id}; do not change the evidence record ID."
                )
            return (
                f"Evidence at {location} cites {record_id}, but its quote does not occur "
                f"in that record and aligns to supplied record(s) {', '.join(matching_ids)}. "
                f"Do not move text between adjacent turns: either cite the matching record ID "
                f"or replace it with one exact continuous quote from {record_id}."
            )
        return (
            f"Evidence at {location} cites {record_id}, but its quote does not align to that "
            f"record. Replace it with one exact continuous quote copied only from {record_id}; "
            "do not paraphrase or cite an adjacent turn."
        )
    return None


def _final_integration_evidence_alignment_feedback(
    project: GroundedTheoryProject,
    packet: dict[str, Any],
    candidate: dict[str, Any],
) -> str | None:
    """Normalize final quotes when their cited corpus record exists.

    Integration uses citation IDs for traceability. A final synthesis may
    paraphrase a retrieved source; retain the real citation and persist a
    deterministic contiguous source excerpt instead of rejecting the whole
    theory transaction. Unknown record IDs remain repairable errors.
    """
    source_by_id = {record.id: record.text for record in project.state.records}
    for path, evidence in _candidate_evidence_items(candidate):
        record_id = evidence.get("record_id")
        quote = evidence.get("quote")
        if not isinstance(record_id, str):
            continue
        source = source_by_id.get(record_id)
        if source is None:
            return (
                f"Evidence at {path}.record_id cites unknown corpus record {record_id}; "
                "remove that evidence item or cite an existing record ID."
            )
        if isinstance(quote, str) and _exact_normalized_quote_in_source(quote, source):
            continue
        words = list(re.finditer(r"\S+", source))
        if not words:
            return f"Evidence at {path}.record_id cites record {record_id}, which has no source text."
        evidence["quote"] = source[:words[min(len(words), 24) - 1].end()]
    return None


def _repair_exact_packet_evidence_citations(
    project: GroundedTheoryProject,
    packet: dict[str, Any],
    candidate: dict[str, Any],
) -> list[dict[str, str]]:
    """Correct provably copied record IDs while retaining an audit trail.

    An evidence quote can recur verbatim across several adjacent conversation
    turns.  When the cited ID lacks that quote but one or more packet records
    contain the *entire normalized quote*, the evidence text itself determines
    that the ID is wrong.  Replace only that ID; never rewrite a claim, quote,
    or a partial/fuzzy match.  With duplicate exact sources, choose the closest
    record in the packet order, which is deterministic and tends to retain the
    local conversational context.  The correction is returned to the caller
    and recorded in the accepted context receipt.
    """
    if packet.get("action") not in {"open_coding", "relational_process_analysis"}:
        return []
    packet_records = packet.get("records")
    if not isinstance(packet_records, list):
        return []
    allowed_ids = [
        item.get("id") for item in packet_records
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    ]
    if not allowed_ids:
        return []
    source_by_id = {record.id: record.text for record in project.state.records}
    allowed_sources = {
        record_id: source_by_id[record_id]
        for record_id in allowed_ids if record_id in source_by_id
    }
    repairs: list[dict[str, str]] = []
    for path, evidence in _candidate_evidence_items(candidate):
        record_id = evidence.get("record_id")
        quote = evidence.get("quote")
        if (
            not isinstance(record_id, str)
            or not isinstance(quote, str)
            or record_id not in allowed_sources
        ):
            continue
        if _exact_normalized_quote_in_source(quote, allowed_sources[record_id]):
            continue
        matching_ids = [
            candidate_id for candidate_id in allowed_ids
            if candidate_id != record_id
            and candidate_id in allowed_sources
            and _exact_normalized_quote_in_source(quote, allowed_sources[candidate_id])
        ]
        if not matching_ids:
            continue
        cited_position = allowed_ids.index(record_id)
        replacement = min(
            matching_ids,
            key=lambda candidate_id: (
                abs(allowed_ids.index(candidate_id) - cited_position),
                allowed_ids.index(candidate_id),
            ),
        )
        evidence["record_id"] = replacement
        repairs.append(
            {
                "evidence_path": f"{path}.record_id",
                "from_record_id": record_id,
                "to_record_id": replacement,
                "match_mode": "source_located",
            }
        )
    return repairs


def _repair_open_coding_evidence_quotes(
    project: GroundedTheoryProject,
    packet: dict[str, Any],
    candidate: dict[str, Any],
) -> list[dict[str, Any]]:
    """Replace safely recoverable Stage-1 quote formatting with source text.

    The compact Stage-1 contract requires a quote no longer than 24 words and
    forbids ellipses because they can stitch unrelated phrases. This repairs
    only two provable formatting slips before validation: an overlong quote
    that occurs as one exact source excerpt, or an ellipsis quote containing a
    wholly exact source segment. The replacement is copied directly from the
    cited record and remains contiguous. Anything fuzzy, absent from the cited
    source, or otherwise ambiguous still follows the normal retry path.
    """
    if packet.get("action") not in {"open_coding", "open_coding_coverage_audit"}:
        return []
    packet_records = packet.get("records")
    if not isinstance(packet_records, list):
        return []
    allowed_ids = {
        item.get("id") for item in packet_records
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    source_by_id = {record.id: record.text for record in project.state.records}
    repairs: list[dict[str, Any]] = []
    for path, evidence in _candidate_evidence_items(candidate):
        record_id = evidence.get("record_id")
        quote = evidence.get("quote")
        if (
            not isinstance(record_id, str)
            or not isinstance(quote, str)
            or record_id not in allowed_ids
            or record_id not in source_by_id
        ):
            continue
        source = source_by_id[record_id]
        replacement: str | None = None
        repair_type: str | None = None
        if "..." in quote or "…" in quote:
            exact_segments = [
                segment for segment in _ellipsis_segments(quote)
                if _exact_normalized_quote_in_source(segment, source)
            ]
            if exact_segments:
                chosen = max(exact_segments, key=_normalized_evidence_length)
                replacement = _compact_source_excerpt(chosen, source, limit=24)
                repair_type = "select_exact_contiguous_ellipsis_segment"
        elif len(quote.split()) > 24 and _exact_normalized_quote_in_source(quote, source):
            replacement = _compact_source_excerpt(quote, source, limit=24)
            repair_type = "truncate_exact_contiguous_excerpt"
        if replacement is None or replacement == quote:
            continue
        evidence["quote"] = replacement
        repairs.append(
            {
                "evidence_path": f"{path}.quote",
                "record_id": record_id,
                "repair_type": repair_type,
                "source_word_count": len(quote.split()),
                "replacement_word_count": len(replacement.split()),
            }
        )
    return repairs


def _ellipsis_segments(value: str) -> list[str]:
    """Extract possible contiguous source spans from literal ellipses."""
    return [segment.strip(" \t\n\\\"'") for segment in value.replace("…", "...").split("...")]


def _normalized_evidence_length(value: str) -> int:
    return sum(character.isalnum() for character in value)


def _compact_source_excerpt(quote: str, source: str, *, limit: int) -> str:
    """Return at most ``limit`` words copied from an exact source excerpt."""
    excerpt = align_evidence_quote(quote, source)
    words = list(re.finditer(r"\S+", excerpt))
    if len(words) <= limit:
        return excerpt
    return excerpt[:words[limit - 1].end()]


def _exact_normalized_quote_in_source(quote: str, source: str) -> bool:
    """Return whether every quoted segment can be located in order in source."""
    try:
        align_evidence_quote(quote, source)
    except ValueError:
        return False
    return True


def _candidate_evidence_items(
    value: Any, path: str = "$"
) -> list[tuple[str, dict[str, Any]]]:
    """Return raw evidence-shaped objects in stable JSON traversal order."""
    found: list[tuple[str, dict[str, Any]]] = []
    if isinstance(value, dict):
        if "record_id" in value and "quote" in value:
            found.append((path, value))
        for key, item in value.items():
            found.extend(_candidate_evidence_items(item, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_candidate_evidence_items(item, f"{path}[{index}]"))
    return found


def _execute_prompt_adaptation_validation(
    project: GroundedTheoryProject,
    complete: Completion,
    *,
    packet: dict[str, Any],
    packet_path: Path,
    result_path: Path,
    attempts: int,
    review_parallelism: int,
) -> dict[str, Any]:
    """Replay parent/candidate on immutable snapshots, then blind-evaluate them.

    Replays live under a dedicated workspace and are never submitted to the
    source project.  Stable filenames make completed replays and pairwise
    rounds reusable after interruption.
    """
    candidate = project.state.prompt_adaptation_state.get("pending_candidate")
    if not isinstance(candidate, dict) or candidate.get("phase") != "validation":
        raise CompletionError("no prompt-adaptation candidate is ready for validation")
    write_json(packet_path, packet)
    validation_root = project.root / "prompt_adaptation" / "validations" / candidate["id"]
    validation: list[dict[str, Any]] = []
    for batch_id, snapshot_id in zip(candidate["validation_batches"], candidate["validation_snapshot_ids"]):
        snapshot_path = project.root / "prompt_adaptation" / "snapshots" / f"{snapshot_id}.json"
        if not snapshot_path.exists():
            raise CompletionError(f"missing frozen validation snapshot: {snapshot_id}")
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        if snapshot.get("snapshot_id") != snapshot_id:
            raise CompletionError(f"validation snapshot identity mismatch: {snapshot_id}")
        batch_root = validation_root / snapshot_id
        context_groups = _validation_stage2_context_groups(
            project,
            snapshot,
            candidate["parent_overlay"],
            candidate["candidate_overlay"],
            batch_root,
        )
        rounds: list[dict[str, Any]] = []
        group_audits: list[dict[str, Any]] = []
        for group in context_groups:
            group_root = (
                batch_root
                if len(context_groups) == 1
                else batch_root / "record-groups" / group["id"]
            )
            parent = _replay_stage2_snapshot(
                project, snapshot, candidate["parent_overlay"], group_root / "parent",
                complete=complete, attempts=attempts, review_parallelism=review_parallelism,
                initial_packet=group["parent_packet"],
            )
            contender = _replay_stage2_snapshot(
                project, snapshot, candidate["candidate_overlay"], group_root / "candidate",
                complete=complete, attempts=attempts, review_parallelism=review_parallelism,
                initial_packet=group["candidate_packet"],
            )
            group_rounds = _blind_pairwise_rounds(
                group_root,
                snapshot_id=f"{snapshot_id}:{group['id']}",
                candidate_id=candidate["id"],
                random_seed=project.state.config.prompt_adaptation.random_seed,
                round_count=project.state.config.prompt_adaptation.pairwise_evaluation_rounds,
                parent=parent,
                contender=contender,
                complete=complete,
                attempts=attempts,
            )
            rounds.extend({**item, "context_group": group["id"]} for item in group_rounds)
            group_audits.append({
                "id": group["id"],
                "record_ids": group["record_ids"],
                "shared_context_sha256": group["envelope"]["shared_context_sha256"],
                "shared_context_mode": group["envelope"]["context_mode"],
            })
        validation.append({
            "batch_id": batch_id,
            "snapshot_id": snapshot_id,
            "rounds": rounds,
            "context_groups": group_audits,
        })
    accepted = project.submit_prompt_adaptation_validation(validation)
    outcome = {
        "action": VALIDATION_ACTION,
        "candidate_id": candidate["id"],
        "validation": validation,
        "accepted": accepted,
    }
    write_json(result_path, outcome)
    return outcome


def _validation_stage2_context_groups(
    source_project: GroundedTheoryProject,
    snapshot: dict[str, Any],
    parent_overlay: dict[str, Any],
    candidate_overlay: dict[str, Any],
    root: Path,
) -> list[dict[str, Any]]:
    """Select one fair envelope, or split the frozen raw batch for both arms.

    Splitting is a last context-management escalation.  Each leaf gets a fresh
    immutable pre-batch state with every *other* record in the original batch
    marked analysed only inside its sandbox, so it processes exactly its own
    raw-record partition.  No source state or held-out snapshot is changed.
    """
    full_ids = list(snapshot["record_ids"])

    def build(record_ids: list[str], label: str, *, full: bool = False) -> list[dict[str, Any]]:
        group_root = root if full else root / "record-groups" / label
        parent_packet, candidate_packet, envelope = _shared_validation_stage2_packets(
            source_project,
            snapshot,
            parent_overlay,
            candidate_overlay,
            group_root,
            record_ids=record_ids,
        )
        if envelope["within_stage2_input_budget"]:
            return [{
                "id": "group-001" if full else label,
                "record_ids": record_ids,
                "parent_packet": parent_packet,
                "candidate_packet": candidate_packet,
                "envelope": envelope,
            }]
        if len(record_ids) == 1:
            raise PacketTooLargeError(
                "one Stage-2 record plus the complete shared validation inventory exceeds the protected input budget; "
                "the held-out comparison is retained without running either arm"
            )
        midpoint = len(record_ids) // 2
        return [
            *build(record_ids[:midpoint], f"{label}-0"),
            *build(record_ids[midpoint:], f"{label}-1"),
        ]

    groups = build(full_ids, "part", full=True)
    if len(groups) > 1:
        _record_prompt_adaptation_context_audit(
            source_project,
            {
                "stage": VALIDATION_ACTION,
                "status": "shared_envelope_record_batch_split",
                "snapshot_id": snapshot.get("snapshot_id"),
                "group_count": len(groups),
                "groups": [
                    {"id": item["id"], "record_ids": item["record_ids"], "shared_context_sha256": item["envelope"]["shared_context_sha256"]}
                    for item in groups
                ],
                "rule": "The same deterministic raw-record partition is replayed for parent and candidate; every original record appears in exactly one group.",
            },
        )
    return groups


def _shared_validation_stage2_packets(
    source_project: GroundedTheoryProject,
    snapshot: dict[str, Any],
    parent_overlay: dict[str, Any],
    candidate_overlay: dict[str, Any],
    root: Path,
    record_ids: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Freeze one non-prompt Stage-2 context for both arms of an A/B replay.

    An overlay may make one rendered prompt slightly longer.  Letting that arm
    independently select full versus compact inventories would turn a context
    policy difference into a spurious strategy win.  Instead we render both
    strategies from the immutable pre-batch state, select the more compact
    complete representation when either needs it, and persist the result.
    Only ``prompt`` and prompt provenance differ between the returned packets.
    """
    root.mkdir(parents=True, exist_ok=True)
    selected_record_ids = list(record_ids if record_ids is not None else snapshot["record_ids"])
    parent_rendered = _render_snapshot_stage2_packet(
        source_project, snapshot, parent_overlay, record_ids=selected_record_ids
    )
    candidate_rendered = _render_snapshot_stage2_packet(
        source_project, snapshot, candidate_overlay, record_ids=selected_record_ids
    )
    validation_input_limit = DEFAULT_INPUT_TOKENS["relational_process_analysis"]
    # A validation replay can run under a smaller protected envelope than the
    # production renderer's normal target.  Use the same deterministic
    # structural-index tier for both arms before splitting raw records; never
    # let prompt length decide which arm loses analytical context.
    if max(
        packet_token_estimate(context_public_packet(parent_rendered)),
        packet_token_estimate(context_public_packet(candidate_rendered)),
    ) > validation_input_limit:
        parent_rendered = _render_snapshot_stage2_packet(
            source_project, snapshot, parent_overlay,
            record_ids=selected_record_ids, force_structural_context=True,
        )
        candidate_rendered = _render_snapshot_stage2_packet(
            source_project, snapshot, candidate_overlay,
            record_ids=selected_record_ids, force_structural_context=True,
        )
    identity = {
        "version": 1,
        "snapshot_id": snapshot.get("snapshot_id"),
        "input_packet_sha256": snapshot.get("input_packet_sha256"),
        "parent_strategy_sha256": parent_overlay.get("sha256"),
        "candidate_strategy_sha256": candidate_overlay.get("sha256"),
        "record_ids": selected_record_ids,
    }
    artifact_path = root / "shared-context.json"
    stored: dict[str, Any] | None = None
    if artifact_path.exists():
        loaded = json.loads(artifact_path.read_text(encoding="utf-8"))
        if loaded.get("identity") != identity:
            raise CompletionError("shared validation context was reused for different strategies or snapshot")
        if not isinstance(loaded.get("shared_context"), dict):
            raise CompletionError("shared validation context artifact is malformed")
        if canonical_hash(loaded["shared_context"]) != loaded.get("shared_context_sha256"):
            raise CompletionError("shared validation context artifact hash mismatch")
        stored = loaded

    if stored is None:
        parent_compact = _stage2_packet_is_compact(parent_rendered)
        candidate_compact = _stage2_packet_is_compact(candidate_rendered)
        # Both renders derive from the same frozen state and record IDs.  The
        # compact arm, if either requires it, is therefore a complete common
        # index rather than a strategy-specific retrieval outcome.
        shared_source = (
            candidate_rendered if candidate_compact else parent_rendered
            if parent_compact else parent_rendered
        )
        shared_context = {
            key: deepcopy(value)
            for key, value in shared_source.items()
            if key not in {"prompt", "prompt_adaptation", CONTEXT_AUDIT_KEY}
        }
        source_audit = shared_source.get(CONTEXT_AUDIT_KEY, {})
        context_mode = (
            "shared_complete_compact_indexes"
            if parent_compact or candidate_compact
            else "shared_complete_inventory"
        )
        stored = {
            "identity": identity,
            "created_at": time.time(),
            "record_ids": [item.get("id") for item in shared_context.get("records", []) if isinstance(item, dict)],
            "context_mode": context_mode,
            "source_context_modes": {
                "parent": source_audit.get("context_mode") if parent_compact else parent_rendered.get(CONTEXT_AUDIT_KEY, {}).get("context_mode"),
                "candidate": source_audit.get("context_mode") if candidate_compact else candidate_rendered.get(CONTEXT_AUDIT_KEY, {}).get("context_mode"),
            },
            "shared_context": shared_context,
            "shared_context_sha256": canonical_hash(shared_context),
            "selection_rule": "Use compact complete inventories for both arms if either rendered strategy requires compact context; otherwise use complete inventories for both.",
        }
        write_json(artifact_path, stored)
        _record_prompt_adaptation_context_audit(
            source_project,
            {
                "stage": VALIDATION_ACTION,
                "status": "shared_envelope_frozen",
                "snapshot_id": snapshot.get("snapshot_id"),
                "candidate_strategy_sha256": candidate_overlay.get("sha256"),
                "parent_strategy_sha256": parent_overlay.get("sha256"),
                "context_mode": context_mode,
                "shared_context_sha256": stored["shared_context_sha256"],
                "estimated_context_tokens": packet_token_estimate(shared_context),
                "source_context_modes": stored["source_context_modes"],
                "selection_rule": stored["selection_rule"],
            },
        )
    else:
        _record_prompt_adaptation_context_audit(
            source_project,
            {
                "stage": VALIDATION_ACTION,
                "status": "shared_envelope_reused",
                "snapshot_id": snapshot.get("snapshot_id"),
                "context_mode": stored["context_mode"],
                "shared_context_sha256": stored["shared_context_sha256"],
            },
        )

    parent_packet = _packet_from_shared_validation_context(stored, parent_rendered)
    candidate_packet = _packet_from_shared_validation_context(stored, candidate_rendered)
    parent_tokens = packet_token_estimate(context_public_packet(parent_packet))
    candidate_tokens = packet_token_estimate(context_public_packet(candidate_packet))
    input_limit = DEFAULT_INPUT_TOKENS["relational_process_analysis"]
    stored["parent_input_tokens"] = parent_tokens
    stored["candidate_input_tokens"] = candidate_tokens
    stored["within_stage2_input_budget"] = max(parent_tokens, candidate_tokens) <= input_limit
    write_json(artifact_path, stored)
    if max(parent_tokens, candidate_tokens) > input_limit:
        # No arm gets an unrecorded additional reduction.  The snapshot remains
        # intact and the error points to an explicit intervention (smaller
        # future Stage-2 batches) instead of admitting an unfair comparison.
        _record_prompt_adaptation_context_audit(
            source_project,
            {
                "stage": VALIDATION_ACTION,
                "status": "shared_envelope_over_budget",
                "snapshot_id": snapshot.get("snapshot_id"),
                "shared_context_sha256": stored["shared_context_sha256"],
                "parent_input_tokens": parent_tokens,
                "candidate_input_tokens": candidate_tokens,
                "input_limit": input_limit,
                "required_action": "Split future Stage-2 batches before collecting eligible validation snapshots; this frozen snapshot will not be evaluated with strategy-specific context reduction.",
            },
        )
    return parent_packet, candidate_packet, stored


def _render_snapshot_stage2_packet(
    source_project: GroundedTheoryProject,
    snapshot: dict[str, Any],
    overlay: dict[str, Any],
    *,
    record_ids: list[str],
    force_structural_context: bool = False,
) -> dict[str, Any]:
    raw = deepcopy(snapshot["state_without_records"])
    records = [asdict(record) for record in source_project.state.records]
    if canonical_hash(records) != snapshot.get("record_set_sha256"):
        raise CompletionError("project records changed; immutable validation snapshot cannot be rendered")
    raw["records"] = records
    full_batch_ids = list(snapshot["record_ids"])
    if not record_ids or any(record_id not in full_batch_ids for record_id in record_ids):
        raise CompletionError("validation record group is not a non-empty subset of its frozen snapshot")
    if record_ids != full_batch_ids:
        already_analyzed = list(raw.get("relationally_analyzed_record_ids", []))
        raw["relationally_analyzed_record_ids"] = [
            *already_analyzed,
            *[record_id for record_id in full_batch_ids if record_id not in record_ids and record_id not in already_analyzed],
        ]
        config = raw.get("config")
        if not isinstance(config, dict):
            raise CompletionError("validation snapshot has no configurable Stage-2 batch size")
        config["relational_batch_size"] = len(record_ids)
    state = state_from_dict(raw)
    state.analysis_metadata["prompt_adaptation_replay"] = True
    state.pending_relational_adaptation_context = None
    state.pending_prompt_adaptation_failure_extraction = None
    state.prompt_adaptation_state["active_overlay"] = deepcopy(overlay)
    state.prompt_adaptation_state["pending_candidate"] = None
    replay_project = GroundedTheoryProject(source_project.root, state)
    rendered = replay_project.task_packet()
    if force_structural_context:
        replay_project._apply_stage2_structural_retrieval(rendered)
    if rendered.get("action") != "relational_process_analysis":
        raise CompletionError("validation snapshot no longer resolves to its frozen Stage-2 action")
    return rendered


def _stage2_packet_is_compact(packet: dict[str, Any]) -> bool:
    audit = packet.get(CONTEXT_AUDIT_KEY)
    return bool(packet.get("context_compaction")) or (
        isinstance(audit, dict) and audit.get("context_mode") == "complete_compact_indexes"
    )


def _packet_from_shared_validation_context(
    envelope: dict[str, Any], rendered: dict[str, Any]
) -> dict[str, Any]:
    packet = deepcopy(envelope["shared_context"])
    packet["prompt"] = rendered["prompt"]
    packet["prompt_adaptation"] = deepcopy(rendered["prompt_adaptation"])
    packet[CONTEXT_AUDIT_KEY] = audit_context(
        packet,
        stage="relational_process_analysis",
        mode=envelope["context_mode"],
        retrieval_level=0,
        counts={
            "records": len(packet.get("records", [])),
            "concepts": len(packet.get("concept_inventory", [])),
            "relationships": len(packet.get("relation_inventory", [])),
            "processes": len(packet.get("process_inventory", [])),
            "memos": len(packet.get("memos", [])),
            "negative_cases": len(packet.get("negative_cases", [])),
        },
        details={
            "shared_context_sha256": envelope["shared_context_sha256"],
            "selection_rule": envelope["selection_rule"],
            "same_non_prompt_context_for_parent_and_candidate": True,
            "strategy_version": rendered["prompt_adaptation"].get("strategy_version"),
            "target_tokens": DEFAULT_INPUT_TOKENS["relational_process_analysis"],
        },
    )
    return packet


def _record_prompt_adaptation_context_audit(
    project: GroundedTheoryProject, value: dict[str, Any]
) -> None:
    path = project.root / "prompt_adaptation" / "context_audit.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def _replay_stage2_snapshot(
    source_project: GroundedTheoryProject,
    snapshot: dict[str, Any],
    overlay: dict[str, Any],
    root: Path,
    *,
    complete: Completion,
    attempts: int,
    review_parallelism: int,
    initial_packet: dict[str, Any],
) -> dict[str, Any]:
    """Run an exact Stage-2 batch/review in a resumable non-source project."""
    summary_path = root / "summary.json"
    if summary_path.exists():
        loaded = json.loads(summary_path.read_text(encoding="utf-8"))
        if loaded.get("snapshot_id") == snapshot.get("snapshot_id"):
            return loaded
    state_path = root / "analysis_state.json"
    if state_path.exists():
        replay = GroundedTheoryProject.load(root)
    else:
        raw = deepcopy(snapshot["state_without_records"])
        records = [asdict(record) for record in source_project.state.records]
        if canonical_hash(records) != snapshot.get("record_set_sha256"):
            raise CompletionError("project records changed; immutable validation snapshot cannot be replayed")
        raw["records"] = records
        replay_state = state_from_dict(raw)
        replay_state.analysis_metadata["prompt_adaptation_replay"] = True
        replay_state.pending_relational_adaptation_context = None
        replay_state.pending_prompt_adaptation_failure_extraction = None
        replay_state.prompt_adaptation_state["active_overlay"] = deepcopy(overlay)
        replay_state.prompt_adaptation_state["pending_candidate"] = None
        replay = GroundedTheoryProject(root, replay_state)
        replay._write_state()
    target_ids = set(snapshot["record_ids"])
    outputs: list[dict[str, Any]] = []
    steps = 0
    while not target_ids.issubset(set(replay.state.relationally_analyzed_record_ids)):
        action = replay.next_action().get("action")
        if action not in {"relational_process_analysis", "validate_relational_grounding"}:
            raise CompletionError(f"snapshot replay left Stage 2 before its batch completed: {action}")
        task_path = root / f"task-{steps:03d}-{action}.json"
        stage_result = root / f"result-{steps:03d}-{action}.json"
        execute_stage(
            replay,
            complete,
            packet_path=task_path,
            result_path=stage_result,
            attempts=attempts,
            review_parallelism=review_parallelism,
            integration_parallelism=1,
            packet_override=(
                initial_packet
                if action == "relational_process_analysis"
                and not any(item["action"] == "relational_process_analysis" for item in outputs)
                else None
            ),
        )
        if stage_result.exists():
            outputs.append({"action": action, "payload": json.loads(stage_result.read_text(encoding="utf-8"))})
        steps += 1
        if steps > 64:
            raise CompletionError("snapshot replay exceeded its fixed Stage-2 step limit")
    analysis_payload = next((item["payload"] for item in outputs if item["action"] == "relational_process_analysis"), None)
    if not isinstance(analysis_payload, dict):
        raise CompletionError("snapshot replay did not retain its Stage-2 proposal artifact")
    summary = {
        "snapshot_id": snapshot["snapshot_id"],
        "input_packet_sha256": snapshot["input_packet_sha256"],
        "validation_context_sha256": initial_packet.get(CONTEXT_AUDIT_KEY, {}).get("estimate_breakdown", {}).get("shared_context_sha256"),
        # The blind evaluator receives only these proposal/review artifacts,
        # never strategy IDs, overlay text, or parent/candidate filenames.
        "analysis": analysis_payload,
        "reviews": [item["payload"] for item in outputs if item["action"] == "validate_relational_grounding"],
    }
    write_json(summary_path, summary)
    return summary


def _blind_pairwise_rounds(
    root: Path,
    *,
    snapshot_id: str,
    candidate_id: str,
    random_seed: int,
    round_count: int,
    parent: dict[str, Any],
    contender: dict[str, Any],
    complete: Completion,
    attempts: int,
) -> list[dict[str, Any]]:
    rounds: list[dict[str, Any]] = []
    for index in range(1, round_count + 1):
        candidate_label = blind_candidate_label(candidate_id, snapshot_id, index, random_seed)
        analysis_a, analysis_b = (contender, parent) if candidate_label == "A" else (parent, contender)
        packet = _pairwise_context_packet(snapshot_id, analysis_a, analysis_b)
        packet_path = root / f"pairwise-round-{index:02d}.packet.json"
        result_path = root / f"pairwise-round-{index:02d}.result.json"
        if result_path.exists():
            cached = json.loads(result_path.read_text(encoding="utf-8"))
            if cached.get("candidate_label") != candidate_label:
                raise CompletionError("cached pairwise result has a different blinded ordering")
            result = normalize_pairwise(cached.get("model_result", {}))
        else:
            write_json(packet_path, packet)
            _append_local_context_audit(root, packet, status="prepared", packet_path=packet_path)
            ensure_packet_within_budget(packet)
            result = _complete_auxiliary(packet, complete, attempts)
            write_json(result_path, {"candidate_label": candidate_label, "model_result": result})
            _append_local_context_audit(root, packet, status="accepted", packet_path=packet_path, result_path=result_path)
        mapping = {candidate_label: "candidate", ("B" if candidate_label == "A" else "A"): "parent", "TIE": "tie"}
        rounds.append({
            "round": index,
            "candidate_label": candidate_label,
            "overall": mapping[result["overall"]],
            "dimensions": {name: mapping[item["winner"]] for name, item in result["dimensions"].items()},
            "rationales": {
                "overall": result["rationale"],
                **{name: item["rationale"] for name, item in result["dimensions"].items()},
            },
        })
    return rounds


def _pairwise_context_packet(
    snapshot_id: str, analysis_a: dict[str, Any], analysis_b: dict[str, Any]
) -> dict[str, Any]:
    """Give a blind evaluator a symmetric, bounded evidence projection.

    Every proposal/review object remains present in both arms.  Only verbose
    prose and quotations are shortened to the same deterministic excerpt cap;
    the union evidence index lets the evaluator see the same support and
    counterevidence pool regardless of which arm cited it.
    """
    selected: dict[str, Any] | None = None
    selected_limit = 0
    for text_limit in (480, 240, 96):
        packet = {
            "action": PAIRWISE_ACTION,
            "validation_snapshot_id": snapshot_id,
            "analysis_A": _compact_pairwise_value(analysis_a, text_limit=text_limit),
            "analysis_B": _compact_pairwise_value(analysis_b, text_limit=text_limit),
            "evidence_bundle": _pairwise_evidence_bundle(analysis_a, analysis_b, text_limit=text_limit),
            "prompt": load_prompt("gt-02f-blind-pairwise-evaluation.md"),
            "expected_output": pairwise_schema(),
        }
        selected = packet
        selected_limit = text_limit
        if packet_token_estimate(packet) <= DEFAULT_INPUT_TOKENS[PAIRWISE_ACTION]:
            break
    assert selected is not None
    counts = {
        "analysis_A_relationships": len(selected["analysis_A"].get("analysis", {}).get("relationship_updates", [])),
        "analysis_B_relationships": len(selected["analysis_B"].get("analysis", {}).get("relationship_updates", [])),
        "analysis_A_reviews": len(selected["analysis_A"].get("reviews", [])),
        "analysis_B_reviews": len(selected["analysis_B"].get("reviews", [])),
        "union_evidence_records": len(selected["evidence_bundle"]["record_ids"]),
    }
    selected[CONTEXT_AUDIT_KEY] = audit_context(
        selected,
        stage=PAIRWISE_ACTION,
        mode="symmetric_complete_proposal_review_bundle",
        retrieval_level=0,
        counts=counts,
        details={
            "evidence_excerpt_chars": selected_limit,
            "same_evidence_bundle_for_A_and_B": True,
            "all_proposal_and_review_objects_retained": True,
            "target_tokens": DEFAULT_INPUT_TOKENS[PAIRWISE_ACTION],
        },
    )
    return selected


def _compact_pairwise_value(value: Any, *, text_limit: int) -> Any:
    if isinstance(value, str):
        return compact_text(value, text_limit)
    if isinstance(value, list):
        return [_compact_pairwise_value(item, text_limit=text_limit) for item in value]
    if isinstance(value, dict):
        return {key: _compact_pairwise_value(item, text_limit=text_limit) for key, item in value.items()}
    return deepcopy(value)


def _pairwise_evidence_bundle(
    analysis_a: dict[str, Any], analysis_b: dict[str, Any], *, text_limit: int
) -> dict[str, Any]:
    evidence: dict[str, list[str]] = {}

    def visit(value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                visit(item)
            return
        if not isinstance(value, dict):
            return
        record_id = value.get("record_id")
        quote = value.get("quote")
        if isinstance(record_id, str):
            excerpt = compact_text(quote, text_limit) if isinstance(quote, str) else ""
            evidence.setdefault(record_id, [])
            if excerpt and excerpt not in evidence[record_id]:
                evidence[record_id].append(excerpt)
        for item in value.values():
            visit(item)

    visit(analysis_a)
    visit(analysis_b)
    return {
        "record_ids": sorted(evidence),
        "evidence_excerpts": [
            {"record_id": record_id, "quotes": evidence[record_id]}
            for record_id in sorted(evidence)
        ],
        "rule": "This is the unlabeled union of evidence excerpts cited by either analysis or its normal independent reviews; do not treat citation volume as quality.",
    }


def _append_local_context_audit(
    root: Path,
    packet: dict[str, Any],
    *,
    status: str,
    packet_path: Path,
    result_path: Path | None = None,
) -> None:
    audit = packet.get(CONTEXT_AUDIT_KEY)
    if not isinstance(audit, dict):
        return
    value = {
        **audit,
        "status": status,
        "packet_file": packet_path.name,
        "result_file": result_path.name if result_path is not None else None,
    }
    path = root / "context_audit.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def _complete_auxiliary(packet: dict[str, Any], complete: Completion, attempts: int) -> dict[str, Any]:
    feedback: str | None = None
    failures: list[str] = []
    for attempt in range(1, attempts + 1):
        try:
            result = complete(packet, feedback)
            result, _ = _discard_stage_output_extra_fields(packet, result)
            validate_stage_json_schema(packet, result)
            return normalize_pairwise(result)
        except (CompletionError, ValueError) as error:
            feedback = str(error)
            failures.append(f"attempt {attempt}: {feedback}")
            if attempt < attempts:
                time.sleep(min(attempt, 3))
    raise CompletionError("; ".join(failures))


def _accepted_update_counts(candidate: dict[str, Any]) -> dict[str, int]:
    """Count substantive model outputs for throughput analysis receipts."""
    keys = (
        "concept_updates",
        "relationship_updates",
        "process_updates",
        "memo_updates",
        "resolutions",
        "theoretical_sampling_needs",
    )
    return {
        key: len(value)
        for key in keys
        if isinstance((value := candidate.get(key)), list)
    }


def _record_context_audit(
    project: GroundedTheoryProject,
    packet: dict[str, Any],
    *,
    status: str,
    packet_path: Path,
    result_path: Path | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Append a task-level context receipt without exposing it to the model."""
    audit = packet.get(CONTEXT_AUDIT_KEY)
    if not isinstance(audit, dict):
        return
    value = {
        **audit,
        # A novelty-expansion retry can alter the visible packet after its
        # first audit was assembled, so calculate visibility fields at write
        # time rather than reporting a stale packet identity.
        **context_receipt_fields(packet),
        "task_id": f"{packet.get('action', 'unknown')}:{packet_path.stem}",
        "status": status,
        "packet_file": packet_path.name,
        "result_file": result_path.name if result_path is not None else None,
        "model_configuration": audit.get("model_configuration", "Prime stateless"),
    }
    if extra:
        value.update(extra)
    destination = project.root / "context_audit.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def _open_coding_novelty_feedback(
    project: GroundedTheoryProject,
    packet: dict[str, Any],
    candidate: dict[str, Any],
    complete: Completion,
    *,
    attempts: int,
    packet_path: Path,
    result_path: Path,
) -> tuple[str | None, list[str]]:
    """Run an isolated duplicate-resolution check before any Stage-1 mutation."""
    if packet.get("action") != "open_coding":
        return None, []
    candidates = novelty_candidates(project.state, candidate)
    if not candidates:
        return None, []
    review_packet = {
        "action": "resolve_concept_novelty",
        "candidate_concepts": candidates,
        "global_concept_index": compact_concept_index(project.state.concepts),
        "prompt": (
            "Independently resolve potential duplicate concepts. Similarity is a trigger only. "
            "For every supplied candidate return candidate_index, decision, existing_concept_ids, "
            "and rationale. Decisions: genuinely_new, existing_equivalent, property_or_dimension, "
            "or refine_existing. Do not merge, edit, or create state."
        ),
        "expected_output": OUTPUT_JSON_SCHEMAS["resolve_concept_novelty"],
        CONTEXT_AUDIT_KEY: {
            "stage": "resolve_concept_novelty",
            "context_mode": "full_global_concept_index",
            "retrieval_or_compression_level": 0,
            "estimated_input_tokens": 0,
            "estimate_breakdown": {},
            "counts": {"candidate_concepts": len(candidates), "global_index_concepts": len(project.state.concepts)},
            "output_token_budget": DEFAULT_OUTPUT_TOKENS["resolve_concept_novelty"],
        },
    }
    review_packet[CONTEXT_AUDIT_KEY]["estimated_input_tokens"] = packet_token_estimate(context_public_packet(review_packet))
    ensure_packet_within_budget(review_packet)
    review_packet_path = packet_path.with_name(f"{packet_path.stem}.novelty-review.json")
    review_result_path = result_path.with_name(f"{result_path.stem}.novelty-review.json")
    write_json(review_packet_path, review_packet)
    _record_context_audit(project, review_packet, status="prepared", packet_path=review_packet_path)
    feedback: str | None = None
    failures: list[str] = []
    result: dict[str, Any] | None = None
    for attempt in range(1, attempts + 1):
        try:
            result = complete(review_packet, feedback)
            result, _ = _discard_stage_output_extra_fields(review_packet, result)
            validate_stage_json_schema(review_packet, result)
            break
        except (CompletionError, ValueError) as error:
            feedback = str(error)
            failures.append(f"attempt {attempt}: {feedback}")
    if result is None:
        raise CompletionError("novelty review could not complete: " + "; ".join(failures))
    write_json(review_result_path, result)
    _record_context_audit(project, review_packet, status="completed", packet_path=review_packet_path, result_path=review_result_path)
    resolutions = result.get("resolutions")
    if not isinstance(resolutions, list):
        raise ValueError("novelty review resolutions must be an array")
    expected_indexes = {item["candidate_index"] for item in candidates}
    by_index: dict[int, dict[str, Any]] = {}
    allowed = {"genuinely_new", "existing_equivalent", "property_or_dimension", "refine_existing"}
    for resolution in resolutions:
        if not isinstance(resolution, dict) or not isinstance(resolution.get("candidate_index"), int):
            raise ValueError("every novelty resolution must name an integer candidate_index")
        index = resolution["candidate_index"]
        decision = resolution.get("decision")
        if index not in expected_indexes or decision not in allowed:
            raise ValueError("novelty review returned an invalid candidate index or decision")
        by_index[index] = resolution
    if set(by_index) != expected_indexes:
        raise ValueError("novelty review must resolve every triggered candidate")
    blocked = [item for item in by_index.values() if item["decision"] != "genuinely_new"]
    if not blocked:
        return None, []
    expansion_ids: list[str] = []
    known_ids = {concept.id for concept in project.state.concepts}
    for item in blocked:
        identifiers = item.get("existing_concept_ids", [])
        if isinstance(identifiers, list):
            expansion_ids.extend(
                identifier for identifier in identifiers
                if isinstance(identifier, str) and identifier in known_ids
            )
    details = [
        f"candidate {item['candidate_index']} was independently classified {item['decision']} against {item.get('existing_concept_ids', [])}: {item.get('rationale', '')}"
        for item in blocked
    ]
    return (
        "Potential concept duplication must be resolved before any state mutation. "
        "Regenerate the full open-coding result using SAME, VARIATION, CONTRADICTION, or an "
        "evidence-grounded distinct NEW concept as appropriate. Do not treat a property/dimension "
        "as a new concept merely to bypass this check. " + " | ".join(details),
        sorted(set(expansion_ids)),
    )


def _expand_open_packet_context(
    packet: dict[str, Any], project: GroundedTheoryProject, concept_ids: list[str]
) -> None:
    """Deterministically add named index matches before the repair retry.

    This is the retrieval expansion fallback: a candidate cannot be rejected as
    a likely duplicate and then repaired against an opaque label-only index.
    The underlying state is unchanged; only this uncommitted retry packet gains
    detailed definitions, evidence, revisions, variations, and boundaries.
    """
    if not concept_ids or packet.get("context_mode") not in {
        "indexed_retrieval", "identifier_retrieval",
    }:
        return
    concepts = {concept.id: concept for concept in project.state.concepts}
    existing = {
        item.get("id"): item
        for item in packet.get("concept_inventory", [])
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    for concept_id in concept_ids:
        concept = concepts.get(concept_id)
        if concept is not None:
            existing[concept_id] = detailed_concept_view(concept)
    packet["concept_inventory"] = [existing[key] for key in sorted(existing)]
    retrieval = packet.get("retrieval")
    if isinstance(retrieval, dict):
        prior = retrieval.get("expanded_concept_ids", [])
        prior_ids = prior if isinstance(prior, list) else []
        retrieval["expanded_concept_ids"] = sorted(
            {item for item in [*prior_ids, *concept_ids] if isinstance(item, str)}
        )
    audit = packet.get(CONTEXT_AUDIT_KEY)
    if isinstance(audit, dict):
        counts = audit.setdefault("counts", {})
        if isinstance(counts, dict):
            counts["detailed_concepts"] = len(packet["concept_inventory"])
            total = counts.get("concepts")
            if isinstance(total, int):
                counts["compact_only_concepts"] = total - len(packet["concept_inventory"])
        details = audit.setdefault("estimate_breakdown", {})
        if isinstance(details, dict):
            details["expansion_concept_ids"] = list(concept_ids)
        audit["estimated_input_tokens"] = packet_token_estimate(context_public_packet(packet))


def _pending_relational_review_packets(
    project: GroundedTheoryProject, template: dict[str, Any]
) -> list[dict[str, Any]]:
    """Build immutable packets for every remaining claim in one staged candidate.

    These packets share the same uncommitted relational candidate.  A review
    result can therefore be completed independently, but it is still submitted
    in stable claim order by the caller so that the durable state machine keeps
    its original resume semantics.
    """
    candidate = project.state.pending_relational_payload
    if candidate is None:
        raise ValueError("there is no pending relational submission to review")
    material = build_relational_review_material(project.state, candidate)
    completed_ids = {
        item.get("claim_id") for item in project.state.pending_relational_review_results
    }
    if any(not isinstance(claim_id, str) for claim_id in completed_ids):
        raise ValueError("stored relational review results must name their claim_id")
    if len(completed_ids) != len(project.state.pending_relational_review_results):
        raise ValueError("stored relational review results contain duplicate claims")
    all_records = [asdict(record) for record in project.state.records]
    records_by_id = {record["id"]: record for record in all_records}
    record_order = {record["id"]: index for index, record in enumerate(all_records)}
    packets: list[dict[str, Any]] = []
    for claim in material["review_claims"]:
        claim_id = claim.get("claim_id")
        if not isinstance(claim_id, str):
            raise ValueError("relational review claim is missing claim_id")
        if claim_id in completed_ids:
            continue
        record_ids = {
            record_id
            for cohort in claim["evidence_sets"].values()
            for record_id in cohort
        }
        review_records = [
            records_by_id[record_id]
            for record_id in record_ids
            if record_id in records_by_id
        ]
        review_records.sort(key=lambda record: record_order[record["id"]])
        packet = deepcopy(template)
        packet.update(
            {
                "review_claim": claim,
                "review_records": review_records,
                "retrieval_rule": material["retrieval_rule"],
                "expected_output": (
                    process_edge_grounding_review_output_schema()
                    if claim.get("claim_type") == "process_edge"
                    else relational_grounding_review_output_schema()
                ),
            }
        )
        packets.append(packet)
    return packets


def _execute_parallel_relational_claim_reviews(
    project: GroundedTheoryProject,
    review_packets: list[dict[str, Any]],
    complete: Completion,
    *,
    packet_path: Path,
    result_path: Path,
    attempts: int,
    parallelism: int,
) -> dict[str, Any]:
    """Fan out independent claim reviews, then commit the valid group once.

    A fan-out group is all-or-none: result artifacts can survive a failure for
    inspection, but no mutable project state is advanced until every ordered
    claim has been successfully validated.
    """
    worker_count = min(parallelism, len(review_packets))
    jobs = [
        (
            index,
            packet,
            _claim_path(packet_path, index),
            _claim_path(result_path, index),
        )
        for index, packet in enumerate(review_packets, start=1)
    ]
    completed: dict[int, tuple[dict[str, Any], list[str]]] = {}
    errors: dict[int, str] = {}
    all_failures: list[str] = []

    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="prime-review") as executor:
        futures = {
            executor.submit(
                _complete_relational_review_packet,
                packet,
                complete,
                attempts=attempts,
                packet_path=claim_packet_path,
                result_path=claim_result_path,
                parallelism=1,
            ): index
            for index, packet, claim_packet_path, claim_result_path in jobs
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                completed[index] = future.result()
            except (CompletionError, PacketTooLargeError, ValueError) as error:
                errors[index] = str(error)

    ordered_candidates: list[dict[str, Any]] = []
    if not errors:
        for index in range(1, len(jobs) + 1):
            candidate, failures = completed[index]
            all_failures.extend(f"claim {index}: {failure}" for failure in failures)
            ordered_candidates.append(candidate)

    manifest = {
        "action": "validate_relational_grounding",
        "parallelism": worker_count,
        "claim_count": len(review_packets),
        "submitted_claim_ids": [item.get("claim_id") for item in ordered_candidates],
        "claim_packet_files": [path.name for _, _, path, _ in jobs],
        "claim_result_files": [path.name for _, _, _, path in jobs],
        "failed_claims": [
            {"claim_index": index, "reason": reason}
            for index, reason in sorted(errors.items())
        ],
    }
    write_json(result_path, manifest)
    if errors:
        details = "; ".join(
            f"claim {index}: {reason}" for index, reason in sorted(errors.items())
        )
        raise CompletionError(f"parallel relational review group remains uncommitted: {details}")
    submit_group = getattr(project, "submit_relational_review_group_atomically", None)
    if callable(submit_group):
        accepted = submit_group(ordered_candidates)
    else:  # Narrow test double compatibility; production projects use the atomic method.
        accepted = None
        for candidate in ordered_candidates:
            accepted = project.submit_agent_result(candidate)
    return {
        "attempt": 1,
        "accepted": accepted if accepted is not None else project.status(),
        "action": "validate_relational_grounding",
        "parallel_claim_review_count": len(review_packets),
        "parallelism": worker_count,
        "failures_before_success": all_failures,
    }


def _claim_path(path: Path, index: int) -> Path:
    return path.with_name(f"{path.stem}.claim-{index}{path.suffix}")


def _complete_relational_review_packet(
    packet: dict[str, Any],
    complete: Completion,
    *,
    attempts: int,
    packet_path: Path,
    result_path: Path,
    parallelism: int,
) -> tuple[dict[str, Any], list[str]]:
    """Run one independent claim review without mutating project state."""
    write_json(packet_path, packet)
    try:
        ensure_packet_within_budget(packet)
    except PacketTooLargeError:
        return _complete_batched_relational_review(
            packet,
            complete,
            packet_path=packet_path,
            result_path=result_path,
            attempts=attempts,
            parallelism=parallelism,
        )
    candidate, failures = _complete_review_observation(
        packet, complete, attempts, result_path
    )
    write_json(result_path, candidate)
    return candidate, failures


def _attempt_path(destination: Path, attempt: int) -> Path:
    return destination.with_name(
        f"{destination.stem}.candidate-attempt{attempt}{destination.suffix}"
    )


def _execute_batched_relational_review(
    project: GroundedTheoryProject,
    packet: dict[str, Any],
    complete: Completion,
    *,
    packet_path: Path,
    result_path: Path,
    attempts: int,
    parallelism: int,
) -> dict[str, Any]:
    """Review every retrieved record in bounded transactions, then synthesize once.

    This path is used only when a normal relation-review packet cannot fit in
    the protected input budget.  Each original record appears in exactly one
    batch packet.  The final reviewer receives the complete set of batch
    observations and is the only call whose decision can change project state.
    """
    batch_count = len(relational_review_batches(packet))
    candidate, all_failures = _complete_batched_relational_review(
        packet,
        complete,
        packet_path=packet_path,
        result_path=result_path,
        attempts=attempts,
        parallelism=parallelism,
    )
    accepted = project.submit_agent_result(candidate)
    write_json(result_path, candidate)
    return {
        "attempt": 1,
        "accepted": accepted,
        "action": packet["action"],
        "batched_review_count": batch_count,
        "failures_before_success": all_failures,
    }


def _complete_batched_relational_review(
    packet: dict[str, Any],
    complete: Completion,
    *,
    packet_path: Path,
    result_path: Path,
    attempts: int,
    parallelism: int,
) -> tuple[dict[str, Any], list[str]]:
    """Complete a large claim review in independent bounded observations."""
    batches = relational_review_batches(packet)
    jobs = [
        (
            index,
            batch,
            _batch_path(packet_path, index),
            _batch_path(result_path, index),
        )
        for index, batch in enumerate(batches, start=1)
    ]
    outcomes = _complete_review_jobs(jobs, complete, attempts, parallelism)
    observations: list[dict[str, Any]] = []
    all_failures: list[str] = []
    for index, candidate, failures in outcomes:
        batch = batches[index - 1]
        all_failures.extend(f"batch {index}: {failure}" for failure in failures)
        observations.append(
            {
                "batch_index": index,
                "record_ids": [record["id"] for record in batch["review_records"]],
                "review": candidate,
            }
        )
    candidate, failures = _synthesize_relational_review_observations(
        packet,
        observations,
        complete,
        attempts=attempts,
        packet_path=packet_path,
        result_path=result_path,
        parallelism=parallelism,
    )
    all_failures.extend(f"synthesis: {failure}" for failure in failures)
    write_json(result_path, candidate)
    return candidate, all_failures


def _complete_review_jobs(
    jobs: list[tuple[int, dict[str, Any], Path, Path]],
    complete: Completion,
    attempts: int,
    parallelism: int,
) -> list[tuple[int, dict[str, Any], list[str]]]:
    """Execute independent review packets with bounded worker concurrency."""
    if parallelism < 1:
        raise ValueError("parallelism must be positive")
    for _, packet, packet_path, _ in jobs:
        write_json(packet_path, packet)
    if len(jobs) == 1 or parallelism == 1:
        return [
            (
                index,
                *_complete_review_observation(packet, complete, attempts, result_path),
            )
            for index, packet, _, result_path in jobs
        ]
    outcomes: dict[int, tuple[dict[str, Any], list[str]]] = {}
    failures: dict[int, str] = {}
    with ThreadPoolExecutor(
        max_workers=min(parallelism, len(jobs)), thread_name_prefix="prime-review"
    ) as executor:
        futures = {
            executor.submit(_complete_review_observation, packet, complete, attempts, result_path): index
            for index, packet, _, result_path in jobs
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                outcomes[index] = future.result()
            except (CompletionError, PacketTooLargeError, ValueError) as error:
                failures[index] = str(error)
    if failures:
        details = "; ".join(
            f"batch {index}: {reason}" for index, reason in sorted(failures.items())
        )
        raise CompletionError(f"parallel review observations failed: {details}")
    return [
        (index, *outcomes[index]) for index, _, _, _ in jobs
    ]


def relational_review_batches(packet: dict[str, Any]) -> list[dict[str, Any]]:
    """Partition an oversized relation review without omitting any record."""
    if packet.get("action") != "validate_relational_grounding":
        raise ValueError("only relational grounding reviews can be batched")
    records = packet.get("review_records")
    if not isinstance(records, list) or not records:
        raise ValueError("an oversized relational review must contain review_records")

    template = deepcopy(packet)
    template["review_records"] = []
    template["review_batch"] = {"batch_index": 0, "batch_count": 0}
    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("id"), str):
            raise ValueError("review_records must contain objects with string IDs")
        proposed = [*current, record]
        candidate = deepcopy(template)
        candidate["review_records"] = proposed
        if packet_token_estimate(candidate) <= RELATIONAL_REVIEW_INPUT_ACCEPTANCE_TOKENS:
            current = proposed
            continue
        if not current:
            raise PacketTooLargeError(
                "one relational-review record exceeds the protected input budget"
            )
        chunks.append(current)
        current = [record]
    if current:
        chunks.append(current)

    # The provisional pack above uses placeholder batch metadata.  A tokenizer
    # can assign a different token count once the final batch index/count is
    # written (for example, a 27,000-token provisional batch became 27,001
    # tokens when its real metadata was attached).  Recheck the final packets
    # and split only an overflowing multi-record chunk; never discard review
    # material.  The 27,000-token target permits the documented 50-token
    # tokenizer-accounting margin, while the wider model context guard stays
    # unchanged.
    while True:
        total = len(chunks)
        batches: list[dict[str, Any]] = []
        overflowing_index: int | None = None
        for index, chunk in enumerate(chunks, start=1):
            batch = deepcopy(template)
            batch["review_records"] = chunk
            batch["review_batch"] = {"batch_index": index, "batch_count": total}
            if packet_token_estimate(batch) > RELATIONAL_REVIEW_INPUT_ACCEPTANCE_TOKENS:
                overflowing_index = index - 1
                break
            batches.append(batch)
        if overflowing_index is None:
            return batches

        overflowing_chunk = chunks[overflowing_index]
        if len(overflowing_chunk) == 1:
            raise PacketTooLargeError(
                "one relational-review record exceeds the protected input budget"
            )
        chunks[overflowing_index:overflowing_index + 1] = [
            overflowing_chunk[:-1],
            overflowing_chunk[-1:],
        ]


def relational_review_synthesis_packet(
    packet: dict[str, Any], observations: list[dict[str, Any]], *, phase: str = "final"
) -> dict[str, Any]:
    """Create the compact final decision packet after all batches were read."""
    if not observations:
        raise ValueError("a relational-review synthesis needs at least one batch observation")
    synthesis = deepcopy(packet)
    synthesis["review_records"] = []
    synthesis.pop("review_batch", None)
    synthesis["batch_reviews"] = observations
    synthesis["review_synthesis"] = {
        "batch_count": len(observations),
        "phase": phase,
        "rule": (
            "Every deterministically retrieved review record was read in exactly one batch. "
            "Use the batch reviews, retrieval audit, and complete cohort counts to make the one final decision."
        ),
    }
    claim = synthesis.get("review_claim")
    if isinstance(claim, dict) and claim.get("claim_type") == "process_edge":
        reviewed_ids: set[str] = set()
        for observation in observations:
            review = observation.get("review")
            if not isinstance(review, dict):
                continue
            for field in ("supporting_evidence", "negative_evidence"):
                for evidence in review.get(field, []):
                    if isinstance(evidence, dict) and isinstance(evidence.get("record_id"), str):
                        reviewed_ids.add(evidence["record_id"])
        synthesis["review_synthesis"]["reviewed_evidence_record_ids"] = sorted(reviewed_ids)
    return synthesis


def relational_review_synthesis_batches(
    packet: dict[str, Any], observations: list[dict[str, Any]]
) -> list[list[dict[str, Any]]]:
    """Partition batch observations when their final synthesis is also oversized."""
    if not observations:
        raise ValueError("a relational-review synthesis needs at least one batch observation")
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for observation in observations:
        proposed = [*current, observation]
        candidate = relational_review_synthesis_packet(
            packet, proposed, phase="intermediate"
        )
        if packet_token_estimate(candidate) <= RELATIONAL_REVIEW_INPUT_ACCEPTANCE_TOKENS:
            current = proposed
            continue
        if not current:
            raise PacketTooLargeError(
                "one relational-review observation exceeds the protected input budget"
            )
        groups.append(current)
        current = [observation]
    if current:
        groups.append(current)
    return groups


def _synthesize_relational_review_observations(
    packet: dict[str, Any],
    observations: list[dict[str, Any]],
    complete: Completion,
    *,
    attempts: int,
    packet_path: Path,
    result_path: Path,
    parallelism: int,
) -> tuple[dict[str, Any], list[str]]:
    """Reduce arbitrarily many bounded review observations to one decision."""
    level = 1
    failures: list[str] = []
    current = observations
    while True:
        synthesis = relational_review_synthesis_packet(packet, current)
        if packet_token_estimate(synthesis) <= RELATIONAL_REVIEW_INPUT_ACCEPTANCE_TOKENS:
            synthesis_packet_path = packet_path.with_name(
                f"{packet_path.stem}.synthesis{packet_path.suffix}"
            )
            write_json(synthesis_packet_path, synthesis)
            candidate, attempt_failures = _complete_review_observation(
                synthesis, complete, attempts, result_path
            )
            failures.extend(f"synthesis: {failure}" for failure in attempt_failures)
            return candidate, failures

        groups = relational_review_synthesis_batches(packet, current)
        if len(groups) >= len(current):
            raise PacketTooLargeError(
                "relational-review synthesis could not be reduced within the protected input budget"
            )
        jobs: list[tuple[int, dict[str, Any], Path, Path]] = []
        for index, group in enumerate(groups, start=1):
            intermediate = relational_review_synthesis_packet(
                packet, group, phase="intermediate"
            )
            intermediate_packet_path = packet_path.with_name(
                f"{packet_path.stem}.synthesis-level-{level}-batch-{index}{packet_path.suffix}"
            )
            intermediate_result_path = result_path.with_name(
                f"{result_path.stem}.synthesis-level-{level}-batch-{index}{result_path.suffix}"
            )
            jobs.append(
                (index, intermediate, intermediate_packet_path, intermediate_result_path)
            )
        summaries: list[dict[str, Any]] = []
        for index, candidate, attempt_failures in _complete_review_jobs(
            jobs, complete, attempts, parallelism
        ):
            failures.extend(
                f"synthesis level {level}, batch {index}: {failure}"
                for failure in attempt_failures
            )
            summaries.append(
                {
                    "batch_index": index,
                    "source_batch_indexes": [
                        item["batch_index"] for item in groups[index - 1]
                    ],
                    "review": candidate,
                }
            )
        current = summaries
        level += 1


def _complete_review_observation(
    packet: dict[str, Any],
    complete: Completion,
    attempts: int,
    destination: Path,
) -> tuple[dict[str, Any], list[str]]:
    """Retry a review result before it is allowed into a synthesis packet."""
    feedback: str | None = None
    failures: list[str] = []
    for attempt in range(1, attempts + 1):
        try:
            candidate = complete(packet, feedback)
            raw_candidate = deepcopy(candidate)
            candidate, ignored_fields = _discard_stage_output_extra_fields(packet, candidate)
            normalization: dict[str, Any] = {}
            if ignored_fields:
                normalization["ignored_top_level_fields"] = ignored_fields
            rationale_compaction = _compact_review_rationale(candidate)
            if rationale_compaction is not None:
                # Preserve the model's unmodified answer for audit, then
                # apply the mechanical output-format repair locally. The
                # decision, conditions, evidence, and claim identity are not
                # changed; only surplus rationale prose is removed. Retrying
                # a deterministic model after this exact formatting failure
                # merely repeats the same invalid completion.
                normalization["rationale_compaction"] = {
                    key: value
                    for key, value in rationale_compaction.items()
                    if key != "original"
                }
            if normalization:
                # Retain exactly what the model emitted.  The candidate used
                # for state submission contains only the review contract's
                # substantive fields, so harmless presentation extras cannot
                # exhaust a bounded retry budget.
                write_json(
                    _raw_attempt_path(destination, attempt),
                    {"candidate": raw_candidate, "normalization": normalization},
                )
            validate_stage_json_schema(packet, candidate)
            claim = packet.get("review_claim")
            if not isinstance(claim, dict):
                raise ValueError("review packet is missing review_claim")
            review_synthesis = packet.get("review_synthesis")
            allowed_record_ids = (
                review_synthesis.get("reviewed_evidence_record_ids")
                if isinstance(review_synthesis, dict)
                and claim.get("claim_type") == "process_edge"
                else None
            )
            # A synthesis packet intentionally omits raw records to keep its
            # context bounded.  Its final decision may still cite any record
            # deterministically retrieved for this claim, including a
            # supporting record mentioned by an earlier batch but not copied
            # into that batch's compact observation.  The state commit path
            # subsequently verifies the quote against the full persistent
            # corpus, so this broadens scope without losing provenance.
            if isinstance(allowed_record_ids, list):
                evidence_sets = claim.get("evidence_sets")
                if isinstance(evidence_sets, dict):
                    for record_ids in evidence_sets.values():
                        if isinstance(record_ids, list):
                            allowed_record_ids.extend(
                                record_id
                                for record_id in record_ids
                                if isinstance(record_id, str)
                            )
            if not isinstance(allowed_record_ids, list):
                allowed_record_ids = [
                    record["id"]
                    for record in packet.get("review_records", [])
                    if isinstance(record, dict) and isinstance(record.get("id"), str)
                ]
            candidate = normalize_claim_review(
                candidate,
                claim,
                allowed_record_ids=set(allowed_record_ids),
                review_record_texts={
                    record["id"]: record["text"]
                    for record in packet.get("review_records", [])
                    if isinstance(record, dict)
                    and isinstance(record.get("id"), str)
                    and isinstance(record.get("text"), str)
                }
                or None,
            )
            write_json(_attempt_path(destination, attempt), candidate)
            write_json(destination, candidate)
            return candidate, failures
        except (CompletionError, ValueError) as error:
            feedback = str(error)
            failures.append(f"attempt {attempt}: {feedback}")
            if attempt < attempts:
                time.sleep(min(attempt, 3))
    raise CompletionError("; ".join(failures))


def _compact_review_rationale(candidate: dict[str, Any]) -> dict[str, Any] | None:
    """Repair only a review rationale that exceeds its mechanical word cap.

    Grounding decisions and evidence are validated separately. The rationale
    is a compact audit explanation, so retaining its first complete sentences
    is safer and much faster than making identical stateless model requests
    until one happens to use fewer words.
    """
    rationale = candidate.get("rationale")
    if not isinstance(rationale, str):
        return None
    words = re.findall(r"\S+", rationale)
    if len(words) <= MAX_RELATIONAL_REVIEW_RATIONALE_WORDS:
        return None
    retained: list[str] = []
    retained_count = 0
    for sentence in re.split(r"(?<=[.!?])\s+", rationale.strip()):
        sentence_words = re.findall(r"\S+", sentence)
        if not sentence_words or retained_count + len(sentence_words) > MAX_RELATIONAL_REVIEW_RATIONALE_WORDS:
            break
        retained.append(sentence)
        retained_count += len(sentence_words)
    compact = " ".join(retained)
    if not compact:
        compact = " ".join(words[:MAX_RELATIONAL_REVIEW_RATIONALE_WORDS])
        retained_count = MAX_RELATIONAL_REVIEW_RATIONALE_WORDS
    candidate["rationale"] = compact
    return {
        "original": rationale,
        "original_word_count": len(words),
        "retained_word_count": retained_count,
        "strategy": "leading_complete_sentences_with_word_boundary_fallback",
    }


def _batch_path(path: Path, index: int) -> Path:
    return path.with_name(f"{path.stem}.batch-{index}{path.suffix}")


def _raw_attempt_path(destination: Path, attempt: int) -> Path:
    """Stable audit artifact for a locally normalized review completion."""
    return destination.with_name(
        f"{destination.stem}.candidate-attempt{attempt}.raw{destination.suffix}"
    )
