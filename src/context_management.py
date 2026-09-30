"""Deterministic, auditable context views for Grounded Theory transactions.

The project state remains the complete analytic record.  This module only
chooses what a stateless model transaction sees; it never removes or rewrites
persisted evidence, concepts, relations, processes, memos, or negative cases.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict
from typing import Any, Iterable

from .models import Concept, GroundedTheoryState
from .token_budget import packet_token_estimate


CONTEXT_AUDIT_KEY = "_context_audit"
CONTEXT_POLICY_VERSION = "bounded-context-v2"
OPEN_CONTEXT_TARGET_TOKENS = 21_000
MIN_INDEX_DEFINITION_CHARS = 96
# The final tier deliberately keeps an ID/label-only catalogue.  A label is
# sufficient to find a possible comparison target; detailed definitions are
# then supplied for deterministically retrieved and explicitly expanded items.
INDEX_DEFINITION_CHARS: tuple[int | None, ...] = (
    480, 240, MIN_INDEX_DEFINITION_CHARS, None,
)
DETAIL_K_STEPS = (12, 8, 4, 2, 0)
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")


def public_packet(packet: dict[str, Any]) -> dict[str, Any]:
    """Remove orchestration-only fields before a packet reaches the model."""
    return {key: value for key, value in packet.items() if not key.startswith("_")}


def packet_sha256(packet: dict[str, Any]) -> str:
    """Return a stable identity for exactly the model-visible packet."""
    encoded = json.dumps(
        public_packet(packet), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def context_receipt_fields(packet: dict[str, Any]) -> dict[str, Any]:
    """Mechanically describe the bounded view selected for one request.

    The receipt records code-selected visibility; it never interprets evidence
    or decides relevance.  Private orchestration metadata is kept out of the
    model packet but pins the request to the state revision that rendered it.
    """
    snapshot = packet.get("_context_snapshot", {})
    snapshot = snapshot if isinstance(snapshot, dict) else {}

    def item_ids(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [
            item["id"] for item in value
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        ]

    raw_record_ids = item_ids(packet.get("records")) + item_ids(packet.get("review_records"))
    retrieval = packet.get("retrieval", {})
    retrieval = retrieval if isinstance(retrieval, dict) else {}
    retrieved = packet.get("retrieved_context", {})
    retrieved = retrieved if isinstance(retrieved, dict) else {}
    memo_retrieval = packet.get("memo_retrieval", {})
    memo_retrieval = memo_retrieval if isinstance(memo_retrieval, dict) else {}
    detailed_concept_ids = item_ids(packet.get("concept_inventory"))
    detailed_concept_ids += item_ids(retrieved.get("concept_details"))
    detailed_concept_ids += [
        value for value in retrieval.get("expanded_concept_ids", [])
        if isinstance(value, str)
    ]
    memo_ids = item_ids(packet.get("memos")) + item_ids(retrieved.get("memo_details"))
    memo_ids += [
        value for value in memo_retrieval.get("selected_memo_ids", [])
        if isinstance(value, str)
    ]
    return {
        "context_policy_version": CONTEXT_POLICY_VERSION,
        "state_snapshot_id": snapshot.get("id"),
        "state_revision": snapshot.get("revision"),
        "packet_sha256": packet_sha256(packet),
        "raw_record_ids": sorted(set(raw_record_ids)),
        "detailed_concept_ids": sorted(set(detailed_concept_ids)),
        "memo_ids": sorted(set(memo_ids)),
        "consolidated_memo_ids": list(retrieved.get("consolidated_memo_ids", [])),
        "retrieval_method": retrieval.get("method") or retrieved.get("mode"),
        "truncation": False,
    }


def compact_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "…"


def complete_identifier_catalogue(ids: Iterable[str]) -> dict[str, Any]:
    """Losslessly encode stable IDs without linear packet growth.

    Sequential project IDs are represented by inclusive numeric ranges; IDs
    without a numeric suffix remain explicit literals.  The representation is
    exact and stable, so it can be used as a final context tier without
    deleting an analytic object from durable state.
    """
    values = list(ids)
    numeric_groups: dict[tuple[str, int], list[int]] = {}
    literal_ids: list[str] = []
    for identifier in values:
        match = re.fullmatch(r"(.*?)(\d+)", identifier)
        if match is None:
            literal_ids.append(identifier)
            continue
        prefix, digits = match.groups()
        numeric_groups.setdefault((prefix, len(digits)), []).append(int(digits))

    ranges: list[dict[str, int | str]] = []
    for (prefix, width), numbers in sorted(numeric_groups.items()):
        ordered = sorted(numbers)
        start = ordered[0]
        previous = start
        for value in ordered[1:]:
            if value == previous + 1:
                previous = value
                continue
            ranges.append({"prefix": prefix, "width": width, "start": start, "end": previous})
            start = previous = value
        ranges.append({"prefix": prefix, "width": width, "start": start, "end": previous})

    return {
        "format": "complete_identifier_ranges_v1",
        "count": len(values),
        "numeric_ranges": ranges,
        "literal_ids": literal_ids,
        "expansion_rule": (
            "For each numeric range, format every integer from start through end, inclusive, "
            "using prefix and zero-padded width; include every literal_id."
        ),
    }


def compact_concept_index(
    concepts: Iterable[Concept], *, definition_limit: int | None = INDEX_DEFINITION_CHARS[0]
) -> list[dict[str, Any]]:
    """A complete ID/label/definition index, in stable creation/ID order."""
    index: list[dict[str, Any]] = []
    for concept in sorted(concepts, key=lambda item: item.id):
        item: dict[str, Any] = {
            "id": concept.id,
            "label": concept.label,
            "level": concept.level,
            "status": concept.status,
            "last_modified_revision": concept.last_modified_revision,
        }
        if definition_limit is not None:
            item["definition"] = compact_text(concept.definition, definition_limit)
        index.append(item)
    return index


def detailed_concept_view(concept: Concept) -> dict[str, Any]:
    """Evidence-aware view used only for deterministically retrieved concepts."""
    item = asdict(concept)
    for field_name, limit in (
        ("evidence", 3),
        ("definition_revisions", 2),
        ("variations", 2),
        ("negative_or_boundary_cases", 2),
    ):
        values = item[field_name]
        item[f"{field_name}_count"] = len(values)
        item[field_name] = _representative(values, limit)
    return item


def _representative(values: list[Any], limit: int) -> list[Any]:
    if len(values) <= limit:
        return values
    first = limit // 2
    return values[:first] + values[-(limit - first):]


def _tokens(value: str) -> set[str]:
    return {word.lower() for word in WORD_RE.findall(value)}


def _concept_terms(concept: Concept) -> set[str]:
    historical = " ".join(
        str(item.get("previous_definition", "")) + " "
        + str(item.get("revised_definition", ""))
        + " "
        + str(item.get("description", ""))
        for item in [*concept.definition_revisions, *concept.variations]
    )
    boundaries = " ".join(item.text_span for item in concept.negative_or_boundary_cases)
    return _tokens(" ".join((concept.label, concept.definition, historical, boundaries)))


def rank_relevant_concepts(
    state: GroundedTheoryState, records: list[dict[str, Any]]
) -> list[tuple[Concept, float]]:
    """Recall-first lexical retrieval with deterministic score and tie-breaks.

    It is intentionally simple and inspectable: similarity only controls which
    detailed records the model sees, never a merge or other analytic decision.
    """
    query = _tokens(" ".join(str(record.get("text", "")) for record in records))
    scored: list[tuple[Concept, float]] = []
    for concept in state.concepts:
        terms = _concept_terms(concept)
        overlap = len(query & terms)
        score = overlap / max(1, len(query | terms))
        scored.append((concept, score))
    return sorted(scored, key=lambda item: (-item[1], item[0].id))


def current_detail_k(config: Any) -> int:
    configured = int(getattr(config, "open_context_detailed_k", DETAIL_K_STEPS[0]))
    return next((value for value in DETAIL_K_STEPS if value <= configured), 0)


def open_context_packet(
    state: GroundedTheoryState,
    packet: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    target_tokens: int = OPEN_CONTEXT_TARGET_TOKENS,
) -> dict[str, Any]:
    """Attach a full or indexed-retrieval concept context to an open packet.

    ``context_mode`` is persisted in packet/audit artifacts, but no state is
    altered here.  Escalation state lives in the project configuration and is
    advanced explicitly by the runner after an oversized packet, never by
    silently deleting an item.
    """
    policy = getattr(state.config, "context_policy", "legacy")
    if policy == "legacy":
        packet[CONTEXT_AUDIT_KEY] = audit_context(
            packet,
            stage="open_coding",
            mode="legacy_bounded_packet",
            retrieval_level=0,
            counts={"concepts": len(state.concepts)},
            details={"policy": "legacy"},
        )
        return packet

    full = [asdict(concept) for concept in state.concepts]
    full_packet = dict(packet)
    full_packet["concept_inventory"] = full
    full_estimate = packet_token_estimate(public_packet(full_packet))
    configured_mode = getattr(state.config, "open_context_mode", "full")
    if configured_mode == "full":
        full_packet["context_mode"] = "full"
        full_packet[CONTEXT_AUDIT_KEY] = audit_context(
            full_packet,
            stage="open_coding",
            mode="full",
            retrieval_level=0,
            counts={"concepts": len(full), "detailed_concepts": len(full)},
            details={"full_estimate": full_estimate, "target_tokens": target_tokens},
        )
        return full_packet

    detail_k = current_detail_k(state.config)
    index_level = int(getattr(state.config, "context_index_compression_level", 0))
    definition_limit = INDEX_DEFINITION_CHARS[min(index_level, len(INDEX_DEFINITION_CHARS) - 1)]
    ranked = rank_relevant_concepts(state, records)
    recent_count = int(getattr(state.config, "open_context_recent_concepts", 4))
    recent = sorted(
        state.concepts,
        key=lambda item: (-item.last_modified_revision, item.id),
    )[:recent_count]
    selected: dict[str, Concept] = {concept.id: concept for concept, _ in ranked[:detail_k]}
    selected.update({concept.id: concept for concept in recent if concept.last_modified_revision > 0})
    detailed = [detailed_concept_view(selected[key]) for key in sorted(selected)]
    packet.update(
        {
            "context_mode": "indexed_retrieval",
            "global_concept_index": compact_concept_index(
                state.concepts, definition_limit=definition_limit
            ),
            # Keep the historical key as the detailed, evidence-aware view so
            # the established prompt remains applicable.
            "concept_inventory": detailed,
            "retrieval": {
                "method": "deterministic_lexical_recall_first",
                "detailed_concept_ids": [item["id"] for item in detailed],
                "recent_concept_ids": [item.id for item in recent if item.last_modified_revision > 0],
                "detail_k": detail_k,
                "index_compression_level": index_level,
                "expansion_rule": "If an indexed item may affect a comparison, request or state its ID in the analytic memo; the orchestrator retries with the named item in detailed context before accepting a novel concept.",
            },
        }
    )
    estimate = packet_token_estimate(public_packet(packet))
    identifier_fallback = (
        definition_limit is None
        and detail_k == 0
        and estimate > target_tokens
    )
    if identifier_fallback:
        packet["context_mode"] = "identifier_retrieval"
        packet["global_concept_index"] = complete_identifier_catalogue(
            concept.id for concept in state.concepts
        )
        packet["retrieval"]["global_index_mode"] = "complete_identifier_ranges"
        packet["retrieval"]["expansion_rule"] = (
            "Every concept ID remains exactly represented by an inclusive range or literal ID. "
            "When novelty review identifies a possible match, the orchestrator supplies its "
            "full detailed concept view before accepting a repaired answer."
        )
        estimate = packet_token_estimate(public_packet(packet))
    packet[CONTEXT_AUDIT_KEY] = audit_context(
        packet,
        stage="open_coding",
        mode="identifier_retrieval" if identifier_fallback else "indexed_retrieval",
        retrieval_level=index_level + 1 if identifier_fallback else index_level,
        counts={
            "concepts": len(state.concepts),
            "global_index_concepts": len(state.concepts),
            "detailed_concepts": len(detailed),
            "compact_only_concepts": len(state.concepts) - len(detailed),
        },
        details={
            "full_estimate": full_estimate,
            "estimate": estimate,
            "target_tokens": target_tokens,
            "detail_k": detail_k,
            "definition_limit": definition_limit,
            "identifier_fallback": identifier_fallback,
        },
    )
    return packet


def audit_context(
    packet: dict[str, Any],
    *,
    stage: str,
    mode: str,
    retrieval_level: int,
    counts: dict[str, int],
    details: dict[str, Any],
) -> dict[str, Any]:
    """The stable per-task audit record retained outside model context."""
    public = public_packet(packet)
    return {
        "stage": stage,
        "context_mode": mode,
        "retrieval_or_compression_level": retrieval_level,
        "estimated_input_tokens": packet_token_estimate(public),
        "estimate_breakdown": details,
        "counts": counts,
        "output_token_budget": None,
        **context_receipt_fields(packet),
    }


def novelty_candidates(
    state: GroundedTheoryState,
    payload: dict[str, Any],
    *,
    threshold: float | None = None,
) -> list[dict[str, Any]]:
    """Return only candidates which require an independent duplicate review."""
    threshold = (
        float(threshold)
        if threshold is not None
        else float(getattr(state.config, "novelty_review_similarity_threshold", 0.28))
    )
    candidates: list[dict[str, Any]] = []
    for index, update in enumerate(payload.get("concept_updates", [])):
        if not isinstance(update, dict) or update.get("comparison") != "NEW":
            continue
        candidate_terms = _tokens(
            str(update.get("label", "")) + " " + str(update.get("definition", ""))
        )
        matches: list[dict[str, Any]] = []
        for concept in state.concepts:
            terms = _concept_terms(concept)
            score = len(candidate_terms & terms) / max(1, len(candidate_terms | terms))
            if score >= threshold:
                matches.append({"concept_id": concept.id, "similarity": round(score, 6)})
        if matches:
            candidates.append(
                {
                    "candidate_index": index,
                    "candidate_label": update.get("label"),
                    "candidate_definition": update.get("definition"),
                    "trigger_matches": sorted(matches, key=lambda item: (-item["similarity"], item["concept_id"])),
                }
            )
    return candidates
