"""Deterministic hierarchical preparation for the one final theory commit.

Local theory memos are isolated, fixed-input observations.  They may run in
parallel because none can mutate the Grounded Theory state.  Only the normal
``theoretical_integration`` submission mutates the state, once, after all
required memo groups have completed and been audited.
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from .context_management import (
    CONTEXT_AUDIT_KEY,
    compact_concept_index,
    compact_text,
    complete_identifier_catalogue,
    context_receipt_fields,
    public_packet,
)
from .models import GroundedTheoryState
from .packets import integration_output_schema, load_prompt
from .persistence import write_json
from .token_budget import packet_token_estimate


WORKSPACE_FILE = "integration_workspace.json"
LOCAL_MEMO_ACTION = "theoretical_integration_local_memo"
REDUCE_MEMO_ACTION = "theoretical_integration_reduce_memo"
EVIDENCE_REQUEST_ACTION = "theoretical_integration_evidence_request"
LOCAL_MEMO_TARGET_TOKENS = 15_000
REDUCE_MEMO_OUTPUT_TARGET_TOKENS = 5_000
REDUCE_MEMO_OUTPUT_TOKEN_BUDGET = 5_000
REDUCE_MEMO_GROUP_SIZE = 3
FINAL_TARGET_TOKENS = 21_000
MAX_EVIDENCE_REQUESTS = 12
EVIDENCE_ON_DEMAND_TEXT_CHARS = 600
EVIDENCE_ON_DEMAND_LIST_ITEMS = 4

Completion = Callable[[dict[str, Any], str | None], dict[str, Any]]


LOCAL_MEMO_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "neighborhood_id", "central_phenomenon", "mechanism", "conditions",
        "actions_interactions", "consequences", "moderators", "negative_case_ids",
        "alternative_pathways", "tentative_alternatives", "propositions", "source_ids",
        "handoff",
    ],
    "properties": {
        "neighborhood_id": {"type": "string"},
        "central_phenomenon": {"type": "string"},
        "mechanism": {"type": "string"},
        "conditions": {"type": "array", "items": {"type": "string"}},
        "actions_interactions": {"type": "array", "items": {"type": "string"}},
        "consequences": {"type": "array", "items": {"type": "string"}},
        "moderators": {"type": "array", "items": {"type": "string"}},
        "negative_case_ids": {"type": "array", "items": {"type": "string"}},
        "alternative_pathways": {"type": "array", "items": {"type": "string"}},
        "tentative_alternatives": {"type": "array", "items": {"type": "string"}},
        "propositions": {"type": "array", "items": {"type": "object"}},
        "source_ids": {"type": "object"},
        "handoff": {
            "type": "object",
            "required": [
                "next_task", "decision_relevant_findings", "uncertainties",
                "evidence_to_rehydrate",
            ],
            "properties": {
                "next_task": {"type": "string"},
                "decision_relevant_findings": {"type": "array", "items": {"type": "object"}},
                "uncertainties": {"type": "array", "items": {"type": "object"}},
                "evidence_to_rehydrate": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
}
# A reducer retains only summary statements and source IDs. Detailed local
# outputs and their underlying evidence remain in the integration workspace
# and canonical project state for controlled rehydration.
REDUCED_MEMO_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["neighborhood_id", "source_ids", "handoff"],
    "properties": {
        "neighborhood_id": {"type": "string"},
        "source_ids": {
            "type": "object",
            "required": ["memos", "records"],
            "properties": {
                "memos": {"type": "array", "items": {"type": "string"}},
                "records": {"type": "array", "items": {"type": "string"}},
            },
        },
        "handoff": {
            "type": "object",
            "required": ["next_task", "decision_relevant_findings", "uncertainties", "evidence_to_rehydrate"],
            "properties": {
                "next_task": {"type": "string"},
                "decision_relevant_findings": {"type": "array", "items": {"type": "object", "required": ["statement", "source_references"]}},
                "uncertainties": {"type": "array", "items": {"type": "object", "required": ["statement", "source_references"]}},
                "evidence_to_rehydrate": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
}


EVIDENCE_REQUEST_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["evidence_requests"],
    "properties": {"evidence_requests": {"type": "array", "items": {"type": "object"}}},
}


def integration_workspace_path(project_root: Path) -> Path:
    return project_root / WORKSPACE_FILE


def execute_hierarchical_integration(
    project: Any,
    complete: Completion,
    *,
    packet_path: Path,
    result_path: Path,
    attempts: int,
    parallelism: int,
    validate: Callable[[dict[str, Any], dict[str, Any]], None],
    ensure_budget: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    """Run fixed-input local memo work then one state-changing final synthesis."""
    if parallelism < 1 or parallelism > 4:
        raise ValueError("integration_parallelism must be between 1 and 4")
    workspace = _load_workspace(project.root, project.state.revision)
    neighborhoods = workspace.get("neighborhoods")
    if not isinstance(neighborhoods, list):
        neighborhoods = build_neighborhoods(project.state)
        workspace["neighborhoods"] = neighborhoods
        workspace["hierarchy"] = {
            "algorithm": "deterministic_accepted_structure_graph",
            "source_state_revision": project.state.revision,
            "cross_neighborhood_relations": _cross_neighborhood_relations(project.state, neighborhoods),
            "evidence_retrievals": [],
            "omissions": [],
        }
        _write_workspace(project.root, workspace)

    local_packets = [
        packet
        for neighborhood in neighborhoods
        for packet in local_memo_packets(project.state, neighborhood)
    ]
    memo_results = _complete_atomic_group(
        project,
        workspace,
        group_id="local-memos",
        packets=local_packets,
        complete=complete,
        attempts=attempts,
        parallelism=parallelism,
        packet_path=packet_path,
        result_path=result_path,
        validate=validate,
        ensure_budget=ensure_budget,
    )
    reduced = _reduce_memos_if_needed(
        project,
        workspace,
        memo_results,
        complete=complete,
        attempts=attempts,
        parallelism=parallelism,
        packet_path=packet_path,
        result_path=result_path,
        validate=validate,
        ensure_budget=ensure_budget,
    )
    # Local workers may inspect a large neighborhood and their complete,
    # validated results remain in the workspace.  The parent path receives
    # only a task-aware projection plus source pointers; detailed material is
    # recovered through the existing evidence-on-demand request.
    handoffs = [_parent_handoff(memo) for memo in reduced]
    hierarchy = workspace.setdefault("hierarchy", {})
    request_packet = evidence_request_packet(project.state, handoffs, workspace)
    request_hash = _stable_hash(request_packet)
    cached_request = hierarchy.get("evidence_request")
    if (
        isinstance(cached_request, dict)
        and cached_request.get("input_hash") == request_hash
        and isinstance(cached_request.get("result"), dict)
    ):
        # A final synthesis retry must not spend another model call deciding
        # the same evidence-on-demand request. The fixed-input request and its
        # returned IDs are durable workspace artifacts, while the source
        # objects are deterministically rehydrated from current state.
        requests = cached_request["result"]
        requested_evidence = retrieve_evidence(project.state, requests)
    else:
        request_packet[CONTEXT_AUDIT_KEY]["estimated_input_tokens"] = packet_token_estimate(public_packet(request_packet))
        ensure_budget(request_packet)
        write_json(packet_path.with_name(f"{packet_path.stem}.evidence-request{packet_path.suffix}"), request_packet)
        requests = _retry_fixed_packet(request_packet, complete, attempts, validate)
        requested_evidence = retrieve_evidence(project.state, requests)
        _append_context_audit(project.root, request_packet, status="completed", task_suffix="evidence-request")
        hierarchy["evidence_request"] = {
            "input_hash": request_hash,
            "result": requests,
        }
    hierarchy["evidence_retrievals"] = requested_evidence
    _write_workspace(project.root, workspace)

    final_packet = final_integration_packet(project.state, handoffs, workspace, requested_evidence)
    final_packet[CONTEXT_AUDIT_KEY]["estimated_input_tokens"] = packet_token_estimate(public_packet(final_packet))
    ensure_budget(final_packet)
    write_json(packet_path, final_packet)
    candidate = _retry_fixed_packet(final_packet, complete, attempts, validate)
    accepted = project.submit_agent_result(candidate)
    _append_context_audit(project.root, final_packet, status="accepted", task_suffix="final-integration")
    write_json(result_path, candidate)
    final_packet[CONTEXT_AUDIT_KEY]["hierarchy"] = hierarchy
    write_json(packet_path, final_packet)
    return {
        "attempt": 1,
        "accepted": accepted,
        "action": "theoretical_integration",
        "hierarchical_local_memo_count": len(memo_results),
        "hierarchical_reduced_memo_count": len(reduced),
        "task_aware_handoff_count": len(handoffs),
        "evidence_on_demand_count": len(requested_evidence),
    }


def build_neighborhoods(state: GroundedTheoryState) -> list[dict[str, Any]]:
    """Graph neighborhoods from accepted structure; no clustering or merging."""
    concept_ids = sorted(concept.id for concept in state.concepts)
    adjacency: dict[str, set[str]] = {concept_id: set() for concept_id in concept_ids}
    for relation in state.relationships:
        adjacency.setdefault(relation.source_concept_id, set()).add(relation.target_concept_id)
        adjacency.setdefault(relation.target_concept_id, set()).add(relation.source_concept_id)
    for process in state.processes:
        for edge in process.edges:
            adjacency.setdefault(edge.source_concept_id, set()).add(edge.target_concept_id)
            adjacency.setdefault(edge.target_concept_id, set()).add(edge.source_concept_id)
    components: list[list[str]] = []
    unseen = set(concept_ids)
    while unseen:
        root = min(unseen)
        frontier = [root]
        component: list[str] = []
        unseen.remove(root)
        while frontier:
            current = frontier.pop(0)
            component.append(current)
            for neighbor in sorted(adjacency.get(current, ())):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    frontier.append(neighbor)
        components.append(sorted(component))
    # Fixed chunks prevent a giant connected component from silently becoming
    # an unbounded local task. Cross-chunk edges are recorded separately.
    neighborhoods: list[dict[str, Any]] = []
    for component_index, component in enumerate(components, start=1):
        for chunk_index, start in enumerate(range(0, len(component), 12), start=1):
            ids = component[start:start + 12]
            neighborhoods.append(
                {
                    "id": f"neighborhood-{component_index:03d}-{chunk_index:03d}",
                    "concept_ids": ids,
                    "component_index": component_index,
                }
            )
    # Memos, negative cases, and processes which are not linked to a concept
    # graph neighborhood are still analytic material. Give them explicit,
    # deterministic fixed-input neighborhoods instead of dropping them.
    unlinked_memo_ids = [
        item.id for item in state.memos
        if not (item.linked_concept_ids or item.linked_relation_ids or item.linked_process_ids)
    ]
    unlinked_process_ids = [item.id for item in state.processes if not item.edges]
    known_targets = {concept.id for concept in state.concepts}
    known_targets.update(item.id for item in state.relationships)
    known_targets.update(item.id for item in state.processes)
    unlinked_negative_ids = [
        item.id for item in state.negative_cases if item.target_id not in known_targets
    ]
    loose_items = (
        [("memo", item) for item in unlinked_memo_ids]
        + [("process", item) for item in unlinked_process_ids]
        + [("negative", item) for item in unlinked_negative_ids]
    )
    for index, start in enumerate(range(0, len(loose_items), 12), start=1):
        group = loose_items[start:start + 12]
        neighborhoods.append(
            {
                "id": f"neighborhood-unlinked-{index:03d}",
                "concept_ids": [],
                "component_index": -1,
                "unlinked_memo_ids": [value for kind, value in group if kind == "memo"],
                "unlinked_process_ids": [value for kind, value in group if kind == "process"],
                "unlinked_negative_case_ids": [value for kind, value in group if kind == "negative"],
            }
        )
    if not neighborhoods:
        neighborhoods.append({"id": "neighborhood-000-001", "concept_ids": [], "component_index": 0})
    return neighborhoods


def local_memo_packet(state: GroundedTheoryState, neighborhood: dict[str, Any]) -> dict[str, Any]:
    ids = set(neighborhood["concept_ids"])
    forced_memo_ids = set(neighborhood.get("unlinked_memo_ids", []))
    forced_process_ids = set(neighborhood.get("unlinked_process_ids", []))
    forced_negative_ids = set(neighborhood.get("unlinked_negative_case_ids", []))
    material_slice = neighborhood.get("material_slice")
    slice_active = isinstance(material_slice, dict)
    allowed_relations = set(material_slice.get("relation_ids", [])) if slice_active else set()
    allowed_processes = set(material_slice.get("process_ids", [])) if slice_active else set()
    allowed_memos = set(material_slice.get("memo_ids", [])) if slice_active else set()
    allowed_negative_cases = set(material_slice.get("negative_case_ids", [])) if slice_active else set()
    concepts = [_concept_summary(item) for item in state.concepts if item.id in ids]
    relations = [
        _relation_summary(item) for item in state.relationships
        if (item.source_concept_id in ids or item.target_concept_id in ids)
        and (not slice_active or item.id in allowed_relations)
    ]
    processes = [
        _process_summary(item) for item in state.processes
        if (item.id in forced_process_ids
            or any(edge.source_concept_id in ids or edge.target_concept_id in ids for edge in item.edges))
        and (not slice_active or item.id in allowed_processes)
    ]
    relation_ids = {item["id"] for item in relations}
    process_ids = {item["id"] for item in processes}
    memos = [
        _memo_summary(item) for item in state.memos
        if (item.id in forced_memo_ids
            or ids.intersection(item.linked_concept_ids)
            or relation_ids.intersection(item.linked_relation_ids)
            or process_ids.intersection(item.linked_process_ids))
        and (not slice_active or item.id in allowed_memos)
    ]
    negative_cases = [
        _negative_summary(item) for item in state.negative_cases
        if (item.id in forced_negative_ids
            or item.target_id in ids or item.target_id in relation_ids or item.target_id in process_ids)
        and (not slice_active or item.id in allowed_negative_cases)
    ]
    return {
        # This transaction is an isolated worker. Its output is deliberately
        # a small, source-addressable handoff for final integration, never a
        # replacement for the durable objects it inspected.
        "_context_snapshot": _state_snapshot(state),
        "action": LOCAL_MEMO_ACTION,
        "neighborhood": neighborhood,
        "concepts": concepts,
        "relations": relations,
        "processes": processes,
        "memos": memos,
        "negative_cases": negative_cases,
        "worker_handoff_contract": {
            "next_task": "final_theoretical_integration",
            "return_only": (
                "decision-relevant findings, unresolved alternatives, and exact source IDs "
                "that final integration may need to rehydrate"
            ),
            "source_of_truth": "canonical project state; do not treat this handoff as the only copy",
            "max_findings": 6,
            "max_uncertainties": 4,
            "max_evidence_to_rehydrate": 12,
        },
        "prompt": (
            "You are an isolated worker preparing a task-aware handoff for final theoretical "
            "integration, not a general summary. Retain only decision-relevant findings, "
            "unresolved alternatives, and exact source IDs that the next task may need to "
            "rehydrate. Retain all supplied status labels and negative cases. Do not merge "
            "categories, infer causality from co-occurrence, or invent evidence. Identify a "
            "central phenomenon only if grounded; otherwise state the uncertainty. Every "
            "handoff finding and uncertainty must cite supplied stable IDs."
        ),
        "expected_output": LOCAL_MEMO_SCHEMA,
        CONTEXT_AUDIT_KEY: {
            "stage": "theoretical_integration_local_memo",
            "context_mode": "graph_neighborhood",
            "retrieval_or_compression_level": 0,
            "estimated_input_tokens": 0,
            "estimate_breakdown": {},
            "counts": {
                "concepts": len(concepts), "relationships": len(relations),
                "processes": len(processes), "memos": len(memos), "negative_cases": len(negative_cases),
            },
            "output_token_budget": 4096,
            "worker_isolation": {
                "parent_task": "theoretical_integration",
                "next_task": "final_theoretical_integration",
                "returns": "task_aware_handoff_with_source_ids",
            },
        },
    }


def local_memo_packets(
    state: GroundedTheoryState, neighborhood: dict[str, Any]
) -> list[dict[str, Any]]:
    """Make complete, fixed-input local-memo packets for one neighborhood.

    A graph neighborhood can legitimately accumulate many linked memos or
    relations.  Split those material objects only when its rendered packet
    exceeds the protected local-memo target; the concept anchors remain in
    every slice and every relation/process/memo/negative-case ID appears in
    exactly one slice.  The subsequent deterministic reduce pass reunites the
    slices before final integration.
    """
    initial = local_memo_packet(state, neighborhood)
    if packet_token_estimate(public_packet(initial)) <= LOCAL_MEMO_TARGET_TOKENS:
        return [initial]

    atoms: list[tuple[str, str]] = []
    for field, items in (
        ("relation_ids", initial["relations"]),
        ("process_ids", initial["processes"]),
        ("memo_ids", initial["memos"]),
        ("negative_case_ids", initial["negative_cases"]),
    ):
        atoms.extend((field, item["id"]) for item in items)
    if not atoms:
        raise ValueError(
            "a local integration packet exceeds its target without splittable analytic material; "
            "no source was omitted"
        )

    def packet_for(slice_atoms: list[tuple[str, str]], index: int) -> dict[str, Any]:
        child = dict(neighborhood)
        child["id"] = f"{neighborhood['id']}-slice-{index:03d}"
        child["parent_neighborhood_id"] = neighborhood["id"]
        child["material_slice"] = {
            field: [value for kind, value in slice_atoms if kind == field]
            for field in ("relation_ids", "process_ids", "memo_ids", "negative_case_ids")
        }
        return local_memo_packet(state, child)

    packets: list[dict[str, Any]] = []
    current: list[tuple[str, str]] = []
    for atom in atoms:
        candidate = [*current, atom]
        proposed = packet_for(candidate, len(packets) + 1)
        if packet_token_estimate(public_packet(proposed)) <= LOCAL_MEMO_TARGET_TOKENS:
            current = candidate
            continue
        if not current:
            raise ValueError(
                "one bounded local analytic object exceeds the protected integration target; "
                "no source was omitted"
            )
        packets.append(packet_for(current, len(packets) + 1))
        current = [atom]
    if current:
        final = packet_for(current, len(packets) + 1)
        if packet_token_estimate(public_packet(final)) > LOCAL_MEMO_TARGET_TOKENS:
            raise ValueError(
                "one bounded local analytic object exceeds the protected integration target; "
                "no source was omitted"
            )
        packets.append(final)
    return packets


def evidence_request_packet(
    state: GroundedTheoryState, theory_memos: list[dict[str, Any]], workspace: dict[str, Any]
) -> dict[str, Any]:
    packet = {
        "_context_snapshot": _state_snapshot(state),
        "action": EVIDENCE_REQUEST_ACTION,
        # Local theory memos carry the analytic definitions and requested
        # evidence carries the original detail.  This global catalogue is for
        # stable-ID lookup, so labels keep it bounded even in a large study.
        "global_concept_index": compact_concept_index(state.concepts, definition_limit=None),
        "local_theory_memos": theory_memos,
        "cross_neighborhood_relations": workspace.get("hierarchy", {}).get("cross_neighborhood_relations", []),
        "prompt": (
            "Review fixed local theory memos before final integration. Return up to 12 exact "
            "source IDs whose original relation, process, memo, negative case, or record evidence "
            "must be inspected because a summary conflicts, is weak, or is uncertain. Do not draw "
            "a theory in this step."
        ),
        "expected_output": EVIDENCE_REQUEST_SCHEMA,
        CONTEXT_AUDIT_KEY: {
            "stage": EVIDENCE_REQUEST_ACTION,
            "context_mode": "hierarchical_summary_first",
            "retrieval_or_compression_level": 0,
            "estimated_input_tokens": 0,
            "estimate_breakdown": {},
            "counts": {"theory_memos": len(theory_memos)},
            "output_token_budget": 2048,
        },
    }
    return _fit_global_integration_catalogues(packet, state)


def final_integration_packet(
    state: GroundedTheoryState,
    theory_memos: list[dict[str, Any]],
    workspace: dict[str, Any],
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    hierarchy = workspace.get("hierarchy", {})
    packet = {
        "_context_snapshot": _state_snapshot(state),
        "action": "theoretical_integration",
        "study": asdict(state.config),
        "analysis_metadata": dict(state.analysis_metadata),
        "rule": "This is the sole final integration commit. Ground every proposition in supplied IDs/evidence; retain uncertainty and do not force a core category.",
        "context_mode": "hierarchical_reduce_with_evidence_on_demand",
        # Final integration receives the reduced theory memos plus explicit
        # evidence-on-demand.  Keep the global catalogue as ID/label lookup
        # rather than duplicating every definition a third time.
        "global_concept_index": compact_concept_index(state.concepts, definition_limit=None),
        "local_theory_memos": theory_memos,
        "cross_neighborhood_relations": hierarchy.get("cross_neighborhood_relations", []),
        "evidence_on_demand": evidence,
        "prompt": load_prompt("gt-03-theoretical-integration.md"),
        "expected_output": integration_output_schema(),
        CONTEXT_AUDIT_KEY: {
            "stage": "theoretical_integration",
            "context_mode": "hierarchical_reduce_with_evidence_on_demand",
            "retrieval_or_compression_level": 0,
            "estimated_input_tokens": 0,
            "estimate_breakdown": {},
            "counts": {
                "concepts": len(state.concepts), "theory_memos": len(theory_memos),
                "evidence_on_demand": len(evidence),
            },
            "output_token_budget": 8192,
            "hierarchy": hierarchy,
        },
    }
    return _fit_global_integration_catalogues(packet, state)


def _fit_global_integration_catalogues(
    packet: dict[str, Any], state: GroundedTheoryState
) -> dict[str, Any]:
    """Keep final global lookup stable when theory inventories grow large.

    Hierarchical local memos and evidence-on-demand carry the analytic detail.
    The global lists are lookup structures, so exact identifier catalogues are
    a lossless final presentation tier if labels or cross-neighborhood cards
    would otherwise consume the integration context.
    """
    if packet_token_estimate(public_packet(packet)) <= FINAL_TARGET_TOKENS:
        return packet
    packet["global_concept_index"] = complete_identifier_catalogue(
        concept.id for concept in state.concepts
    )
    cross = packet.get("cross_neighborhood_relations", [])
    if isinstance(cross, list):
        packet["cross_neighborhood_relations"] = complete_identifier_catalogue(
            str(item["id"])
            for item in cross
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        )
    if packet_token_estimate(public_packet(packet)) > FINAL_TARGET_TOKENS:
        requested = packet.get("evidence_on_demand")
        if isinstance(requested, list):
            packet["evidence_on_demand"] = [
                _compact_evidence_on_demand_item(item) for item in requested
                if isinstance(item, dict)
            ]
            packet["evidence_on_demand_compaction"] = {
                "mode": "bounded_source_summaries",
                "rule": (
                    "Each requested durable object remains identified exactly. Scalar text and "
                    "repeated fields are deterministically bounded for this final context only; "
                    "the full source object remains in project state and exported evidence."
                ),
            }
    packet["context_mode"] = "hierarchical_reduce_with_identifier_lookup"
    audit = packet.get(CONTEXT_AUDIT_KEY)
    if isinstance(audit, dict):
        audit["context_mode"] = "hierarchical_reduce_with_identifier_lookup"
        audit["retrieval_or_compression_level"] = 1
        details = audit.setdefault("estimate_breakdown", {})
        if isinstance(details, dict):
            details["global_identifier_fallback"] = True
            details["target_tokens"] = FINAL_TARGET_TOKENS
    return packet


def _compact_evidence_on_demand_item(item: dict[str, Any]) -> dict[str, Any]:
    """Bound an oversized requested source without changing durable evidence."""
    source = item.get("source")
    return {
        "id": item.get("id"),
        "kind": item.get("kind"),
        "reason": compact_text(str(item.get("reason", "")), EVIDENCE_ON_DEMAND_TEXT_CHARS),
        "source_summary": _bounded_evidence_value(source, field_name="source"),
    }


def _bounded_evidence_value(value: Any, *, field_name: str) -> Any:
    """Create a deterministic, inspectable final-context view of one source."""
    if isinstance(value, str):
        return compact_text(value, EVIDENCE_ON_DEMAND_TEXT_CHARS)
    if isinstance(value, dict):
        return {
            str(key): _bounded_evidence_value(item, field_name=str(key))
            for key, item in value.items()
        }
    if isinstance(value, list):
        if value and all(isinstance(item, str) for item in value) and field_name.endswith("_ids"):
            return complete_identifier_catalogue(value)
        selected = _bounded_items(value, EVIDENCE_ON_DEMAND_LIST_ITEMS)
        return {
            "count": len(value),
            "representative": [
                _bounded_evidence_value(item, field_name=field_name)
                for item in selected
            ],
        }
    return value


def retrieve_evidence(state: GroundedTheoryState, payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Retrieve only explicit IDs requested by the independent summary reader."""
    requests = payload.get("evidence_requests", [])
    if not isinstance(requests, list):
        raise ValueError("evidence_requests must be an array")
    if len(requests) > MAX_EVIDENCE_REQUESTS:
        raise ValueError(f"at most {MAX_EVIDENCE_REQUESTS} evidence requests are allowed")
    lookup: dict[str, tuple[str, Any]] = {}
    lookup.update({item.id: ("record", item) for item in state.records})
    lookup.update({item.id: ("relationship", item) for item in state.relationships})
    lookup.update({item.id: ("process", item) for item in state.processes})
    lookup.update({item.id: ("memo", item) for item in state.memos})
    lookup.update({item.id: ("negative_case", item) for item in state.negative_cases})
    lookup.update({item.id: ("concept", item) for item in state.concepts})
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for request in requests:
        if not isinstance(request, dict) or not isinstance(request.get("id"), str):
            raise ValueError("each evidence request must name an id")
        identifier = request["id"]
        if identifier in seen:
            continue
        if identifier not in lookup:
            raise ValueError(f"evidence request names unknown ID: {identifier}")
        kind, item = lookup[identifier]
        result.append({"id": identifier, "kind": kind, "reason": str(request.get("reason", "")), "source": asdict(item)})
        seen.add(identifier)
    return result


def _reduce_memos_if_needed(
    project: Any,
    workspace: dict[str, Any],
    memos: list[dict[str, Any]],
    *,
    complete: Completion,
    attempts: int,
    parallelism: int,
    packet_path: Path,
    result_path: Path,
    validate: Callable[[dict[str, Any], dict[str, Any]], None],
    ensure_budget: Callable[[dict[str, Any]], None],
) -> list[dict[str, Any]]:
    current = memos
    level = 1
    while packet_token_estimate(public_packet(evidence_request_packet(project.state, current, workspace))) > FINAL_TARGET_TOKENS:
        groups = [
            current[index:index + REDUCE_MEMO_GROUP_SIZE]
            for index in range(0, len(current), REDUCE_MEMO_GROUP_SIZE)
        ]
        if len(groups) >= len(current):
            raise ValueError("hierarchical integration summaries remain too large after deterministic reduction; no source was omitted")
        packets = []
        for index, group in enumerate(groups, start=1):
            packets.append({
                "_context_snapshot": _state_snapshot(project.state),
                "action": REDUCE_MEMO_ACTION,
                "neighborhood": {"id": f"reduce-{level:02d}-{index:03d}", "source_memo_ids": [item.get("neighborhood_id") for item in group]},
                "source_theory_memos": group,
                "worker_handoff_contract": {
                    "next_task": "final_theoretical_integration",
                    "return_only": "summary statements, uncertainties, and supplied canonical memo or record references only",
                    "source_of_truth": "canonical project state and the durable local-memo workspace",
                    "allowed_source_ids": _reduction_group_source_ids(project.state, group),
                    "max_summary_statements": 6,
                    "max_uncertainties": 4,
                    "max_source_references": 12,
                },
                "prompt": "You are an isolated reduction worker preparing compact input for final theoretical integration. Return only neighborhood_id, source_ids, and handoff. source_ids must use only memo and record IDs listed in worker_handoff_contract.allowed_source_ids. Each decision-relevant finding and uncertainty contains statement plus source_references. Do not copy record quotes, evidence objects, relations, concepts, processes, full local-memo fields, or underlying evidence. Preserve qualification briefly in a statement or uncertainty. Do not merge categories or invent support. Use at most 6 findings, 4 uncertainties, and 12 distinct source references; use empty arrays when absent. Produce complete valid JSON within 5000 visible tokens.",
                "expected_output": REDUCED_MEMO_SCHEMA,
                CONTEXT_AUDIT_KEY: {"stage": REDUCE_MEMO_ACTION, "context_mode": "hierarchical_reduce", "retrieval_or_compression_level": level, "estimated_input_tokens": 0, "estimate_breakdown": {}, "counts": {"source_theory_memos": len(group)}, "output_token_budget": REDUCE_MEMO_OUTPUT_TOKEN_BUDGET, "generation_target_visible_tokens": REDUCE_MEMO_OUTPUT_TARGET_TOKENS},
            })
        current = _complete_atomic_group(
            project, workspace, group_id=f"reduce-{level}", packets=packets, complete=complete,
            attempts=attempts, parallelism=parallelism, packet_path=packet_path,
            result_path=result_path, validate=validate, ensure_budget=ensure_budget,
        )
        level += 1
    return current


def _complete_atomic_group(
    project: Any,
    workspace: dict[str, Any],
    *,
    group_id: str,
    packets: list[dict[str, Any]],
    complete: Completion,
    attempts: int,
    parallelism: int,
    packet_path: Path,
    result_path: Path,
    validate: Callable[[dict[str, Any], dict[str, Any]], None],
    ensure_budget: Callable[[dict[str, Any]], None],
) -> list[dict[str, Any]]:
    groups = workspace.setdefault("groups", {})
    previous = groups.get(group_id)
    hashes = [_stable_hash(packet) for packet in packets]
    if isinstance(previous, dict) and previous.get("status") == "completed" and previous.get("input_hashes") == hashes:
        results = previous.get("results")
        if isinstance(results, list) and len(results) == len(packets):
            return results
    jobs = list(enumerate(zip(packets, hashes), start=1))
    outcomes: dict[int, dict[str, Any]] = {}
    failures: dict[int, str] = {}
    artifact_dir = result_path.parent
    with ThreadPoolExecutor(max_workers=min(parallelism, max(1, len(jobs))), thread_name_prefix="prime-integration") as executor:
        futures = {
            executor.submit(
                _complete_one_memo, packet, complete, attempts, validate, ensure_budget,
                artifact_dir / f"{result_path.stem}.{group_id}.task-{index}.json",
                artifact_dir / f"{packet_path.stem}.{group_id}.task-{index}.json",
            ): index
            for index, (packet, _) in jobs
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                outcomes[index] = future.result()
            except (ValueError, RuntimeError) as error:
                failures[index] = str(error)
    if failures:
        # No group state is committed: a later run can reuse external artifacts
        # but cannot treat a partial fan-out as analytically complete.
        _write_workspace(project.root, workspace)
        detail = "; ".join(f"task {index}: {reason}" for index, reason in sorted(failures.items()))
        raise RuntimeError(f"atomic hierarchical group {group_id} failed; no group results were committed: {detail}")
    results = [outcomes[index] for index, _ in jobs]
    groups[group_id] = {
        "status": "completed",
        "input_hashes": hashes,
        "task_ids": [f"{group_id}:{index:03d}:{hash[:12]}" for index, (_, hash) in jobs],
        "results": results,
    }
    _write_workspace(project.root, workspace)
    for index, packet in enumerate(packets, start=1):
        _append_context_audit(project.root, packet, status="completed", task_suffix=f"{group_id}:{index:03d}")
    return results


def _complete_one_memo(
    packet: dict[str, Any], complete: Completion, attempts: int,
    validate: Callable[[dict[str, Any], dict[str, Any]], None],
    ensure_budget: Callable[[dict[str, Any]], None], result_artifact: Path, packet_artifact: Path,
) -> dict[str, Any]:
    packet[CONTEXT_AUDIT_KEY]["estimated_input_tokens"] = packet_token_estimate(public_packet(packet))
    ensure_budget(packet)
    write_json(packet_artifact, packet)
    candidate = _retry_fixed_packet(packet, complete, attempts, validate)
    write_json(result_artifact, candidate)
    return candidate


def _reduction_group_source_ids(
    state: GroundedTheoryState, group: list[dict[str, Any]],
) -> dict[str, list[str]]:
    """Return only source IDs both present in this group and canonical in state."""
    referenced_ids: set[str] = set()
    for worker_result in group:
        if not isinstance(worker_result, dict):
            continue
        source_ids = worker_result.get("source_ids")
        if isinstance(source_ids, dict):
            for values in source_ids.values():
                if isinstance(values, list):
                    referenced_ids.update(value for value in values if isinstance(value, str))
        handoff = worker_result.get("handoff")
        if isinstance(handoff, dict):
            rehydrate = handoff.get("evidence_to_rehydrate")
            if isinstance(rehydrate, list):
                referenced_ids.update(value for value in rehydrate if isinstance(value, str))
    canonical_memos = {memo.id for memo in state.memos}
    canonical_records = {record.id for record in state.records}
    return {
        "memos": sorted(referenced_ids & canonical_memos),
        "records": sorted(referenced_ids & canonical_records),
    }


def _retry_fixed_packet(
    packet: dict[str, Any], complete: Completion, attempts: int,
    validate: Callable[[dict[str, Any], dict[str, Any]], None],
) -> dict[str, Any]:
    failures: list[str] = []
    feedback: str | None = None
    for attempt in range(1, attempts + 1):
        try:
            candidate = complete(packet, feedback)
            validate(packet, candidate)
            _validate_task_aware_handoff(packet, candidate)
            return candidate
        except (ValueError, RuntimeError) as error:
            feedback = str(error)
            failures.append(f"attempt {attempt}: {feedback}")
    raise RuntimeError("; ".join(failures))


def _validate_task_aware_handoff(packet: dict[str, Any], candidate: dict[str, Any]) -> None:
    """Apply the compact reducer's declared source-provenance boundary."""
    if packet.get("action") != REDUCE_MEMO_ACTION:
        return
    source_ids = candidate.get("source_ids")
    memo_ids = source_ids.get("memos") if isinstance(source_ids, dict) else None
    record_ids = source_ids.get("records") if isinstance(source_ids, dict) else None
    if (
        set(source_ids or {}) != {"memos", "records"}
        or not isinstance(memo_ids, list)
        or not isinstance(record_ids, list)
        or not all(isinstance(item, str) for item in [*memo_ids, *record_ids])
    ):
        raise ValueError("reduced handoff source_ids must contain only memo and record ID arrays")
    contract = packet.get("worker_handoff_contract")
    allowed_source_ids = contract.get("allowed_source_ids") if isinstance(contract, dict) else None
    if not isinstance(allowed_source_ids, dict):
        raise ValueError("reduced handoff packet is missing its allowed source IDs")
    allowed_memos = allowed_source_ids.get("memos")
    allowed_records = allowed_source_ids.get("records")
    if not isinstance(allowed_memos, list) or not isinstance(allowed_records, list):
        raise ValueError("reduced handoff packet has malformed allowed source IDs")
    unsupplied = sorted((set(memo_ids) - set(allowed_memos)) | (set(record_ids) - set(allowed_records)))
    if unsupplied:
        raise ValueError("reduced handoff may cite only IDs supplied by its reduction packet; unsupplied IDs: " + ", ".join(unsupplied))
    allowed = set(allowed_memos) | set(allowed_records)
    handoff = candidate.get("handoff")
    if not isinstance(handoff, dict):
        raise ValueError("reduced handoff is missing its source-referenced handoff")
    for field in ("decision_relevant_findings", "uncertainties"):
        for item in handoff.get(field, []):
            references = item.get("source_references") if isinstance(item, dict) else None
            if not isinstance(references, list) or not all(isinstance(value, str) and value in allowed for value in references):
                raise ValueError(f"reduced handoff {field} must cite only its declared source IDs")
    rehydrate = handoff.get("evidence_to_rehydrate", [])
    if not isinstance(rehydrate, list) or not all(isinstance(value, str) and value in allowed for value in rehydrate):
        raise ValueError("reduced handoff evidence_to_rehydrate must cite only its declared source IDs")


def _parent_handoff(worker_result: dict[str, Any]) -> dict[str, Any]:
    """Project one validated worker result into the parent-visible context.

    This is deliberately not another generated summary.  It is a mechanical,
    lossless selection of the worker's task-aware output contract and its
    provenance.  The complete worker result remains in the integration
    workspace under its packet hash for inspection, retry, and rehydration.
    """
    handoff = worker_result.get("handoff")
    if not isinstance(handoff, dict):
        raise ValueError("validated worker result is missing its task-aware handoff")
    source_ids = worker_result.get("source_ids")
    if not isinstance(source_ids, dict):
        raise ValueError("validated worker result is missing source IDs")
    return {
        "neighborhood_id": worker_result.get("neighborhood_id"),
        "next_task": handoff.get("next_task"),
        "decision_relevant_findings": handoff.get("decision_relevant_findings", []),
        "uncertainties": handoff.get("uncertainties", []),
        "evidence_to_rehydrate": handoff.get("evidence_to_rehydrate", []),
        "negative_case_ids": worker_result.get("negative_case_ids", []),
        "source_ids": source_ids,
        "provenance": {
            "worker_result_sha256": _stable_hash(worker_result),
            "workspace": WORKSPACE_FILE,
            "source_of_truth": "canonical_project_state",
            "rehydration_rule": (
                "Use source_ids or evidence_to_rehydrate to load the durable source object; "
                "the complete worker result is retained in the named workspace."
            ),
        },
        "projection_rule": (
            "Task-aware parent projection; complete worker memo remains in the "
            "integration workspace and canonical project state remains authoritative."
        ),
    }


def _state_snapshot(state: GroundedTheoryState) -> dict[str, Any]:
    """Private state identity copied to receipts, never to a worker prompt."""
    return {
        "id": f"state-revision-{state.revision:08d}",
        "revision": state.revision,
    }


def _load_workspace(root: Path, revision: int) -> dict[str, Any]:
    path = integration_workspace_path(root)
    if path.exists():
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, dict) and value.get("state_revision") == revision:
                return value
        except (OSError, json.JSONDecodeError):
            pass
    return {"schema_version": 1, "state_revision": revision, "groups": {}}


def _write_workspace(root: Path, workspace: dict[str, Any]) -> None:
    write_json(integration_workspace_path(root), workspace)


def _append_context_audit(root: Path, packet: dict[str, Any], *, status: str, task_suffix: str) -> None:
    audit = packet.get(CONTEXT_AUDIT_KEY)
    if not isinstance(audit, dict):
        return
    value = {
        **audit,
        **context_receipt_fields(packet),
        "task_id": f"{packet.get('action', 'unknown')}:{task_suffix}",
        "status": status,
        "model_configuration": audit.get("model_configuration", "Prime stateless"),
    }
    with (root / "context_audit.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def _stable_hash(value: dict[str, Any]) -> str:
    raw = json.dumps(public_packet(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _cross_neighborhood_relations(state: GroundedTheoryState, neighborhoods: list[dict[str, Any]]) -> list[dict[str, Any]]:
    membership = {concept_id: item["id"] for item in neighborhoods for concept_id in item["concept_ids"]}
    return [
        _relation_summary(relation) | {
            "source_neighborhood_id": membership.get(relation.source_concept_id),
            "target_neighborhood_id": membership.get(relation.target_concept_id),
        }
        for relation in state.relationships
        if membership.get(relation.source_concept_id) != membership.get(relation.target_concept_id)
    ]


def _bounded_items(values: list[Any], limit: int = 8) -> list[Any]:
    if len(values) <= limit:
        return values
    first = limit // 2
    return values[:first] + values[-(limit - first):]


def _bounded_record_ids(evidence: list[Any], limit: int = 8) -> list[str]:
    return _bounded_items(sorted({item.record_id for item in evidence}), limit)


def _concept_summary(item: Any) -> dict[str, Any]:
    return {
        "id": item.id,
        "label": item.label,
        "definition": compact_text(item.definition, 600),
        "level": item.level,
        "status": item.status,
        "parent_category_id": item.parent_category_id,
        "evidence_count": len(item.evidence),
        "evidence_record_ids": _bounded_record_ids(item.evidence),
        "definition_revision_count": len(item.definition_revisions),
        "variation_count": len(item.variations),
        "negative_or_boundary_case_count": len(item.negative_or_boundary_cases),
    }


def _relation_summary(item: Any) -> dict[str, Any]:
    return {
        "id": item.id, "source_concept_id": item.source_concept_id,
        "relationship": item.relationship, "target_concept_id": item.target_concept_id,
        "status": item.status, "grounding_kind": item.grounding_kind,
        "conditions": _bounded_items(item.conditions), "condition_count": len(item.conditions),
        "evidence_record_ids": _bounded_record_ids(item.evidence), "evidence_count": len(item.evidence),
        "negative_evidence_record_ids": _bounded_record_ids(item.negative_cases),
        "negative_evidence_count": len(item.negative_cases),
    }


def _process_summary(item: Any) -> dict[str, Any]:
    return {
        "id": item.id, "label": item.label, "description": compact_text(item.description, 600),
        "status": item.status, "conditions": _bounded_items(item.conditions), "condition_count": len(item.conditions),
        "actions_interactions": _bounded_items(item.actions_interactions), "action_count": len(item.actions_interactions),
        "consequences": _bounded_items(item.consequences), "consequence_count": len(item.consequences),
        "alternative_pathways": _bounded_items(item.alternative_pathways), "alternative_pathway_count": len(item.alternative_pathways),
        "supporting_relation_ids": _bounded_items(item.supporting_relation_ids), "supporting_relation_count": len(item.supporting_relation_ids),
        "negative_evidence_record_ids": _bounded_record_ids(item.negative_cases), "negative_evidence_count": len(item.negative_cases),
        "edges": [
            {"source_concept_id": edge.source_concept_id, "relationship": edge.relationship,
             "target_concept_id": edge.target_concept_id, "status": edge.status,
             "supporting_record_ids": _bounded_items(edge.supporting_record_ids),
             "supporting_record_count": len(edge.supporting_record_ids)}
            for edge in _bounded_items(item.edges)
        ],
        "edge_count": len(item.edges),
    }


def _memo_summary(item: Any) -> dict[str, Any]:
    return {
        "id": item.id, "topic": item.topic,
        "analytic_observation": compact_text(item.analytic_observation, 600),
        "tentative_interpretation": compact_text(item.tentative_interpretation, 500),
        "uncertainties": compact_text(item.uncertainties, 400),
        "linked_concept_ids": _bounded_items(item.linked_concept_ids), "linked_concept_count": len(item.linked_concept_ids),
        "linked_relation_ids": _bounded_items(item.linked_relation_ids), "linked_relation_count": len(item.linked_relation_ids),
        "linked_process_ids": _bounded_items(item.linked_process_ids), "linked_process_count": len(item.linked_process_ids),
        "supporting_evidence_record_ids": _bounded_record_ids(item.supporting_evidence),
        "supporting_evidence_count": len(item.supporting_evidence),
        "negative_evidence_record_ids": _bounded_record_ids(item.negative_or_contradictory_cases),
        "negative_evidence_count": len(item.negative_or_contradictory_cases),
    }


def _negative_summary(item: Any) -> dict[str, Any]:
    return {
        "id": item.id, "target_type": item.target_type, "target_id": item.target_id,
        "description": compact_text(item.description, 600),
        "evidence_record_ids": _bounded_record_ids(item.evidence), "evidence_count": len(item.evidence),
    }
