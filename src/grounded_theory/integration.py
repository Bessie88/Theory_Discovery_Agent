"""Stage 3: grounded theoretical integration and unresolved sampling needs."""

from __future__ import annotations

from typing import Any

from .memoing import add_analytic_memo, apply_memo_updates
from .models import Evidence, GroundedTheoryState, IntegratedTheory, TheoreticalProposition, TheoreticalSamplingNeed
from .validation import (
    by_id,
    choice,
    dedupe_evidence,
    evidence_list,
    evidence_list_from_key,
    existing_ids,
    next_id,
    object_list,
    object_value,
    optional_text,
    string_list,
    text,
)


def apply_integration(state: GroundedTheoryState, payload: dict[str, Any]) -> dict[str, Any]:
    raw = object_value(payload, "integration")
    status = choice(raw, "status", {"integrated", "no_adequately_grounded_core_category"})
    core_category_id = optional_text(raw, "core_category_id")
    if status == "integrated":
        if core_category_id is None:
            raise ValueError("an integrated theory requires core_category_id")
        by_id(state.concepts, core_category_id, "core category")
    elif core_category_id is not None:
        raise ValueError("no-core-category outcome must use null core_category_id")
    propositions = _propositions(state, raw)
    negative_case_ids = existing_ids(
        state.negative_cases,
        string_list(raw, "negative_case_ids", allow_empty=True),
        "negative case",
    )
    if status == "integrated" and not propositions:
        raise ValueError(
            "an integrated theory needs at least one traceable proposition; otherwise report no adequately grounded core category"
        )
    state.integrated_theory = IntegratedTheory(
        status=status,  # type: ignore[arg-type]
        core_category_id=core_category_id,
        account=text(raw, "account"),
        propositions=propositions,
        alternative_pathways=string_list(raw, "alternative_pathways", allow_empty=True),
        negative_case_ids=negative_case_ids,
        unresolved_questions=string_list(raw, "unresolved_questions", allow_empty=True),
    )
    if status == "integrated" and core_category_id is not None:
        theory = state.integrated_theory
        add_analytic_memo(
            state,
            topic=f"Core category candidate: {core_category_id}",
            observation=theory.account,
            evidence=integration_evidence(state, propositions),
            comparisons="Theoretical integration linked cited concepts, relationships, processes, and negative cases.",
            concept_ids=[core_category_id],
            relation_ids=[item for proposition in propositions for item in proposition.relation_ids],
            process_ids=[item for proposition in propositions for item in proposition.process_ids],
        )
    _sampling_needs(state, payload)
    apply_memo_updates(
        state,
        object_list(payload, "memo_updates", allow_empty=True),
        allowed_record_ids=None,
    )
    return {
        "integration_status": status,
        "propositions": len(propositions),
        "sampling_needs": len(state.theoretical_sampling_needs),
    }


def _propositions(
    state: GroundedTheoryState, raw: dict[str, Any]
) -> list[TheoreticalProposition]:
    propositions: list[TheoreticalProposition] = []
    for item in object_list(raw, "propositions", allow_empty=True):
        process_ids = existing_ids(
            state.processes, string_list(item, "process_ids", allow_empty=True), "process"
        )
        relation_ids = existing_ids(
            state.relationships,
            string_list(item, "relation_ids", allow_empty=True),
            "relationship",
        )
        evidence = evidence_list(item, state, allow_empty=True)
        if not process_ids and not relation_ids and not evidence:
            raise ValueError("every theoretical proposition needs traceable support")
        propositions.append(
            TheoreticalProposition(
                id=next_id(propositions, "proposition"),
                statement=text(item, "statement"),
                status=choice(item, "status", {"well_grounded", "tentative", "unresolved"}),  # type: ignore[arg-type]
                process_ids=process_ids,
                relation_ids=relation_ids,
                evidence=evidence,
            )
        )
    return propositions


def _sampling_needs(state: GroundedTheoryState, payload: dict[str, Any]) -> None:
    for item in object_list(payload, "theoretical_sampling_needs", allow_empty=True):
        evidence = evidence_list_from_key(item, "evidence", state)
        need = TheoreticalSamplingNeed(
            id=next_id(state.theoretical_sampling_needs, "sampling_need"),
            target=text(item, "target"),
            reason=text(item, "reason"),
            evidence_needed=text(item, "evidence_needed"),
            evidence=evidence,
            status=choice(item, "status", {"unresolved"}, default="unresolved"),
        )
        state.theoretical_sampling_needs.append(need)
        add_analytic_memo(
            state,
            topic=f"Unresolved theoretical sampling need: {need.target}",
            observation=need.reason,
            evidence=need.evidence,
            comparisons="The fixed corpus leaves this analytic target underdeveloped.",
        )


def integration_evidence(
    state: GroundedTheoryState, propositions: list[TheoreticalProposition]
) -> list[Evidence]:
    relations = {relation.id: relation for relation in state.relationships}
    processes = {process.id: process for process in state.processes}
    evidence: list[Evidence] = []
    for proposition in propositions:
        evidence.extend(proposition.evidence)
        for relation_id in proposition.relation_ids:
            evidence.extend(relations[relation_id].evidence)
        for process_id in proposition.process_ids:
            process = processes[process_id]
            for relation_id in process.supporting_relation_ids:
                evidence.extend(relations[relation_id].evidence)
            evidence.extend(process.negative_cases)
    return dedupe_evidence(evidence)
