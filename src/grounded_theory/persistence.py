"""State loading, CSV input, JSON output, and the reviewable report."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .models import (
    Concept,
    Evidence,
    GroundedTheoryConfig,
    GroundedTheoryState,
    IntegratedTheory,
    Memo,
    NegativeCase,
    Process,
    ProcessEdge,
    QualitativeRecord,
    Relationship,
    TheoreticalProposition,
    TheoreticalSamplingNeed,
)
from .validation import (
    evidence_from_dict,
    object_list,
    object_value,
    optional_text,
    string_list,
    text,
)


SCHEMA_VERSION = 3
STATE_FILE = "analysis_state.json"
EVENTS_FILE = "events.jsonl"


def records_from_csv(
    path: str | Path, *, text_column: str = "text_review", id_column: str | None = None
) -> list[dict[str, str]]:
    csv_path = Path(path)
    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if text_column not in (reader.fieldnames or []):
            raise ValueError(f"CSV does not contain text column {text_column!r}")
        if id_column is not None and id_column not in (reader.fieldnames or []):
            raise ValueError(f"CSV does not contain ID column {id_column!r}")
        records = []
        for index, row in enumerate(reader, start=1):
            record_id = (row[id_column] if id_column else f"record_{index:04d}").strip()
            records.append({"id": record_id, "text": row[text_column], "source": csv_path.name})
    return records


def record_from_value(value: dict[str, str] | QualitativeRecord) -> QualitativeRecord:
    if isinstance(value, QualitativeRecord):
        return value
    if not isinstance(value, dict):
        raise ValueError("each record must be an object")
    return QualitativeRecord(
        id=text(value, "id"), text=text(value, "text"), source=optional_text(value, "source") or ""
    )


def validate_records(records: list[QualitativeRecord]) -> None:
    if not records:
        raise ValueError("at least one qualitative record is required")
    identifiers = [item.id for item in records]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("record IDs must be unique")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary_name = handle.name
    os.replace(temporary_name, path)


def state_from_dict(raw: dict[str, Any]) -> GroundedTheoryState:
    if not isinstance(raw, dict) or raw.get("schema_version") not in {1, 2, SCHEMA_VERSION}:
        raise ValueError("unsupported Grounded Theory state schema")
    records = [QualitativeRecord(**item) for item in object_list(raw, "records")]
    validate_records(records)
    integrated_raw = raw.get("integrated_theory")
    return GroundedTheoryState(
        schema_version=SCHEMA_VERSION,
        revision=int(raw.get("revision", 0)),
        config=GroundedTheoryConfig(**object_value(raw, "config")),
        analysis_metadata=dict(raw.get("analysis_metadata", {})),
        records=records,
        concepts=[concept_from_dict(item) for item in object_list(raw, "concepts")],
        relationships=[
            relationship_from_dict(item) for item in object_list(raw, "relationships")
        ],
        processes=[process_from_dict(item) for item in object_list(raw, "processes")],
        memos=[memo_from_dict(item) for item in object_list(raw, "memos")],
        negative_cases=[
            negative_case_from_dict(item) for item in object_list(raw, "negative_cases")
        ],
        theoretical_sampling_needs=[
            sampling_need_from_dict(item)
            for item in object_list(raw, "theoretical_sampling_needs")
        ],
        integrated_theory=(
            integrated_theory_from_dict(integrated_raw)
            if isinstance(integrated_raw, dict)
            else None
        ),
        open_coded_record_ids=string_list(raw, "open_coded_record_ids", allow_empty=True),
        relationally_analyzed_record_ids=string_list(
            raw, "relationally_analyzed_record_ids", allow_empty=True
        ),
        pending_relational_payload=(
            object_value(raw, "pending_relational_payload")
            if isinstance(raw.get("pending_relational_payload"), dict)
            else None
        ),
        pending_relational_record_ids=string_list(
            raw, "pending_relational_record_ids", allow_empty=True
        ),
        pending_relational_review_results=object_list(
            raw, "pending_relational_review_results", allow_empty=True
        ),
        relational_validation_feedback=string_list(
            raw, "relational_validation_feedback", allow_empty=True
        ),
        relational_validation_attempts=int(raw.get("relational_validation_attempts", 0)),
        relational_validation_blocked=bool(raw.get("relational_validation_blocked", False)),
    )


def concept_from_dict(raw: dict[str, Any]) -> Concept:
    copied = dict(raw)
    copied["evidence"] = [evidence_from_dict(item) for item in object_list(raw, "evidence")]
    copied["negative_or_boundary_cases"] = [
        evidence_from_dict(item) for item in object_list(raw, "negative_or_boundary_cases")
    ]
    copied.setdefault("definition_revisions", [])
    return Concept(**copied)


def relationship_from_dict(raw: dict[str, Any]) -> Relationship:
    copied = dict(raw)
    copied.setdefault("grounding_kind", "tentative_theoretical_inference")
    copied.setdefault("grounding_explanation", "Migrated state without explicit grounding metadata.")
    copied["evidence"] = [evidence_from_dict(item) for item in object_list(raw, "evidence")]
    copied["negative_cases"] = [
        evidence_from_dict(item) for item in object_list(raw, "negative_cases")
    ]
    copied.setdefault("conditions", [])
    return Relationship(**copied)


def process_from_dict(raw: dict[str, Any]) -> Process:
    copied = dict(raw)
    copied["negative_cases"] = [
        evidence_from_dict(item) for item in object_list(raw, "negative_cases")
    ]
    copied["edges"] = [
        ProcessEdge(
            source_concept_id=text(item, "source_concept_id"),
            relationship=text(item, "relationship"),
            target_concept_id=text(item, "target_concept_id"),
            supporting_relation_ids=string_list(item, "supporting_relation_ids", allow_empty=True),
            conditions=string_list(item, "conditions", allow_empty=True),
        )
        for item in object_list(raw, "edges", allow_empty=True)
    ]
    return Process(**copied)


def memo_from_dict(raw: dict[str, Any]) -> Memo:
    copied = dict(raw)
    copied["supporting_evidence"] = [
        evidence_from_dict(item) for item in object_list(raw, "supporting_evidence")
    ]
    copied["negative_or_contradictory_cases"] = [
        evidence_from_dict(item)
        for item in object_list(raw, "negative_or_contradictory_cases")
    ]
    return Memo(**copied)


def negative_case_from_dict(raw: dict[str, Any]) -> NegativeCase:
    copied = dict(raw)
    copied["evidence"] = [evidence_from_dict(item) for item in object_list(raw, "evidence")]
    return NegativeCase(**copied)


def sampling_need_from_dict(raw: dict[str, Any]) -> TheoreticalSamplingNeed:
    copied = dict(raw)
    copied["evidence"] = [
        evidence_from_dict(item) for item in object_list(raw, "evidence", allow_empty=True)
    ]
    return TheoreticalSamplingNeed(**copied)


def integrated_theory_from_dict(raw: dict[str, Any]) -> IntegratedTheory:
    copied = dict(raw)
    propositions = []
    for item in object_list(raw, "propositions"):
        item_copy = dict(item)
        item_copy["evidence"] = [
            evidence_from_dict(value) for value in object_list(item, "evidence")
        ]
        propositions.append(TheoreticalProposition(**item_copy))
    copied["propositions"] = propositions
    return IntegratedTheory(**copied)


def render_analysis_report(state: GroundedTheoryState) -> str:
    lines = ["# Grounded Theory analysis report", "", "## Research question", "", state.config.research_question, "", "## Major concepts", ""]
    lines.extend(f"- `{item.id}` — {item.label}: {item.definition}" for item in state.concepts)
    lines.extend(["", "## Major relationships", ""])
    lines.extend(
        f"- `{item.id}` — `{item.source_concept_id}` {item.relationship} `{item.target_concept_id}` ({item.status}; {item.grounding_kind})"
        for item in state.relationships
    )
    lines.extend(["", "## Process patterns", ""])
    lines.extend(
        f"- `{item.id}` — {item.label}: {item.description} (relations: {', '.join(item.supporting_relation_ids)})"
        for item in state.processes
    )
    lines.extend(["", "## Integration", ""])
    theory = state.integrated_theory
    if theory is None:
        lines.append("Integration has not yet been run.")
    elif theory.status == "no_adequately_grounded_core_category":
        lines.extend(["No adequately grounded core category was identified.", "", theory.account])
    else:
        lines.extend([f"Theory: `{theory.id}`; core category: `{theory.core_category_id}`", "", theory.account])
        for proposition in theory.propositions:
            support = ", ".join([*proposition.process_ids, *proposition.relation_ids])
            lines.append(f"- `{proposition.id}` ({proposition.status}; {support}): {proposition.statement}")
    lines.extend(["", "## Negative cases", ""])
    lines.extend(f"- `{item.id}` challenges `{item.target_id}`: {item.description}" for item in state.negative_cases)
    lines.extend(["", "## Important variations", ""])
    for concept in state.concepts:
        lines.extend(f"- concept `{concept.id}`: {item['description']}" for item in concept.variations)
    for relationship in state.relationships:
        lines.extend(f"- relationship `{relationship.id}`: {item['description']}" for item in relationship.variations)
    for process in state.processes:
        lines.extend(f"- process `{process.id}`: {path}" for path in process.alternative_pathways)
    lines.extend(["", "## Unresolved questions", ""])
    if theory is not None:
        lines.extend(f"- theory `{theory.id}`: {question}" for question in theory.unresolved_questions)
    for memo in state.memos:
        lines.extend(f"- memo `{memo.id}`: {question}" for question in memo.questions_for_further_analysis)
    lines.extend(["", "## Theoretical sampling needs", ""])
    lines.extend(
        f"- `{item.id}` — {item.target}: {item.evidence_needed} (records: {', '.join(evidence.record_id for evidence in item.evidence)})"
        for item in state.theoretical_sampling_needs
    )
    return "\n".join(lines) + "\n"


def state_to_dict(state: GroundedTheoryState) -> dict[str, Any]:
    return asdict(state)
