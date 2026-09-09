"""PRIME-driven graph-to-theory-to-measurement workflow."""

from __future__ import annotations

import json
from typing import cast

from .models import (
    DataDesign,
    DiscoveryDocument,
    EvidenceReference,
    FalsificationTest,
    FalsificationSpecification,
    GraphEdge,
    GraphNode,
    Measurement,
    MeasurementPlan,
    MeasurementRule,
    MeasurementRun,
    MeasurementSpecification,
    Prediction,
    ProjectState,
    ReadinessAssessment,
    ReadinessParentType,
    ReadinessStatus,
    RelationDirection,
    RelationshipCandidate,
    RelationType,
    StudyConfig,
    TestImplementability,
    TestRelevance,
    Theory,
    TheoryStatement,
    ValidationResult,
    ValidationPartition,
    ValidationStatus,
    WorkflowStatus,
)
from .project import TheoryDiscoveryProject


async def run(
    project_path: str,
    action: str = "status",
    payload_json: str = "{}",
) -> str:
    """Create, inspect, or submit one stage of a PRIME-driven workflow."""

    if action == "init":
        payload = _parse_payload(payload_json)
        project = TheoryDiscoveryProject.create(
            project_path,
            study_id=_required_string(payload, "study_id"),
            question=_required_string(payload, "question"),
            evidence_records=_object_list(payload, "evidence_records"),
            graph_source=_optional_string(payload, "graph_source"),
            discovery_documents=_object_list(payload, "discovery_documents"),
            validation_data=_optional_object(payload, "validation_data"),
        )
        result = project.status()
    else:
        project = TheoryDiscoveryProject.open(project_path)
        if action == "status":
            result = project.status()
        elif action == "next":
            result = project.next_action()
        elif action == "task-packet":
            result = project.task_packet()
        elif action == "submit":
            try:
                result = project.submit(_parse_payload(payload_json))
            except ValueError as error:
                result = {
                    "accepted": False,
                    "retry": True,
                    "errors": [str(error)],
                    "next_action": project.next_action(),
                    "retry_instruction": (
                        "Correct every reported validation error, regenerate only the "
                        "current stage, and resubmit it."
                    ),
                }
        elif action == "record-validation":
            payload = _parse_payload(payload_json)
            result = {
                "validation": project.record_validation(
                    prediction_id=_required_string(payload, "prediction_id"),
                    status=cast(ValidationStatus, _required_string(payload, "status")),
                    summary=_required_string(payload, "summary"),
                    evidence=_object_list(payload, "evidence"),
                    validation_id=_optional_string(payload, "id") or None,
                    execution_artifact=_optional_string(payload, "execution_artifact") or "",
                    falsification_specification_id=(
                        _optional_string(payload, "falsification_specification_id")
                        or ""
                    ),
                    specification_hash=_optional_string(payload, "specification_hash") or "",
                )
            }
        elif action == "configure-validation-execution":
            payload = _parse_payload(payload_json)
            result = {
                "validation_data": project.configure_validation_execution(
                    _required_bool(payload, "auto_execute")
                )
            }
        else:
            raise ValueError(
                "action must be init, status, next, task-packet, submit, record-validation, or configure-validation-execution"
            )
    return json.dumps(_json_ready(result), ensure_ascii=False, indent=2, sort_keys=True)


def _parse_payload(payload_json: str) -> dict[str, object]:
    payload = json.loads(payload_json)
    if not isinstance(payload, dict) or not all(isinstance(key, str) for key in payload):
        raise ValueError("payload_json must encode an object with string keys")
    return payload


def _required_string(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"payload field {key!r} must be a non-empty string")
    return value


def _optional_string(payload: dict[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"payload field {key!r} must be a string or null")
    return value


def _required_bool(payload: dict[str, object], key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"payload field {key!r} must be a boolean")
    return value


def _optional_object(payload: dict[str, object], key: str) -> dict[str, object] | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, dict) or not all(isinstance(item_key, str) for item_key in value):
        raise ValueError(f"payload field {key!r} must be an object or null")
    return value


def _object_list(payload: dict[str, object], key: str) -> list[dict[str, object]]:
    value = payload.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"payload field {key!r} must be an array of objects")
    return value


def _json_ready(value: object) -> object:
    """Convert dataclasses returned by the optional validation action."""
    if hasattr(value, "__dataclass_fields__"):
        from dataclasses import asdict

        return asdict(value)
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


__all__ = [
    "DataDesign",
    "DiscoveryDocument",
    "EvidenceReference",
    "FalsificationTest",
    "FalsificationSpecification",
    "GraphEdge",
    "GraphNode",
    "Measurement",
    "MeasurementPlan",
    "MeasurementRule",
    "MeasurementRun",
    "MeasurementSpecification",
    "Prediction",
    "ProjectState",
    "ReadinessAssessment",
    "ReadinessParentType",
    "ReadinessStatus",
    "RelationDirection",
    "RelationshipCandidate",
    "RelationType",
    "StudyConfig",
    "TestImplementability",
    "TestRelevance",
    "Theory",
    "TheoryStatement",
    "TheoryDiscoveryProject",
    "ValidationResult",
    "ValidationPartition",
    "ValidationStatus",
    "WorkflowStatus",
    "run",
]
