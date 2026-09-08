"""Persistent state machine for the literature-grounded theory MVP."""

from __future__ import annotations

import json
import os
import re
import tempfile
import csv
from hashlib import sha256
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import cast
from uuid import uuid4

from .models import (
    DataDesign,
    DiscoveryDocument,
    EvidenceReference,
    EvidenceRole,
    FalsificationTest,
    FalsificationSpecification,
    GraphEdge,
    GraphNode,
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
    ValidationPartition,
    ValidationResult,
    ValidationStatus,
    WorkflowStatus,
    utc_now,
)


STATE_FILE = "state.json"
EVENTS_FILE = "events.jsonl"
SCHEMA_VERSION = 13
PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"
EXECUTION_DIRECTORY = "validation-execution"
MEASUREMENT_DIRECTORY = "validation-measurements"
MAX_MEASUREMENT_REPAIR_ROUNDS = 2

# This is protocol, not model-generated content.  A measurement question is
# useful only when the whole decoding contract is frozen with it *before* the
# validation partition is made available to a model.
FROZEN_MEASUREMENT_PROTOCOL: dict[str, object] = {
    "labels": ["PRESENT", "ABSENT", "UNCERTAIN"],
    "model_id": "qwen3.6-27b-fp8",
    "model_revision": "qwen3.6-27b-fp8",
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 16,
    "prompt_version": "frozen-measurement-v1",
    "uncertain_handling": "exclude",
}

THEORY_FIELDS = (
    "description",
    "theory_statements",
    "conflicting_evidence",
    "unaccounted_evidence",
    "new_predictions_likely",
    "new_predictions_unknown",
    "negative_experiments",
)
PREDICTION_FIELDS = (
    "law_id",
    "specific_prediction",
    "operational_signals",
    "strong_test_requirement",
    "support_criteria",
    "contradiction_criteria",
    "theory_id",
)
MEASUREMENT_FIELDS = ("indicators", "operationalization", "data_design")
SPECIFICATION_FIELDS = ("execution_contract",)

# A deliberately small test language.  Each entry fixes the group assignment,
# estimand, and decision semantics, leaving Stage 05 only to bind already
# frozen measurement rules to the variables in the template.
FALSIFICATION_TEMPLATES: dict[str, dict[str, object]] = {
    "directional_prevalence_contrast_v1": {
        "measurement_mode": "fixed_variables",
        "measurement_keys": ("condition", "outcome"),
        "expression_keys": (),
        "relation": "higher_prevalence",
        "comparison": {
            "group_a": "condition_is_true",
            "group_b": "condition_is_false",
            "estimand": "outcome_prevalence_difference",
        },
        "hypotheses": {
            "expected": "observed_difference_positive",
            "opposite": "observed_difference_non_positive",
        },
        "decision_rule": {
            "type": "directional_prevalence",
            "consistent_when": "observed_difference_positive",
            "contradicted_when": "observed_difference_non_positive",
            "not_testable_when": "comparison_group_empty_or_required_field_missing",
        },
    },
    "cooccurrence_v1": {
        "measurement_mode": "fixed_variables",
        "measurement_keys": ("condition", "outcome"),
        "expression_keys": (),
        "relation": "cooccurs",
        "comparison": {
            "group_a": "condition_is_true",
            "group_b": "condition_is_false",
            "estimand": "outcome_prevalence_difference",
        },
        "hypotheses": {
            "expected": "observed_difference_positive",
            "opposite": "observed_difference_non_positive",
        },
        "decision_rule": {
            "type": "directional_prevalence",
            "consistent_when": "observed_difference_positive",
            "contradicted_when": "observed_difference_non_positive",
            "not_testable_when": "comparison_group_empty_or_required_field_missing",
        },
    },
    "necessary_condition_v1": {
        "measurement_mode": "fixed_variables",
        "measurement_keys": ("condition", "outcome"),
        "expression_keys": (),
        "relation": "necessary_condition",
        "comparison": {
            "group_a": "outcome_is_true",
            "group_b": "outcome_is_false",
            "estimand": "condition_absence_among_outcome_present",
        },
        "hypotheses": {
            "expected": "condition_absence_count_zero",
            "opposite": "condition_absence_count_nonzero",
        },
        "decision_rule": {
            "type": "necessary_condition",
            "consistent_when": "condition_absence_count_zero",
            "contradicted_when": "condition_absence_count_nonzero",
            "not_testable_when": "outcome_group_empty_or_required_field_missing",
        },
    },
    # These templates have explicitly declared comparison groups.  They use
    # only the frozen Boolean expression language validated below; the
    # compiler cannot add a post-hoc coding or decision rule.
    "explicit_group_prevalence_contrast_v1": {
        "measurement_mode": "expression_aliases",
        "measurement_keys": (),
        "expression_keys": ("group_a", "group_b", "outcome"),
        "relation": "higher_prevalence",
        "comparison": {
            "group_a": "expression_group_a_is_true_and_group_b_is_false",
            "group_b": "expression_group_b_is_true_and_group_a_is_false",
            "estimand": "outcome_prevalence_difference",
        },
        "hypotheses": {
            "expected": "observed_difference_positive",
            "opposite": "observed_difference_non_positive",
        },
        "decision_rule": {
            "type": "explicit_group_directional_prevalence",
            "consistent_when": "observed_difference_positive",
            "contradicted_when": "observed_difference_non_positive",
            "not_testable_when": "comparison_group_empty_or_required_field_missing",
        },
    },
    "majority_group_prevalence_contrast_v1": {
        "measurement_mode": "expression_aliases",
        "measurement_keys": (),
        "expression_keys": ("group_a", "group_b", "outcome"),
        "relation": "higher_prevalence_with_group_a_majority",
        "comparison": {
            "group_a": "expression_group_a_is_true_and_group_b_is_false",
            "group_b": "expression_group_b_is_true_and_group_a_is_false",
            "estimand": "outcome_prevalence_difference_and_group_a_majority",
        },
        "hypotheses": {
            "expected": "observed_difference_positive_and_group_a_outcome_rate_above_one_half",
            "opposite": "observed_difference_non_positive_or_group_a_outcome_rate_not_above_one_half",
        },
        "decision_rule": {
            "type": "explicit_group_majority_directional_prevalence",
            "consistent_when": "observed_difference_positive_and_group_a_outcome_rate_above_one_half",
            "contradicted_when": "observed_difference_non_positive_or_group_a_outcome_rate_not_above_one_half",
            "not_testable_when": "comparison_group_empty_or_required_field_missing",
        },
    },
}

ID_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,127}\Z")
NUMBER_PATTERN = re.compile(r"(?<![A-Za-z0-9_])\d+(?:\.\d+)?")
FORBIDDEN_THRESHOLD_OR_WINDOW_PATTERN = re.compile(
    r"""(?ix)
    (?:\b(?:p|alpha|α)\s*(?:<|>|≤|≥|=)\s*\d|(?:<|>|≤|≥|=)\s*\d|\b\d+(?:\.\d+)?\s*%)
    |\b(?:19|20)\d{2}\b
    |\b\d+\s*(?:day|week|month|year)s?\b
    """
)


def _load_prompt(file_name: str) -> str:
    return (PROMPTS_DIR / file_name).read_text(encoding="utf-8").strip()


def _migrate_v4_state(raw: dict[str, object]) -> dict[str, object]:
    """Convert the former single-law schema without discarding saved content."""
    migrated = deepcopy(raw)
    legacy_theories = _mapping_object_list(migrated, "theories", allow_empty=True)
    law_ids_by_theory: dict[str, str] = {}
    evidence_records_by_id: dict[str, dict[str, object]] = {}
    converted_theories: list[dict[str, object]] = []

    for legacy in legacy_theories:
        theory_id = _mapping_text(legacy, "id")
        law_id = f"{theory_id}_law_1"
        law_ids_by_theory[theory_id] = law_id
        evidence = _mapping_object_list(legacy, "evidence", allow_empty=True)
        for record in evidence:
            record_id = _mapping_optional_text(record, "id") or _new_id("evidence")
            evidence_records_by_id.setdefault(record_id, {**record, "id": record_id})
        supporting_evidence = [
            item for item in evidence if _mapping_optional_text(item, "role") != "conflicting"
        ]
        converted_theories.append(
            {
                "id": theory_id,
                "name": _mapping_text(legacy, "name"),
                "description": (
                    "Migrated from the prior single-law format. "
                    f"Previous proposed mechanism: {_mapping_text(legacy, 'mechanism')}"
                ),
                "theory_statements": [
                    {
                        "id": law_id,
                        "law": _mapping_text(legacy, "law"),
                        "scope": _mapping_text(legacy, "scope"),
                        "evidence": supporting_evidence or evidence,
                    }
                ],
                "conflicting_evidence": [
                    item
                    for item in evidence
                    if _mapping_optional_text(item, "role") == "conflicting"
                ],
                "unaccounted_evidence": [
                    item
                    for item in evidence
                    if _mapping_optional_text(item, "role") == "unclear"
                ],
                "new_predictions_likely": [],
                "new_predictions_unknown": [],
                "negative_experiments": [],
                "created_at": _mapping_optional_text(legacy, "created_at"),
            }
        )

    converted_predictions: list[dict[str, object]] = []
    for legacy in _mapping_object_list(migrated, "predictions", allow_empty=True):
        theory_id = _mapping_text(legacy, "theory_id")
        law_id = law_ids_by_theory.get(theory_id)
        if law_id is None:
            raise ValueError(f"legacy prediction references unknown theory ID: {theory_id}")
        condition = _mapping_text(legacy, "condition_or_exposure")
        outcome = _mapping_text(legacy, "expected_outcome")
        comparison = _mapping_text(legacy, "comparison")
        pattern = _mapping_text(legacy, "expected_pattern")
        rationale = _mapping_text(legacy, "rationale")
        converted_predictions.append(
            {
                "id": _mapping_text(legacy, "id"),
                "theory_id": theory_id,
                "law_id": law_id,
                "specific_prediction": _mapping_text(legacy, "statement"),
                "operational_signals": [condition, outcome],
                "strong_test_requirement": (
                    f"Compare {condition} with {comparison}; assess whether {pattern}."
                ),
                "support_criteria": f"The comparison shows {pattern}. {rationale}",
                "contradiction_criteria": (
                    f"The comparison does not show {pattern}."
                ),
                "created_at": _mapping_optional_text(legacy, "created_at"),
            }
        )

    migrated["schema_version"] = 5
    migrated["evidence_records"] = list(evidence_records_by_id.values())
    migrated["theories"] = converted_theories
    migrated["predictions"] = converted_predictions
    return migrated


def _migrate_v5_state(raw: dict[str, object]) -> dict[str, object]:
    """Add supplied-data and falsification-test state to the v5 schema."""
    migrated = deepcopy(raw)
    legacy_plans = _mapping_object_list(migrated, "measurement_plans", allow_empty=True)
    migrated["available_data"] = (
        _mapping_object(legacy_plans[0], "data_design") if legacy_plans else None
    )
    migrated["falsification_tests"] = []
    migrated["schema_version"] = 6
    return migrated


def _migrate_v6_state(raw: dict[str, object]) -> dict[str, object]:
    """Separate discovery material from data reserved for falsification."""
    migrated = deepcopy(raw)
    migrated.setdefault("discovery_documents", [])
    legacy_data = migrated.pop("available_data", None)
    if isinstance(legacy_data, dict):
        legacy_data.setdefault("partition_role", "unknown")
        legacy_data.setdefault(
            "separation_note",
            "Legacy project: separation from discovery material was not declared.",
        )
    migrated["validation_data"] = legacy_data
    migrated["schema_version"] = 7
    return migrated


def _migrate_v7_state(raw: dict[str, object]) -> dict[str, object]:
    """Add an opt-in, artifact-backed held-out execution stage."""
    migrated = deepcopy(raw)
    validation_data = migrated.get("validation_data")
    if isinstance(validation_data, dict):
        validation_data.setdefault("auto_execute", False)
    for result in _mapping_object_list(migrated, "validation_results", allow_empty=True):
        result.setdefault("execution_artifact", "")
    migrated["schema_version"] = 8
    return migrated


def _migrate_v8_state(raw: dict[str, object]) -> dict[str, object]:
    """Preserve legacy tests/results while starting the frozen compiler stage.

    Existing v8 executions were based on prose tests and are intentionally
    retained as historical pilot output.  They must not satisfy the v9
    compiler-backed execution requirement.
    """
    migrated = deepcopy(raw)
    validation_data = migrated.get("validation_data")
    if isinstance(validation_data, dict):
        validation_data.setdefault("measurement_rules", [])
    migrated.setdefault("falsification_specifications", [])
    for result in _mapping_object_list(migrated, "validation_results", allow_empty=True):
        result.setdefault("falsification_specification_id", "")
        result.setdefault("specification_hash", "")
        result.setdefault("result_semantics", "legacy_v1")
    migrated["schema_version"] = 9
    return migrated


def _migrate_v9_state(raw: dict[str, object]) -> dict[str, object]:
    """Start the frozen-question protocol without deleting prior pilot work.

    Version 9 bound templates to lexical rules.  Those contracts and their
    results are historical observations, not a valid run of the new protocol:
    no question/model/uncertainty handling was frozen before held-out access.
    """
    migrated = deepcopy(raw)
    prior = _mapping_object_list(migrated, "falsification_specifications", allow_empty=True)
    existing_legacy = _mapping_object_list(
        migrated, "legacy_falsification_specifications", allow_empty=True
    )
    for specification in [*existing_legacy, *prior]:
        specification.setdefault("measurement_protocol", "legacy_lexical_v1")
    migrated["legacy_falsification_specifications"] = [*existing_legacy, *prior]
    migrated["falsification_specifications"] = []
    migrated["measurement_specifications"] = []
    migrated["measurement_runs"] = []
    for result in _mapping_object_list(migrated, "validation_results", allow_empty=True):
        result.setdefault("result_semantics", "legacy_v9_lexical_pilot")
    migrated["schema_version"] = 10
    return migrated


def _migrate_v10_state(raw: dict[str, object]) -> dict[str, object]:
    """Move unrun Qwen3.8 measurement contracts aside before model isolation.

    The new protocol freezes Qwen3.6 as the blind held-out measurement model.
    A question generated under the discovery-side model may be reused only by
    generating a new contract before validation; a saved Qwen3.8 contract must
    never be relabelled as Qwen3.6 after it has measured held-out text.
    """
    migrated = deepcopy(raw)
    current = _mapping_object_list(
        migrated, "measurement_specifications", allow_empty=True
    )
    runs = _mapping_object_list(migrated, "measurement_runs", allow_empty=True)
    current_falsification = _mapping_object_list(
        migrated, "falsification_specifications", allow_empty=True
    )
    old_model_contracts = [
        item for item in current if item.get("model_id") == "qwen3.8-27b"
    ]
    if old_model_contracts and not runs and not current_falsification:
        archived = _mapping_object_list(
            migrated, "superseded_measurement_specifications", allow_empty=True
        )
        migrated["superseded_measurement_specifications"] = [
            *archived,
            *current,
        ]
        migrated["measurement_specifications"] = []
        old_ids = {str(item.get("id", "")) for item in current}
        migrated["readiness_assessments"] = [
            item
            for item in _mapping_object_list(
                migrated, "readiness_assessments", allow_empty=True
            )
            if not (
                item.get("parent_type") == "measurement"
                and item.get("parent_id") in old_ids
            )
        ]
    migrated.setdefault("superseded_measurement_specifications", [])
    migrated["schema_version"] = 11
    return migrated


def _migrate_v11_state(raw: dict[str, object]) -> dict[str, object]:
    """Require a fresh v12 compilation before any unmeasured contract runs.

    Schema v11 could only express a single condition against its complement.
    Its contracts therefore cannot faithfully represent predictions requiring
    explicit comparison groups, conjunctions, or thresholded composites.  If
    no held-out labels exist, archive those unrun contracts and regenerate the
    atomic questions plus v12 execution contracts.  Never rewrite a contract
    that has already measured held-out records.
    """
    migrated = deepcopy(raw)
    current_measurements = _mapping_object_list(
        migrated, "measurement_specifications", allow_empty=True
    )
    current_contracts = _mapping_object_list(
        migrated, "falsification_specifications", allow_empty=True
    )
    runs = _mapping_object_list(migrated, "measurement_runs", allow_empty=True)
    if not runs and (current_measurements or current_contracts):
        archived_measurements = _mapping_object_list(
            migrated, "superseded_measurement_specifications", allow_empty=True
        )
        migrated["superseded_measurement_specifications"] = [
            *archived_measurements,
            *current_measurements,
        ]
        archived_contracts = _mapping_object_list(
            migrated, "legacy_falsification_specifications", allow_empty=True
        )
        for contract in current_contracts:
            contract.setdefault(
                "measurement_protocol", "frozen_llm_question_v1_schema_v11_superseded"
            )
        migrated["legacy_falsification_specifications"] = [
            *archived_contracts,
            *current_contracts,
        ]
        measurement_ids = {str(item.get("id", "")) for item in current_measurements}
        contract_ids = {str(item.get("id", "")) for item in current_contracts}
        migrated["readiness_assessments"] = [
            item
            for item in _mapping_object_list(
                migrated, "readiness_assessments", allow_empty=True
            )
            if not (
                (item.get("parent_type") == "measurement" and item.get("parent_id") in measurement_ids)
                or (
                    item.get("parent_type") == "falsification_specification"
                    and item.get("parent_id") in contract_ids
                )
            )
        ]
        migrated["measurement_specifications"] = []
        migrated["falsification_specifications"] = []
    migrated["schema_version"] = 12
    return migrated


def _migrate_v12_state(raw: dict[str, object]) -> dict[str, object]:
    """Upgrade unrun contracts to the structured missing-construct protocol.

    v12 recorded free-text ``unresolved`` notes.  v13 needs a structured,
    pre-validation diagnosis before it can request an additional atomic
    question.  When held-out labels do not exist, archive the old contracts
    and recompile them under the new protocol.  A project that has already
    measured held-out records is never rewritten.
    """
    migrated = deepcopy(raw)
    current_contracts = _mapping_object_list(
        migrated, "falsification_specifications", allow_empty=True
    )
    runs = _mapping_object_list(migrated, "measurement_runs", allow_empty=True)
    migrated.setdefault("superseded_falsification_specifications", [])
    migrated.setdefault("measurement_repair_rounds", 0)
    migrated.setdefault("falsification_repair_prediction_ids", [])
    if not runs and current_contracts:
        blocked_contracts = [
            contract
            for contract in current_contracts
            if contract.get("readiness") != "ready"
        ]
        for contract in current_contracts:
            contract.setdefault("missing_measurement_constructs", [])
        if not blocked_contracts:
            migrated["schema_version"] = SCHEMA_VERSION
            return migrated
        archived = _mapping_object_list(
            migrated, "superseded_falsification_specifications", allow_empty=True
        )
        migrated["superseded_falsification_specifications"] = [
            *archived,
            *blocked_contracts,
        ]
        contract_ids = {str(item.get("id", "")) for item in blocked_contracts}
        migrated["readiness_assessments"] = [
            item
            for item in _mapping_object_list(
                migrated, "readiness_assessments", allow_empty=True
            )
            if not (
                item.get("parent_type") == "falsification_specification"
                and item.get("parent_id") in contract_ids
            )
        ]
        migrated["falsification_specifications"] = [
            contract
            for contract in current_contracts
            if contract.get("readiness") == "ready"
        ]
    migrated["schema_version"] = SCHEMA_VERSION
    return migrated


class TheoryDiscoveryProject:
    """A small single-writer store with optional, separate validation records."""

    def __init__(self, root: str | Path, state: ProjectState) -> None:
        self.root = Path(root).resolve()
        self.state = state

    @classmethod
    def create(
        cls,
        root: str | Path,
        study_id: str,
        question: str,
        evidence_records: list[dict[str, object]],
        graph_source: str | None = None,
        discovery_documents: list[dict[str, object]] | None = None,
        validation_data: dict[str, object] | None = None,
        available_data: dict[str, object] | None = None,
    ) -> TheoryDiscoveryProject:
        _require_text("study_id", study_id)
        _require_text("question", question)
        if not evidence_records:
            raise ValueError("evidence_records must not be empty")
        parsed_evidence_records = [_parse_evidence(item) for item in evidence_records]
        record_ids = [item.id for item in parsed_evidence_records]
        if len(record_ids) != len(set(record_ids)):
            raise ValueError("evidence_records must have unique IDs")
        for record_id in record_ids:
            _require_identifier("evidence record ID", record_id)
        if not discovery_documents:
            raise ValueError("discovery_documents must not be empty")
        parsed_discovery_documents = [
            _parse_discovery_document(item) for item in discovery_documents
        ]
        document_ids = [item.id for item in parsed_discovery_documents]
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("discovery_documents must have unique IDs")
        for document_id in document_ids:
            _require_identifier("discovery document ID", document_id)
        if validation_data is not None and available_data is not None:
            raise ValueError("provide validation_data instead of available_data, not both")
        raw_validation_data = (
            validation_data if validation_data is not None else available_data
        )
        parsed_validation_data = (
            _parse_data_design(raw_validation_data, "validation_data")
            if raw_validation_data is not None
            else None
        )
        if parsed_validation_data is not None:
            _require_validation_data_separation(parsed_validation_data)
        project_root = Path(root).resolve()
        if (project_root / STATE_FILE).exists():
            raise FileExistsError(f"project already exists: {project_root / STATE_FILE}")
        project = cls(
            project_root,
            ProjectState(
                schema_version=SCHEMA_VERSION,
                revision=0,
                config=StudyConfig(
                    study_id=study_id,
                    question=question,
                    graph_source=graph_source,
                ),
                evidence_records=parsed_evidence_records,
                discovery_documents=parsed_discovery_documents,
                validation_data=parsed_validation_data,
            ),
        )
        project._commit("project_created", {"study_id": study_id})
        return project

    @classmethod
    def open(cls, root: str | Path) -> TheoryDiscoveryProject:
        project_root = Path(root).resolve()
        state_path = project_root / STATE_FILE
        if not state_path.exists():
            raise FileNotFoundError(f"not a theory-discovery project: {state_path}")
        raw = json.loads(state_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("state.json must contain an object")
        while raw.get("schema_version") != SCHEMA_VERSION:
            if raw.get("schema_version") == 4:
                raw = _migrate_v4_state(raw)
            elif raw.get("schema_version") == 5:
                raw = _migrate_v5_state(raw)
            elif raw.get("schema_version") == 6:
                raw = _migrate_v6_state(raw)
            elif raw.get("schema_version") == 7:
                raw = _migrate_v7_state(raw)
            elif raw.get("schema_version") == 8:
                raw = _migrate_v8_state(raw)
            elif raw.get("schema_version") == 9:
                raw = _migrate_v9_state(raw)
            elif raw.get("schema_version") == 10:
                raw = _migrate_v10_state(raw)
            elif raw.get("schema_version") == 11:
                raw = _migrate_v11_state(raw)
            elif raw.get("schema_version") == 12:
                raw = _migrate_v12_state(raw)
            else:
                break
        state = ProjectState.from_dict(raw)
        if state.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema version {state.schema_version}; expected {SCHEMA_VERSION}"
            )
        return cls(project_root, state)

    def configure_validation_execution(self, auto_execute: bool) -> DataDesign:
        """Opt in or out of automatic execution on the isolated data partition."""
        if not isinstance(auto_execute, bool):
            raise ValueError("auto_execute must be a boolean")
        if self.state.validation_data is None:
            raise ValueError("automatic execution requires validation_data")
        if auto_execute:
            self._require_validation_data_separation()
            self._validation_records_path()
        self.state.validation_data.auto_execute = auto_execute
        self._commit(
            "validation_execution_configured",
            {"auto_execute": auto_execute},
        )
        return self.state.validation_data

    def import_graph(
        self,
        nodes: list[dict[str, object]],
        edges: list[dict[str, object]] | None = None,
    ) -> None:
        if self.state.graph_nodes or self.state.graph_edges:
            raise ValueError("a graph is already imported")
        if not nodes:
            raise ValueError("nodes must not be empty")
        parsed_nodes: list[GraphNode] = []
        node_ids: set[str] = set()
        for item in nodes:
            node_id = _mapping_text(item, "id")
            _require_identifier("graph node ID", node_id)
            if node_id in node_ids:
                raise ValueError(f"duplicate graph node ID: {node_id}")
            node_ids.add(node_id)
            parsed_nodes.append(
                GraphNode(
                    id=node_id,
                    label=_mapping_text(item, "label"),
                    description=_mapping_optional_text(item, "description"),
                    examples=_mapping_string_list(item, "examples"),
                )
            )
        parsed_edges: list[GraphEdge] = []
        edge_ids: set[str] = set()
        for item in edges or []:
            source = _mapping_text(item, "source_node_id")
            target = _mapping_text(item, "target_node_id")
            missing = sorted({source, target} - node_ids)
            if missing:
                raise ValueError(f"graph edge references unknown node IDs: {', '.join(missing)}")
            edge_id = _mapping_optional_text(item, "id") or _new_id("edge")
            _require_identifier("graph edge ID", edge_id)
            if edge_id in edge_ids or edge_id in node_ids:
                raise ValueError(f"duplicate graph item ID: {edge_id}")
            edge_ids.add(edge_id)
            parsed_edges.append(
                GraphEdge(
                    id=edge_id,
                    source_node_id=source,
                    target_node_id=target,
                    label=_mapping_text(item, "label"),
                )
            )
        self.state.graph_nodes.extend(parsed_nodes)
        self.state.graph_edges.extend(parsed_edges)
        self._commit("graph_imported", {"node_count": len(parsed_nodes), "edge_count": len(parsed_edges)})

    def add_relationship(
        self,
        source_node_id: str,
        target_node_id: str,
        relation_type: RelationType,
        direction: RelationDirection,
        rationale: str,
        *,
        relationship_id: str | None = None,
    ) -> RelationshipCandidate:
        node_ids = {node.id for node in self.state.graph_nodes}
        missing = sorted({source_node_id, target_node_id} - node_ids)
        if missing:
            raise ValueError(f"relationship references unknown node IDs: {', '.join(missing)}")
        if source_node_id == target_node_id:
            raise ValueError("a relationship must connect two different nodes")
        if relation_type not in {"association", "sequence", "mechanism", "moderation"}:
            raise ValueError("invalid relation_type")
        if direction not in {"positive", "negative", "conditional", "unspecified"}:
            raise ValueError("invalid direction")
        _require_text("rationale", rationale)
        relationship = RelationshipCandidate(
            id=relationship_id or _new_id("relationship"),
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            relation_type=relation_type,
            direction=direction,
            rationale=rationale,
        )
        _require_identifier("relationship ID", relationship.id)
        self._require_unique_id(relationship.id)
        self.state.relationships.append(relationship)
        self._commit("relationship_added", {"relationship_id": relationship.id})
        return relationship

    def add_theory(
        self,
        name: str,
        description: str,
        theory_statements: list[dict[str, object]],
        conflicting_evidence: list[dict[str, object]],
        unaccounted_evidence: list[dict[str, object]],
        new_predictions_likely: list[str],
        new_predictions_unknown: list[str],
        negative_experiments: list[str],
        *,
        theory_id: str | None = None,
    ) -> Theory:
        _require_text("name", name)
        _require_text("description", description)
        if not theory_statements:
            raise ValueError("theory_statements must not be empty")
        parsed_statements: list[TheoryStatement] = []
        for item in theory_statements:
            statement_evidence = [_parse_evidence(entry) for entry in _required_object_list(item, "evidence")]
            if not statement_evidence:
                raise ValueError("a theory statement must cite at least one evidence record")
            parsed_statements.append(
                TheoryStatement(
                    id=_mapping_optional_text(item, "id") or _new_id("law"),
                    law=_mapping_text(item, "law"),
                    scope=_mapping_text(item, "scope"),
                    evidence=statement_evidence,
                )
            )
        parsed_conflicting_evidence = [
            _parse_evidence(item)
            for item in _validated_object_list("conflicting_evidence", conflicting_evidence)
        ]
        parsed_unaccounted_evidence = [
            _parse_evidence(item)
            for item in _validated_object_list("unaccounted_evidence", unaccounted_evidence)
        ]
        all_evidence = [
            *(evidence for statement in parsed_statements for evidence in statement.evidence),
            *parsed_conflicting_evidence,
            *parsed_unaccounted_evidence,
        ]
        self._validate_evidence_references(all_evidence)
        likely_predictions = _validated_string_list(
            "new_predictions_likely", new_predictions_likely, allow_empty=True
        )
        unknown_predictions = _validated_string_list(
            "new_predictions_unknown", new_predictions_unknown, allow_empty=True
        )
        negative_experiment_list = _validated_string_list(
            "negative_experiments", negative_experiments, allow_empty=True
        )
        self._require_supported_numbers("description", description, all_evidence)
        for statement in parsed_statements:
            self._require_supported_numbers("law", statement.law, statement.evidence)
            self._require_supported_numbers("scope", statement.scope, statement.evidence)
        for field_name, value in {
            "new_predictions_likely": likely_predictions,
            "new_predictions_unknown": unknown_predictions,
            "negative_experiments": negative_experiment_list,
        }.items():
            for text in value:
                self._require_supported_numbers(field_name, text, all_evidence)
        resolved_theory_id = theory_id or _new_id("theory")
        proposed_ids = [resolved_theory_id, *(statement.id for statement in parsed_statements)]
        if len(proposed_ids) != len(set(proposed_ids)):
            raise ValueError("theory and theory statement IDs must be unique")
        for item_id in proposed_ids:
            _require_identifier("theory or law ID", item_id)
            self._require_unique_id(item_id)
        theory = Theory(
            id=resolved_theory_id,
            name=name,
            description=description,
            theory_statements=parsed_statements,
            conflicting_evidence=parsed_conflicting_evidence,
            unaccounted_evidence=parsed_unaccounted_evidence,
            new_predictions_likely=likely_predictions,
            new_predictions_unknown=unknown_predictions,
            negative_experiments=negative_experiment_list,
        )
        self.state.theories.append(theory)
        self._commit("theory_added", {"theory_id": theory.id})
        return theory

    def add_prediction(
        self,
        theory_id: str,
        law_id: str,
        specific_prediction: str,
        operational_signals: list[str],
        strong_test_requirement: str,
        support_criteria: str,
        contradiction_criteria: str,
        *,
        prediction_id: str | None = None,
    ) -> Prediction:
        theory = cast(Theory, _find_by_id(self.state.theories, theory_id, "theory"))
        statement = next(
            (item for item in theory.theory_statements if item.id == law_id), None
        )
        if statement is None:
            raise ValueError(f"law {law_id} does not belong to theory {theory_id}")
        for field_name, value in {
            "specific_prediction": specific_prediction,
            "strong_test_requirement": strong_test_requirement,
            "support_criteria": support_criteria,
            "contradiction_criteria": contradiction_criteria,
        }.items():
            _require_text(field_name, value)
            _reject_forbidden_analysis_content(field_name, value)
            self._require_supported_numbers(field_name, value, statement.evidence)
        parsed_signals = _validated_string_list("operational_signals", operational_signals)
        for signal in parsed_signals:
            _reject_forbidden_analysis_content("operational_signals", signal)
            self._require_supported_numbers("operational_signals", signal, statement.evidence)
        prediction = Prediction(
            id=prediction_id or _new_id("prediction"),
            theory_id=theory_id,
            law_id=law_id,
            specific_prediction=specific_prediction,
            operational_signals=parsed_signals,
            strong_test_requirement=strong_test_requirement,
            support_criteria=support_criteria,
            contradiction_criteria=contradiction_criteria,
        )
        _require_identifier("prediction ID", prediction.id)
        self._require_unique_id(prediction.id)
        self.state.predictions.append(prediction)
        self._commit("prediction_added", {"prediction_id": prediction.id, "theory_id": theory_id})
        return prediction

    def revise_prediction_before_measurement(
        self,
        prediction_id: str,
        specific_prediction: str,
        operational_signals: list[str],
        strong_test_requirement: str,
        support_criteria: str,
        contradiction_criteria: str,
    ) -> Prediction:
        """Revise a prediction and invalidate only its unused contract.

        This is allowed solely before held-out labels exist.  It retains the
        prior falsification contract for audit while ensuring a revised theory
        statement cannot be executed through an old frozen compiler contract.
        """
        if self.state.measurement_runs:
            raise ValueError("cannot revise a prediction after held-out measurement")
        prediction = cast(
            Prediction, _find_by_id(self.state.predictions, prediction_id, "prediction")
        )
        theory = cast(
            Theory, _find_by_id(self.state.theories, prediction.theory_id, "theory")
        )
        statement = next(
            (item for item in theory.theory_statements if item.id == prediction.law_id),
            None,
        )
        if statement is None:
            raise RuntimeError("prediction law is absent from its theory")
        for field_name, value in {
            "specific_prediction": specific_prediction,
            "strong_test_requirement": strong_test_requirement,
            "support_criteria": support_criteria,
            "contradiction_criteria": contradiction_criteria,
        }.items():
            _require_text(field_name, value)
            _reject_forbidden_analysis_content(field_name, value)
            self._require_supported_numbers(field_name, value, statement.evidence)
        parsed_signals = _validated_string_list("operational_signals", operational_signals)
        for signal in parsed_signals:
            _reject_forbidden_analysis_content("operational_signals", signal)
            self._require_supported_numbers("operational_signals", signal, statement.evidence)
        prediction.specific_prediction = specific_prediction
        prediction.operational_signals = parsed_signals
        prediction.strong_test_requirement = strong_test_requirement
        prediction.support_criteria = support_criteria
        prediction.contradiction_criteria = contradiction_criteria
        active_contract = next(
            (
                item
                for item in self.state.falsification_specifications
                if item.prediction_id == prediction_id
            ),
            None,
        )
        if active_contract is not None:
            self.state.superseded_falsification_specifications.append(
                deepcopy(active_contract)
            )
            self.state.falsification_specifications.remove(active_contract)
            self.state.readiness_assessments = [
                item
                for item in self.state.readiness_assessments
                if not (
                    item.parent_type == "falsification_specification"
                    and item.parent_id == active_contract.id
                )
            ]
        if prediction_id in self.state.falsification_repair_prediction_ids:
            self.state.falsification_repair_prediction_ids.remove(prediction_id)
        self._commit(
            "prediction_revised_prevalidation",
            {
                "prediction_id": prediction_id,
                "invalidated_falsification_specification_id": (
                    active_contract.id if active_contract is not None else ""
                ),
            },
        )
        return prediction

    def add_falsification_test(
        self,
        prediction_id: str,
        test_description: str,
        null_hypothesis: str,
        alternative_hypothesis: str,
        required_data: list[str],
        missing_information: list[str],
        relevance: TestRelevance,
        implementability: TestImplementability,
        *,
        test_id: str | None = None,
    ) -> FalsificationTest:
        _find_by_id(self.state.predictions, prediction_id, "prediction")
        if relevance not in {"direct", "limited", "none"}:
            raise ValueError("invalid test relevance")
        if relevance != "direct":
            raise ValueError("a falsification test must have direct relevance")
        if implementability not in {"ready", "partial", "blocked"}:
            raise ValueError("invalid test implementability")
        for field_name, value in {
            "test_description": test_description,
            "null_hypothesis": null_hypothesis,
            "alternative_hypothesis": alternative_hypothesis,
        }.items():
            _require_text(field_name, value)
            _reject_forbidden_analysis_content(field_name, value)
        parsed_required_data = _validated_string_list(
            "required_data", required_data, allow_empty=True
        )
        if len(parsed_required_data) != len(set(parsed_required_data)):
            raise ValueError("required_data must not repeat a data field")
        parsed_missing_information = _validated_string_list(
            "missing_information", missing_information, allow_empty=True
        )
        if implementability == "ready":
            if not parsed_required_data:
                raise ValueError("an implementable test must require validation data fields")
            if parsed_missing_information:
                raise ValueError("an implementable test cannot list missing information")
            self._require_validation_data_separation()
        elif not parsed_missing_information:
            raise ValueError("a partial or blocked test must state missing_information")
        self._validate_required_data_fields(parsed_required_data)

        existing = next(
            (item for item in self.state.falsification_tests if item.prediction_id == prediction_id),
            None,
        )
        if existing is not None:
            if test_id and test_id != existing.id:
                raise ValueError(
                    f"prediction {prediction_id} already has falsification test {existing.id}"
                )
            existing.test_description = test_description
            existing.null_hypothesis = null_hypothesis
            existing.alternative_hypothesis = alternative_hypothesis
            existing.required_data = parsed_required_data
            existing.missing_information = parsed_missing_information
            existing.relevance = relevance
            existing.implementability = implementability
            self._commit(
                "falsification_test_updated",
                {"test_id": existing.id, "prediction_id": prediction_id},
            )
            return existing

        duplicate = next(
            (
                item
                for item in self.state.falsification_tests
                if _normalized_text(item.test_description)
                == _normalized_text(test_description)
            ),
            None,
        )
        if duplicate is not None:
            raise ValueError(f"duplicate falsification test: {duplicate.id}")
        test = FalsificationTest(
            id=test_id or _new_id("falsification_test"),
            prediction_id=prediction_id,
            test_description=test_description,
            null_hypothesis=null_hypothesis,
            alternative_hypothesis=alternative_hypothesis,
            required_data=parsed_required_data,
            missing_information=parsed_missing_information,
            relevance=relevance,
            implementability=implementability,
        )
        _require_identifier("falsification test ID", test.id)
        self._require_unique_id(test.id)
        self.state.falsification_tests.append(test)
        self._commit(
            "falsification_test_added",
            {"test_id": test.id, "prediction_id": prediction_id},
        )
        return test

    def add_falsification_specification(
        self,
        prediction_id: str,
        template_id: str,
        measurable_implication: dict[str, str],
        measurement: dict[str, str],
        population: dict[str, object],
        comparison: dict[str, str],
        hypotheses: dict[str, str],
        decision_rule: dict[str, str],
        required_fields: list[str],
        unresolved: list[str],
        readiness: str,
        *,
        specification_id: str | None = None,
        expressions: dict[str, dict[str, object]] | None = None,
        missing_measurement_constructs: list[str] | None = None,
    ) -> FalsificationSpecification:
        """Bind a canonical test template to already-frozen questions.

        The compiler may choose a template and bind its variables to question
        IDs, but it cannot invent a question, edit a frozen question, or alter
        the template's comparison and decision semantics.
        """
        _find_by_id(self.state.predictions, prediction_id, "prediction")
        template = FALSIFICATION_TEMPLATES.get(template_id)
        if template is None:
            raise ValueError(
                "unknown falsification template; allowed templates are: "
                + ", ".join(sorted(FALSIFICATION_TEMPLATES))
            )
        if readiness not in {"ready", "not_ready"}:
            raise ValueError("specification readiness must be ready or not_ready")
        parsed_measurement = _mapping_string_dict_value(measurement, "measurement")
        parsed_expressions = _validated_expression_dict(
            "expressions", {} if expressions is None else expressions
        )
        parsed_unresolved = _validated_string_list(
            "unresolved", unresolved, allow_empty=True
        )
        parsed_missing_constructs = _validated_string_list(
            "missing_measurement_constructs",
            [] if missing_measurement_constructs is None else missing_measurement_constructs,
            allow_empty=True,
        )
        if len(parsed_missing_constructs) != len(set(parsed_missing_constructs)):
            raise ValueError("missing_measurement_constructs must not repeat a construct ID")
        for construct_id in parsed_missing_constructs:
            _require_identifier("missing measurement construct ID", construct_id)
        measurement_mode = cast(str, template["measurement_mode"])
        required_keys = set(cast(tuple[str, ...], template["measurement_keys"]))
        required_expression_keys = set(
            cast(tuple[str, ...], template["expression_keys"])
        )
        compiler_failures: list[str] = []
        frozen_construct_ids = {
            item.construct_id for item in self.state.measurement_specifications
        }
        for construct_id in parsed_missing_constructs:
            if construct_id in frozen_construct_ids:
                compiler_failures.append(
                    "missing_measurement_construct_already_frozen:" + construct_id
                )
        if parsed_missing_constructs and not parsed_unresolved:
            compiler_failures.append(
                "missing_measurement_construct_requires_a_concrete_unresolved_diagnosis"
            )
        if measurement_mode == "fixed_variables":
            unexpected_keys = sorted(set(parsed_measurement) - required_keys)
            if unexpected_keys:
                raise ValueError(
                    "measurement names are not valid for the selected template: "
                    + ", ".join(unexpected_keys)
                )
            if parsed_expressions:
                compiler_failures.append("expressions_not_permitted_for_template")
            for variable in sorted(required_keys - set(parsed_measurement)):
                compiler_failures.append(
                    f"missing_measurement_specification:{variable}"
                )
        elif measurement_mode == "expression_aliases":
            for variable in sorted(required_expression_keys - set(parsed_expressions)):
                compiler_failures.append(f"missing_frozen_expression:{variable}")
            unexpected_expressions = sorted(
                set(parsed_expressions) - required_expression_keys
            )
            if unexpected_expressions:
                compiler_failures.append(
                    "expressions_are_not_valid_for_template:"
                    + ",".join(unexpected_expressions)
                )
        else:
            raise RuntimeError("unknown measurement mode in falsification template")
        self._validate_canonical_specification_fields(
            template_id,
            measurable_implication,
            population,
            comparison,
            hypotheses,
            decision_rule,
        )

        resolved_questions: dict[str, MeasurementSpecification] = {}
        for variable, measurement_id in sorted(parsed_measurement.items()):
            try:
                resolved_questions[variable] = self._resolve_measurement_specification(
                    measurement_id
                )
            except ValueError as error:
                compiler_failures.append(str(error))
        if measurement_mode == "expression_aliases":
            referenced_aliases: set[str] = set()
            for variable in sorted(required_expression_keys & set(parsed_expressions)):
                try:
                    referenced_aliases.update(
                        _expression_measurement_aliases(parsed_expressions[variable])
                    )
                except ValueError as error:
                    compiler_failures.append(f"invalid_expression:{variable}:{error}")
            for alias in sorted(referenced_aliases - set(parsed_measurement)):
                compiler_failures.append(f"missing_measurement_binding:{alias}")
            for alias in sorted(set(parsed_measurement) - referenced_aliases):
                compiler_failures.append(f"unused_measurement_binding:{alias}")
        expected_fields = self._measurement_required_fields()
        parsed_required_fields = _validated_string_list(
            "required_fields", required_fields, allow_empty=True
        )
        if len(parsed_required_fields) != len(set(parsed_required_fields)):
            raise ValueError("required_fields must not repeat a data field")
        if sorted(parsed_required_fields) != expected_fields:
            compiler_failures.append("required_fields_do_not_match_frozen_measurement_protocol")
        if measurement_mode == "fixed_variables" and len(resolved_questions) == len(required_keys):
            duplicate_constructs = len(
                {item.construct_id for item in resolved_questions.values()}
            ) != len(resolved_questions)
            if duplicate_constructs:
                compiler_failures.append("condition_and_outcome_must_use_distinct_constructs")
        try:
            self._require_validation_data_separation()
        except ValueError as error:
            compiler_failures.append("validation_partition_not_isolated:" + str(error))

        all_unresolved = list(dict.fromkeys([*parsed_unresolved, *compiler_failures]))
        computed_readiness = "ready" if not all_unresolved else "not_ready"
        if readiness != computed_readiness:
            raise ValueError(
                "specification readiness does not match the deterministic verifier: "
                + computed_readiness
            )
        if computed_readiness == "ready" and parsed_unresolved:
            raise ValueError("a ready specification cannot list unresolved items")

        canonical = self._canonical_specification_fields(template_id)
        candidate = FalsificationSpecification(
            id=specification_id or _new_id("falsification_specification"),
            prediction_id=prediction_id,
            template_id=template_id,
            measurable_implication=cast(dict[str, str], canonical["measurable_implication"]),
            measurement=parsed_measurement,
            expressions=parsed_expressions,
            population=cast(dict[str, object], canonical["population"]),
            comparison=cast(dict[str, str], canonical["comparison"]),
            hypotheses=cast(dict[str, str], canonical["hypotheses"]),
            decision_rule=cast(dict[str, str], canonical["decision_rule"]),
            measurement_protocol="frozen_llm_question_v1",
            measurement_specification_snapshot=deepcopy(resolved_questions),
            required_fields=expected_fields,
            unresolved=all_unresolved,
            missing_measurement_constructs=parsed_missing_constructs,
            readiness=cast(str, computed_readiness),
        )
        _require_identifier("falsification specification ID", candidate.id)
        candidate.specification_hash = self._specification_hash(candidate)

        existing = next(
            (
                item
                for item in self.state.falsification_specifications
                if item.prediction_id == prediction_id
            ),
            None,
        )
        if existing is not None:
            planned_recompile = (
                prediction_id in self.state.falsification_repair_prediction_ids
            )
            if specification_id and specification_id != existing.id and not planned_recompile:
                raise ValueError(
                    f"prediction {prediction_id} already has specification {existing.id}"
                )
            if planned_recompile:
                if self.state.measurement_runs:
                    raise ValueError(
                        "cannot repair a falsification contract after held-out measurement"
                    )
                self.state.superseded_falsification_specifications.append(
                    deepcopy(existing)
                )
            candidate.id = existing.id
            candidate.created_at = existing.created_at
            candidate.specification_hash = self._specification_hash(candidate)
            index = self.state.falsification_specifications.index(existing)
            self.state.falsification_specifications[index] = candidate
            self._upsert_specification_readiness(candidate)
            if planned_recompile:
                self.state.falsification_repair_prediction_ids.remove(prediction_id)
            self._commit(
                "falsification_specification_recompiled",
                {
                    "specification_id": candidate.id,
                    "prediction_id": prediction_id,
                    "readiness": candidate.readiness,
                    "missing_measurement_constructs": candidate.missing_measurement_constructs,
                },
            )
            return candidate

        self._require_unique_id(candidate.id)
        self.state.falsification_specifications.append(candidate)
        self._upsert_specification_readiness(candidate)
        self._commit(
            "falsification_specification_compiled",
            {
                "specification_id": candidate.id,
                "prediction_id": prediction_id,
                "readiness": candidate.readiness,
                "missing_measurement_constructs": candidate.missing_measurement_constructs,
            },
        )
        return candidate

    def add_measurement_specification(
        self,
        construct_id: str,
        measurement_question: str,
        source_prediction_ids: list[str],
        *,
        specification_id: str | None = None,
    ) -> MeasurementSpecification:
        """Freeze one construct-to-question mapping before held-out access."""
        _require_identifier("construct ID", construct_id)
        _require_text("measurement_question", measurement_question)
        normalized_question = " ".join(measurement_question.split())
        if len(normalized_question) < 12:
            raise ValueError("measurement_question must state an observable coding question")
        parsed_prediction_ids = _validated_string_list(
            "source_prediction_ids", source_prediction_ids, allow_empty=False
        )
        if len(parsed_prediction_ids) != len(set(parsed_prediction_ids)):
            raise ValueError("source_prediction_ids must not repeat a prediction ID")
        for prediction_id in parsed_prediction_ids:
            _find_by_id(self.state.predictions, prediction_id, "prediction")
        candidate = MeasurementSpecification(
            id=specification_id or _new_id("measurement_specification"),
            construct_id=construct_id,
            measurement_question=normalized_question,
            source_prediction_ids=sorted(parsed_prediction_ids),
            **deepcopy(FROZEN_MEASUREMENT_PROTOCOL),
        )
        _require_identifier("measurement specification ID", candidate.id)
        candidate.specification_hash = self._measurement_specification_hash(candidate)
        existing = next(
            (
                item
                for item in self.state.measurement_specifications
                if item.id == candidate.id or item.construct_id == construct_id
            ),
            None,
        )
        if existing is not None:
            if existing.specification_hash != candidate.specification_hash:
                raise ValueError(
                    "measurement specifications are frozen; create no replacement after "
                    "generation has been accepted"
                )
            return existing
        self._require_unique_id(candidate.id)
        self.state.measurement_specifications.append(candidate)
        self._upsert_measurement_readiness(candidate)
        self._commit(
            "measurement_specification_frozen",
            {
                "measurement_specification_id": candidate.id,
                "construct_id": candidate.construct_id,
                "specification_hash": candidate.specification_hash,
            },
        )
        return candidate

    def _measurement_specification_hash(
        self, specification: MeasurementSpecification
    ) -> str:
        frozen = {
            "id": specification.id,
            "construct_id": specification.construct_id,
            "measurement_question": specification.measurement_question,
            "source_prediction_ids": specification.source_prediction_ids,
            "labels": specification.labels,
            "model_id": specification.model_id,
            "model_revision": specification.model_revision,
            "temperature": specification.temperature,
            "top_p": specification.top_p,
            "max_tokens": specification.max_tokens,
            "prompt_version": specification.prompt_version,
            "uncertain_handling": specification.uncertain_handling,
        }
        encoded = json.dumps(frozen, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return sha256(encoded.encode("utf-8")).hexdigest()

    def _upsert_measurement_readiness(
        self, specification: MeasurementSpecification
    ) -> None:
        expected_protocol = FROZEN_MEASUREMENT_PROTOCOL
        complete = (
            specification.labels == expected_protocol["labels"]
            and specification.model_id == expected_protocol["model_id"]
            and specification.model_revision == expected_protocol["model_revision"]
            and specification.temperature == expected_protocol["temperature"]
            and specification.top_p == expected_protocol["top_p"]
            and specification.max_tokens == expected_protocol["max_tokens"]
            and specification.prompt_version == expected_protocol["prompt_version"]
            and specification.uncertain_handling == expected_protocol["uncertain_handling"]
            and bool(specification.measurement_question.strip())
        )
        status: ReadinessStatus = "ready" if complete else "blocked"
        note = (
            "Question, labels, model revision, decoding, and uncertain handling are frozen before held-out measurement."
            if complete
            else "The measurement contract is incomplete and cannot be frozen."
        )
        existing = next(
            (
                item
                for item in self.state.readiness_assessments
                if (item.parent_type, item.parent_id, item.field_key)
                == ("measurement", specification.id, "frozen_question")
            ),
            None,
        )
        if existing is not None:
            existing.status = status
            existing.review_note = note
            existing.indicator_available = complete
            return
        assessment = ReadinessAssessment(
            id=_new_id("assessment"),
            parent_type="measurement",
            parent_id=specification.id,
            field_key="frozen_question",
            status=status,
            review_note=note,
            indicator_available=complete,
        )
        self._require_unique_id(assessment.id)
        self.state.readiness_assessments.append(assessment)

    def _canonical_specification_fields(
        self, template_id: str
    ) -> dict[str, object]:
        template = FALSIFICATION_TEMPLATES[template_id]
        return {
            "measurable_implication": {
                "condition": "condition",
                "outcome": "outcome",
                "relation": cast(str, template["relation"]),
            },
            "population": {"include": "all_valid_records", "exclude": []},
            "comparison": deepcopy(cast(dict[str, str], template["comparison"])),
            "hypotheses": deepcopy(cast(dict[str, str], template["hypotheses"])),
            "decision_rule": deepcopy(cast(dict[str, str], template["decision_rule"])),
        }

    def _validate_canonical_specification_fields(
        self,
        template_id: str,
        measurable_implication: dict[str, str],
        population: dict[str, object],
        comparison: dict[str, str],
        hypotheses: dict[str, str],
        decision_rule: dict[str, str],
    ) -> None:
        canonical = self._canonical_specification_fields(template_id)
        supplied_string_fields = {
            "measurable_implication": measurable_implication,
            "comparison": comparison,
            "hypotheses": hypotheses,
            "decision_rule": decision_rule,
        }
        for name, supplied in supplied_string_fields.items():
            parsed = _mapping_string_dict_value(supplied, name)
            if parsed != canonical[name]:
                raise ValueError(
                    f"{name} must exactly match the frozen {template_id} template"
                )
        if population != canonical["population"]:
            raise ValueError(
                "population must be the frozen all_valid_records/no_exclusions contract"
            )

    def _resolve_measurement_rule(self, rule_id: str) -> MeasurementRule:
        if rule_id.startswith("field_truthy:"):
            source_field = rule_id.removeprefix("field_truthy:")
            _require_identifier("field_truthy source field", source_field)
            self._validate_required_data_fields([source_field])
            return MeasurementRule(
                id=rule_id,
                source_field=source_field,
                kind="field_truthy",
            )
        if self.state.validation_data is None:
            raise ValueError("unknown_measurement_rule:" + rule_id)
        matching = [
            rule
            for rule in self.state.validation_data.measurement_rules
            if rule.id == rule_id
        ]
        if not matching:
            raise ValueError("unknown_measurement_rule:" + rule_id)
        return matching[0]

    def _resolve_measurement_specification(
        self, specification_id: str
    ) -> MeasurementSpecification:
        matching = [
            item
            for item in self.state.measurement_specifications
            if item.id == specification_id
        ]
        if not matching:
            raise ValueError("unknown_measurement_specification:" + specification_id)
        specification = matching[0]
        if specification.specification_hash != self._measurement_specification_hash(
            specification
        ):
            raise ValueError("measurement_specification_hash_mismatch:" + specification_id)
        return specification

    def _measurement_required_fields(self) -> list[str]:
        if self.state.validation_data is None:
            return []
        available = set(self.state.validation_data.required_data_fields)
        text_field = next(
            (
                field
                for field in ("text_review", "text", "content", "document")
                if field in available
            ),
            None,
        )
        if text_field is None or "record_id" not in available:
            return []
        return ["record_id", text_field]

    def _specification_hash(self, specification: FalsificationSpecification) -> str:
        frozen = {
            "id": specification.id,
            "prediction_id": specification.prediction_id,
            "template_id": specification.template_id,
            "measurable_implication": specification.measurable_implication,
            "measurement": specification.measurement,
            "expressions": specification.expressions,
            "measurement_rule_snapshot": {
                name: asdict(rule)
                for name, rule in specification.measurement_rule_snapshot.items()
            },
            "measurement_protocol": specification.measurement_protocol,
            "measurement_specification_snapshot": {
                name: {
                    **asdict(question),
                    "specification_hash": question.specification_hash,
                }
                for name, question in specification.measurement_specification_snapshot.items()
            },
            "population": specification.population,
            "comparison": specification.comparison,
            "hypotheses": specification.hypotheses,
            "decision_rule": specification.decision_rule,
            "required_fields": specification.required_fields,
            "missing_measurement_constructs": specification.missing_measurement_constructs,
            "readiness": specification.readiness,
        }
        encoded = json.dumps(frozen, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return sha256(encoded.encode("utf-8")).hexdigest()

    def _upsert_specification_readiness(
        self, specification: FalsificationSpecification
    ) -> None:
        status: ReadinessStatus = (
            "ready" if specification.readiness == "ready" else "blocked"
        )
        note = (
            "A deterministic executor can run this frozen specification from the "
            "permitted validation fields without a new scientific, measurement, or "
            "decision judgement."
            if status == "ready"
            else "The frozen specification is not executable without new judgement: "
            + "; ".join(specification.unresolved)
        )
        existing = next(
            (
                item
                for item in self.state.readiness_assessments
                if (
                    item.parent_type,
                    item.parent_id,
                    item.field_key,
                )
                == ("falsification_specification", specification.id, "execution_contract")
            ),
            None,
        )
        if existing is not None:
            existing.status = status
            existing.review_note = note
            existing.indicator_available = specification.readiness == "ready"
            existing.evidence_needed = ""
            existing.missing_information = list(specification.unresolved)
            return
        assessment = ReadinessAssessment(
            id=_new_id("assessment"),
            parent_type="falsification_specification",
            parent_id=specification.id,
            field_key="execution_contract",
            status=status,
            review_note=note,
            indicator_available=specification.readiness == "ready",
            missing_information=list(specification.unresolved),
        )
        self._require_unique_id(assessment.id)
        self.state.readiness_assessments.append(assessment)

    def add_measurement_plan(
        self,
        prediction_id: str,
        indicators: dict[str, str],
        operationalization: dict[str, str],
        data_design: dict[str, object],
        *,
        missing_information: list[str] | None = None,
        measurement_id: str | None = None,
    ) -> MeasurementPlan:
        _find_by_id(self.state.predictions, prediction_id, "prediction")
        parsed_indicators = _mapping_string_dict_value(indicators, "indicators")
        parsed_operationalization = _mapping_string_dict_value(
            operationalization, "operationalization"
        )
        if not parsed_indicators:
            raise ValueError("indicators must not be empty")
        if not parsed_operationalization:
            raise ValueError("operationalization must not be empty")
        parsed_data_design = DataDesign(
            unit_of_analysis=_mapping_text(data_design, "unit_of_analysis"),
            population=_mapping_text(data_design, "population"),
            data_source=_mapping_text(data_design, "data_source"),
            time_window=_mapping_text(data_design, "time_window"),
            required_data_fields=_mapping_string_list(data_design, "required_data_fields"),
        )
        if not parsed_data_design.required_data_fields:
            raise ValueError("required_data_fields must not be empty")
        existing_plan = next(
            (item for item in self.state.measurement_plans if item.prediction_id == prediction_id),
            None,
        )
        if existing_plan is not None:
            if measurement_id and measurement_id != existing_plan.id:
                raise ValueError(f"prediction {prediction_id} already has measurement {existing_plan.id}")
            existing_plan.indicators = parsed_indicators
            existing_plan.operationalization = parsed_operationalization
            existing_plan.data_design = parsed_data_design
            existing_plan.missing_information = _validated_string_list(
                "missing_information", missing_information or [], allow_empty=True
            )
            self._commit(
                "measurement_plan_updated",
                {"measurement_id": existing_plan.id, "prediction_id": prediction_id},
            )
            return existing_plan
        plan = MeasurementPlan(
            id=measurement_id or _new_id("measurement"),
            prediction_id=prediction_id,
            indicators=parsed_indicators,
            operationalization=parsed_operationalization,
            data_design=parsed_data_design,
            missing_information=_validated_string_list(
                "missing_information", missing_information or [], allow_empty=True
            ),
        )
        self._require_unique_id(plan.id)
        self.state.measurement_plans.append(plan)
        self._commit("measurement_plan_added", {"measurement_id": plan.id, "prediction_id": prediction_id})
        return plan

    def add_readiness_assessment(
        self,
        parent_type: ReadinessParentType,
        parent_id: str,
        field_key: str,
        status: ReadinessStatus,
        review_note: str,
        *,
        indicator_available: bool | None = None,
        evidence_needed: str = "",
        missing_information: list[str] | None = None,
        assessment_id: str | None = None,
    ) -> ReadinessAssessment:
        if parent_type not in {
            "theory",
            "prediction",
            "measurement",
            "falsification_specification",
        }:
            raise ValueError("invalid readiness parent_type")
        if parent_type == "theory":
            _find_by_id(self.state.theories, parent_id, "theory")
            allowed_fields = THEORY_FIELDS
        elif parent_type == "prediction":
            _find_by_id(self.state.predictions, parent_id, "prediction")
            allowed_fields = PREDICTION_FIELDS
        elif parent_type == "measurement":
            _find_by_id(self.state.measurement_plans, parent_id, "measurement")
            allowed_fields = MEASUREMENT_FIELDS
        else:
            _find_by_id(
                self.state.falsification_specifications,
                parent_id,
                "falsification specification",
            )
            allowed_fields = SPECIFICATION_FIELDS
        if field_key not in allowed_fields:
            raise ValueError(f"invalid field_key {field_key!r} for {parent_type}")
        if status not in {"missing", "draft", "ready", "needs_revision", "blocked"}:
            raise ValueError("invalid readiness status")
        _require_text("review_note", review_note)
        if indicator_available is not None and not isinstance(indicator_available, bool):
            raise ValueError("indicator_available must be a boolean or null")
        if not isinstance(evidence_needed, str):
            raise ValueError("evidence_needed must be a string")
        existing_assessment = next(
            (
                item
                for item in self.state.readiness_assessments
                if (item.parent_type, item.parent_id, item.field_key)
                == (parent_type, parent_id, field_key)
            ),
            None,
        )
        if existing_assessment is not None:
            existing_assessment.status = status
            existing_assessment.review_note = review_note
            existing_assessment.indicator_available = indicator_available
            existing_assessment.evidence_needed = evidence_needed
            existing_assessment.missing_information = _validated_string_list(
                "missing_information", missing_information or [], allow_empty=True
            )
            self._commit(
                "readiness_reassessed",
                {"assessment_id": existing_assessment.id, "parent_type": parent_type, "field_key": field_key},
            )
            return existing_assessment
        assessment = ReadinessAssessment(
            id=assessment_id or _new_id("assessment"),
            parent_type=parent_type,
            parent_id=parent_id,
            field_key=field_key,
            status=status,
            review_note=review_note,
            indicator_available=indicator_available,
            evidence_needed=evidence_needed,
            missing_information=_validated_string_list(
                "missing_information", missing_information or [], allow_empty=True
            ),
        )
        self._require_unique_id(assessment.id)
        self.state.readiness_assessments.append(assessment)
        self._commit(
            "readiness_assessed",
            {"assessment_id": assessment.id, "parent_type": parent_type, "field_key": field_key},
        )
        return assessment

    def record_validation(
        self,
        prediction_id: str,
        status: ValidationStatus,
        summary: str,
        evidence: list[dict[str, object]] | None = None,
        *,
        validation_id: str | None = None,
        execution_artifact: str = "",
        falsification_specification_id: str = "",
        specification_hash: str = "",
    ) -> ValidationResult:
        _find_by_id(self.state.predictions, prediction_id, "prediction")
        allowed_statuses = {
            "support",
            "contradict",
            "no_evidence",
            "prediction_consistent",
            "observed_directional_contradiction",
            "inconclusive",
            "not_testable",
        }
        if status not in allowed_statuses:
            raise ValueError("invalid validation status")
        _require_text("summary", summary)
        result_semantics = "legacy_v1"
        if falsification_specification_id:
            specification = cast(
                FalsificationSpecification,
                _find_by_id(
                    self.state.falsification_specifications,
                    falsification_specification_id,
                    "falsification specification",
                ),
            )
            if specification.prediction_id != prediction_id:
                raise ValueError(
                    "validation specification does not belong to its prediction"
                )
            if specification.readiness != "ready":
                raise ValueError("cannot execute a falsification specification that is not ready")
            if specification_hash != specification.specification_hash:
                raise ValueError("validation result does not match the frozen specification hash")
            if status in {"support", "contradict", "no_evidence"}:
                raise ValueError(
                    "compiler-backed validation must use the v1 result semantics"
                )
            result_semantics = "falsification_compiler_v1"
        else:
            if specification_hash:
                raise ValueError("a specification hash requires a specification ID")
            if status not in {"support", "contradict", "no_evidence"}:
                raise ValueError(
                    "new validation semantics require a frozen specification ID and hash"
                )
            if prediction_id not in {
                item.prediction_id for item in self.state.falsification_tests
            }:
                raise ValueError("a prediction needs a falsification test before legacy validation")
        if any(
            item.prediction_id == prediction_id
            and item.falsification_specification_id == falsification_specification_id
            for item in self.state.validation_results
        ):
            raise ValueError("a validation result already exists for this frozen specification")
        if execution_artifact:
            self._validate_execution_artifact(prediction_id, execution_artifact)
        result = ValidationResult(
            id=validation_id or _new_id("validation"),
            prediction_id=prediction_id,
            status=status,
            summary=summary,
            evidence=[_parse_evidence(item) for item in evidence or []],
            execution_artifact=execution_artifact,
            falsification_specification_id=falsification_specification_id,
            specification_hash=specification_hash,
            result_semantics=result_semantics,
        )
        self._require_unique_id(result.id)
        self.state.validation_results.append(result)
        self._commit("validation_recorded", {"validation_id": result.id, "status": status})
        return result

    def _measurement_repair_requests(self) -> list[dict[str, object]]:
        """Return the compiler-diagnosed atomic gaps eligible for one repair.

        This deliberately refuses to infer a construct from prose.  A repair
        is possible only when every blocked contract provides the compiler's
        structured atomic construct IDs, no held-out label has been produced,
        and the bounded repair budget remains.
        """
        if (
            self.state.measurement_runs
            or self.state.measurement_repair_rounds >= MAX_MEASUREMENT_REPAIR_ROUNDS
        ):
            return []
        blocked = [
            item
            for item in self.state.falsification_specifications
            if item.readiness != "ready"
        ]
        if not blocked or any(not item.missing_measurement_constructs for item in blocked):
            return []
        existing_construct_ids = {
            item.construct_id for item in self.state.measurement_specifications
        }
        requests: dict[str, dict[str, object]] = {}
        for specification in blocked:
            for construct_id in specification.missing_measurement_constructs:
                if construct_id in existing_construct_ids:
                    return []
                request = requests.setdefault(
                    construct_id,
                    {
                        "construct_id": construct_id,
                        "prediction_ids": [],
                        "specification_ids": [],
                        "diagnoses": [],
                    },
                )
                cast(list[str], request["prediction_ids"]).append(
                    specification.prediction_id
                )
                cast(list[str], request["specification_ids"]).append(
                    specification.id
                )
                cast(list[str], request["diagnoses"]).extend(specification.unresolved)
        for request in requests.values():
            request["prediction_ids"] = sorted(set(cast(list[str], request["prediction_ids"])))
            request["specification_ids"] = sorted(
                set(cast(list[str], request["specification_ids"]))
            )
            request["diagnoses"] = list(
                dict.fromkeys(cast(list[str], request["diagnoses"]))
            )
        return [requests[key] for key in sorted(requests)]

    def _submit_measurement_repair(self, payload: dict[str, object]) -> None:
        requests = self._measurement_repair_requests()
        if not requests:
            raise ValueError("no compiler-diagnosed measurement repair is pending")
        expected_prediction_ids = {
            cast(str, request["construct_id"]): set(
                cast(list[str], request["prediction_ids"])
            )
            for request in requests
        }
        submitted = _mapping_object_list(payload, "measurement_specifications")
        submitted_construct_ids = {
            _mapping_text(item, "construct_id") for item in submitted
        }
        if len(submitted_construct_ids) != len(submitted):
            raise ValueError("measurement repair must not repeat a construct_id")
        if submitted_construct_ids != set(expected_prediction_ids):
            raise ValueError(
                "measurement repair must generate exactly the compiler-diagnosed constructs: "
                + ", ".join(sorted(expected_prediction_ids))
            )
        for item in submitted:
            construct_id = _mapping_text(item, "construct_id")
            source_prediction_ids = set(
                _required_string_list(item, "source_prediction_ids")
            )
            if source_prediction_ids != expected_prediction_ids[construct_id]:
                raise ValueError(
                    "measurement repair source_prediction_ids must match the "
                    "compiler diagnosis for " + construct_id
                )
        for item in submitted:
            self.add_measurement_specification(
                _mapping_text(item, "construct_id"),
                _mapping_text(item, "measurement_question"),
                _required_string_list(item, "source_prediction_ids"),
                specification_id=_mapping_optional_text(item, "id") or None,
            )
        if self.state.measurement_runs:
            raise RuntimeError("held-out measurement appeared during repair")
        self.state.measurement_repair_rounds += 1
        # A new question changes only the contracts that named that missing
        # construct.  Other already-frozen, unused contracts remain valid and
        # must not be needlessly regenerated.
        self.state.falsification_repair_prediction_ids = sorted(
            {
                prediction_id
                for request in requests
                for prediction_id in cast(list[str], request["prediction_ids"])
            }
        )
        self._commit(
            "measurement_repair_frozen",
            {
                "round": self.state.measurement_repair_rounds,
                "construct_ids": sorted(expected_prediction_ids),
                "recompile_prediction_ids": self.state.falsification_repair_prediction_ids,
            },
        )

    def next_action(self) -> dict[str, object]:
        if not self.state.graph_nodes:
            return {"action": "import_graph", "reason": "no thematic graph is loaded"}
        if not self.state.relationships:
            return {"action": "discover_relationships", "reason": "no candidate relationships are saved"}
        if not self.state.theories:
            return {"action": "generate_theories", "reason": "no theory law has been synthesized"}
        laws_with_predictions = {
            (item.theory_id, item.law_id) for item in self.state.predictions
        }
        missing_laws = [
            {"theory_id": theory.id, "law_id": statement.id}
            for theory in self.state.theories
            for statement in theory.theory_statements
            if (theory.id, statement.id) not in laws_with_predictions
        ]
        if missing_laws:
            return {
                "action": "generate_predictions",
                "reason": "some theory laws do not yet have a falsifiable prediction",
                "laws": missing_laws,
            }
        covered_prediction_ids = {
            prediction_id
            for specification in self.state.measurement_specifications
            for prediction_id in specification.source_prediction_ids
        }
        missing_measurement_predictions = [
            item.id
            for item in self.state.predictions
            if item.id not in covered_prediction_ids
        ]
        if missing_measurement_predictions:
            return {
                "action": "generate_measurement_specifications",
                "reason": (
                    "some predictions have no pre-validation frozen construct "
                    "measurement question"
                ),
                "prediction_ids": missing_measurement_predictions,
            }
        if self.state.falsification_repair_prediction_ids:
            return {
                "action": "compile_falsification_specifications",
                "reason": (
                    "a newly frozen atomic measurement requires its dependent "
                    "falsification contract to be recompiled"
                ),
                "prediction_ids": list(self.state.falsification_repair_prediction_ids),
            }
        missing_falsification_specifications = [
            prediction.id
            for prediction in self.state.predictions
            if prediction.id
            not in {
                item.prediction_id
                for item in self.state.falsification_specifications
            }
        ]
        if missing_falsification_specifications:
            return {
                "action": "compile_falsification_specifications",
                "reason": "some predictions do not yet have a frozen falsification specification",
                "prediction_ids": missing_falsification_specifications,
            }
        repair_requests = self._measurement_repair_requests()
        if repair_requests:
            return {
                "action": "repair_measurement_specifications",
                "reason": (
                    "the compiler diagnosed missing atomic constructs before "
                    "held-out measurement"
                ),
                "repair_round": self.state.measurement_repair_rounds + 1,
                "max_repair_rounds": MAX_MEASUREMENT_REPAIR_ROUNDS,
                "repairs": repair_requests,
            }
        if self.state.validation_data is not None and self.state.validation_data.auto_execute:
            non_ready_specifications = [
                {
                    "id": item.id,
                    "prediction_id": item.prediction_id,
                    "unresolved": item.unresolved,
                }
                for item in self.state.falsification_specifications
                if item.readiness != "ready"
            ]
            if non_ready_specifications:
                return {
                    "action": "validation_execution_blocked",
                    "reason": (
                        "automatic execution requires every frozen falsification "
                        "specification to be ready"
                    ),
                    "specifications": non_ready_specifications,
                }
            try:
                self._validation_records_path()
            except ValueError as error:
                return {
                    "action": "validation_execution_blocked",
                    "reason": str(error),
                }
            pending_measurements = self._pending_measurement_specification_ids()
            if pending_measurements:
                return {
                    "action": "measure_validation_records",
                    "reason": (
                        "a frozen question must label the isolated validation "
                        "records before deterministic falsification execution"
                    ),
                    "measurement_specification_ids": [pending_measurements[0]],
                    "remaining_measurement_count": len(pending_measurements),
                }
            missing_validation_results = self._pending_validation_prediction_ids()
            if missing_validation_results:
                return {
                    "action": "execute_falsification_tests",
                    "reason": "the next ready falsification test requires isolated execution",
                    "prediction_ids": [missing_validation_results[0]],
                    "remaining_prediction_count": len(missing_validation_results),
                }
            return {
                "action": "complete",
                "reason": "all ready falsification tests have an isolated validation result",
            }
        return {
            "action": "ready_for_review",
            "reason": (
                "all theories, predictions, and frozen falsification specifications "
                "are saved"
            ),
        }

    def task_packet(self) -> dict[str, object]:
        next_action = self.next_action()
        action = str(next_action["action"])
        packet: dict[str, object] = {
            "action": action,
            "study": asdict(self.state.config),
            "rule": "Complete only this stage. Preserve the separation between discovery material and held-out validation data.",
            "validation": {
                "id_regex": ID_PATTERN.pattern,
                "evidence_rule": "Evidence IDs and summaries must exactly match supplied evidence records.",
                "reference_rule": "Referenced IDs and required data fields must already exist in the packet.",
                "retry_rule": "If submission returns retry=true, correct the errors and resubmit only this stage.",
            },
        }
        if action == "import_graph":
            packet["prompt"] = _load_prompt("01-import-graph.md")
            packet["expected_output"] = {"nodes": [{"id": "node_id", "label": "node label"}], "edges": []}
        elif action == "discover_relationships":
            packet["graph"] = {
                "nodes": [asdict(item) for item in self.state.graph_nodes],
                "edges": [asdict(item) for item in self.state.graph_edges],
            }
            packet["discovery_documents"] = [
                asdict(item) for item in self.state.discovery_documents
            ]
            packet["prompt"] = _load_prompt("02-discover-relationships.md")
            packet["expected_output"] = {"relationships": [{"source_node_id": "node_id", "target_node_id": "node_id", "relation_type": "association|sequence|mechanism|moderation", "direction": "positive|negative|conditional|unspecified", "rationale": "why it is worth investigating"}]}
        elif action == "generate_theories":
            packet["graph"] = {
                "nodes": [asdict(item) for item in self.state.graph_nodes],
                "edges": [asdict(item) for item in self.state.graph_edges],
            }
            packet["relationships"] = [asdict(item) for item in self.state.relationships]
            packet["discovery_documents"] = [
                asdict(item) for item in self.state.discovery_documents
            ]
            packet["evidence_records"] = [asdict(item) for item in self.state.evidence_records]
            packet["prompt"] = _load_prompt("03-generate-theories.md")
            packet["expected_output"] = {
                "theories": [{
                    "id": "theory_id",
                    "name": "theory name",
                    "description": "what the theory explains and how its laws fit together",
                    "theory_statements": [{
                        "id": "law_id",
                        "law": "specific qualitative or quantitative relationship",
                        "scope": "conditions, population, domain, and exceptions",
                        "evidence": [{"id": "evidence_record_id", "summary": "exact summary copied from the supplied evidence record"}],
                    }],
                    "conflicting_evidence": [{"id": "evidence_record_id", "summary": "exact summary copied from the supplied evidence record"}],
                    "unaccounted_evidence": [{"id": "evidence_record_id", "summary": "exact summary copied from the supplied evidence record"}],
                    "new_predictions_likely": ["prediction likely to follow from the theory"],
                    "new_predictions_unknown": ["potential prediction that needs additional evidence"],
                    "negative_experiments": ["a result or experiment that would count against the theory"],
                }]
            }
        elif action == "generate_predictions":
            packet["graph"] = {
                "nodes": [asdict(item) for item in self.state.graph_nodes],
                "edges": [asdict(item) for item in self.state.graph_edges],
            }
            packet["discovery_documents"] = [
                asdict(item) for item in self.state.discovery_documents
            ]
            packet["theories"] = [asdict(item) for item in self.state.theories]
            packet["missing_laws"] = list(next_action.get("laws", []))
            packet["prompt"] = _load_prompt("04-generate-predictions.md")
            packet["expected_output"] = {
                "predictions": [{
                    "id": "prediction_id",
                    "theory_id": "theory_id",
                    "law_id": "law_id",
                    "specific_prediction": "concrete empirical commitment",
                    "operational_signals": ["observable signal"],
                    "strong_test_requirement": "comparison or counterfactual needed for a strong test",
                    "support_criteria": "observable result consistent with the prediction",
                    "contradiction_criteria": "observable result that would contradict the prediction",
                }]
            }
        elif action in {
            "generate_measurement_specifications",
            "repair_measurement_specifications",
        }:
            # Deliberately contains discovery-side material only.  The agent
            # may formulate a construct question from the prediction, but has
            # no access to held-out text, prevalence, or prior measurements.
            requested_prediction_ids = (
                set(next_action.get("prediction_ids", []))
                if action == "generate_measurement_specifications"
                else {
                    prediction_id
                    for repair in cast(list[dict[str, object]], next_action["repairs"])
                    for prediction_id in cast(list[str], repair["prediction_ids"])
                }
            )
            packet["predictions"] = [
                asdict(item)
                for item in self.state.predictions
                if item.id in requested_prediction_ids
            ]
            packet["discovery_documents"] = [
                asdict(item) for item in self.state.discovery_documents
            ]
            packet["falsification_tests"] = [
                asdict(item)
                for item in self.state.falsification_tests
                if item.prediction_id in requested_prediction_ids
            ]
            packet["existing_measurement_specifications"] = [
                asdict(item) for item in self.state.measurement_specifications
            ]
            packet["frozen_measurement_protocol"] = deepcopy(
                FROZEN_MEASUREMENT_PROTOCOL
            )
            packet["isolation"] = {
                "rule": (
                    "Do not inspect, request, infer from, or optimize against held-out "
                    "validation text, label frequencies, previous validation results, or "
                    "a desired direction of support. Your output will be frozen before any "
                    "held-out record is measured."
                )
            }
            if action == "repair_measurement_specifications":
                packet["measurement_repair_requests"] = deepcopy(
                    cast(list[dict[str, object]], next_action["repairs"])
                )
                packet["repair_round"] = next_action["repair_round"]
                packet["max_repair_rounds"] = next_action["max_repair_rounds"]
                packet["prompt"] = _load_prompt(
                    "05b-repair-missing-measurements.md"
                )
            else:
                packet["prompt"] = _load_prompt(
                    "05a-generate-measurement-specifications.md"
                )
            packet["expected_output"] = {
                "measurement_specifications": [{
                    "id": "measurement_specification_id",
                    "construct_id": "stable_construct_id",
                    "measurement_question": "Does the text ...?",
                    "source_prediction_ids": ["prediction_id"],
                }]
            }
        elif action == "compile_falsification_specifications":
            # Do not leak the graph path, which could otherwise be used to
            # recover discovery material outside the intentionally narrow packet.
            packet["study"] = {
                "study_id": self.state.config.study_id,
                "question": self.state.config.question,
            }
            packet["predictions"] = [asdict(item) for item in self.state.predictions]
            packet["falsification_tests"] = [
                asdict(item) for item in self.state.falsification_tests
            ]
            packet["missing_prediction_ids"] = [
                item for item in next_action.get("prediction_ids", [])
            ]
            packet["validation_data"] = self._validation_data_description()
            packet["measurement_specifications"] = [
                asdict(item) for item in self.state.measurement_specifications
            ]
            packet["falsification_language"] = {
                "templates": deepcopy(FALSIFICATION_TEMPLATES),
                "measurement_binding": (
                    "v1 templates bind condition/outcome directly. Explicit-group "
                    "templates bind stable aliases to frozen measurement specification IDs."
                ),
                "expression_language": {
                    "purpose": (
                        "Freeze group membership and composite outcomes before held-out "
                        "labels are generated. Expressions are evaluated only from frozen labels."
                    ),
                    "measurement": {
                        "op": "measurement",
                        "measurement": "measurement alias from measurement",
                    },
                    "not": {"op": "not", "operand": "expression"},
                    "all_or_any": {
                        "op": "all | any",
                        "operands": ["expression", "expression"],
                    },
                    "at_least": {
                        "op": "at_least",
                        "minimum": "pre-specified integer",
                        "operands": ["expression", "expression"],
                    },
                    "uncertain_handling": (
                        "An expression is UNCERTAIN whenever its frozen labels cannot "
                        "determine PRESENT or ABSENT; such records are mechanically excluded."
                    ),
                },
            }
            packet["falsification_isolation"] = {
                "rule": (
                    "Do not use discovery documents, evidence records, graph-node examples, "
                    "or records selected to build them. Design from this validation-data "
                    "description only. When the test is executed, apply it to the full "
                    "declared validation partition rather than only node-selected records."
                ),
                "partition_role": (
                    self.state.validation_data.partition_role
                    if self.state.validation_data is not None
                    else "unknown"
                ),
                "separation_note": (
                    self.state.validation_data.separation_note
                    if self.state.validation_data is not None
                    else ""
                ),
            }
            packet["existing_falsification_specifications"] = [
                asdict(item) for item in self.state.falsification_specifications
            ]
            packet["prompt"] = _load_prompt("05-make-predictions-measurable.md")
            packet["expected_output"] = {
                "falsification_specifications": [{
                    "id": "falsification_specification_id",
                    "prediction_id": "prediction_id",
                    "template_id": "a supplied closed template ID",
                    "measurable_implication": {"condition": "condition", "outcome": "outcome", "relation": "template relation"},
                    "measurement": {"measurement_alias_or_template_variable": "frozen measurement specification ID"},
                    "expressions": {"group_a_or_condition": {"op": "measurement", "measurement": "measurement alias"}},
                    "population": {"include": "all_valid_records", "exclude": []},
                    "comparison": {"group_a": "template group", "group_b": "template group", "estimand": "template estimand"},
                    "hypotheses": {"expected": "template expectation", "opposite": "template opposite"},
                    "decision_rule": {"type": "template decision type"},
                    "required_fields": ["validation_data_field"],
                    "unresolved": [],
                    "missing_measurement_constructs": ["atomic_construct_id_if_missing"],
                    "readiness": "ready|not_ready",
                }],
            }
        elif action == "measure_validation_records":
            requested_ids = list(next_action.get("measurement_specification_ids", []))
            if len(requested_ids) != 1:
                raise ValueError("automatic measurement packets must contain exactly one specification")
            specification = self._resolve_measurement_specification(requested_ids[0])
            # The measurement worker receives no prediction, theory, expected
            # relation, decision rule, or existing label. It only gets the
            # frozen question and its isolated input records.
            packet["study"] = {"study_id": self.state.config.study_id}
            packet["measurement_specification"] = asdict(specification)
            packet["validation_records_path"] = str(self._validation_records_path())
            packet["text_field"] = self._measurement_required_fields()[1]
            packet["measurement_artifact_path"] = str(
                self._measurement_artifact_path(specification.id)
            )
            packet["measurement_isolation"] = {
                "rule": (
                    "For every record, use only its text and this frozen question. "
                    "Return PRESENT, ABSENT, or UNCERTAIN. Do not read or infer any "
                    "theory, prediction, hypothesis, support criterion, or validation result."
                )
            }
        elif action == "execute_falsification_tests":
            # Execution sees a frozen prediction/test contract and the isolated
            # validation partition only. It intentionally cannot recover the
            # discovery graph, source documents, or evidence records from this packet.
            requested_prediction_ids = list(next_action.get("prediction_ids", []))
            if len(requested_prediction_ids) != 1:
                raise ValueError("automatic execution packets must contain exactly one prediction")
            prediction_id = requested_prediction_ids[0]
            packet["study"] = {"study_id": self.state.config.study_id}
            packet["falsification_specifications"] = [
                asdict(item)
                for item in self.state.falsification_specifications
                if item.prediction_id == prediction_id
            ]
            packet["measurement_runs"] = [
                asdict(item)
                for item in self.state.measurement_runs
                if item.measurement_specification_id
                in {
                    value
                    for spec in self.state.falsification_specifications
                    if spec.prediction_id == prediction_id
                    for value in spec.measurement.values()
                }
            ]
            packet["execution_artifact_path"] = str(
                self._execution_artifact_path(prediction_id)
            )
            packet["execution_results_path"] = str(
                self._execution_results_path(prediction_id)
            )
            packet["execution_isolation"] = {
                "rule": (
                    "Use only validation_records_path and the frozen falsification "
                    "specification in this packet. Do not read discovery documents, "
                    "evidence records, the graph, the graph source, or any source corpus."
                ),
                "partition_role": self.state.validation_data.partition_role,
                "separation_note": self.state.validation_data.separation_note,
            }
            packet["prompt"] = _load_prompt("06-execute-falsification-tests.md")
            packet["expected_output"] = {
                "validation_results": [{
                    "id": "validation_result_id",
                    "prediction_id": "prediction_id",
                    "status": "prediction_consistent|observed_directional_contradiction|inconclusive|not_testable",
                    "summary": "verbatim summary emitted by execution_results_path",
                    "evidence": [],
                    "falsification_specification_id": "exact frozen specification ID",
                    "specification_hash": "exact frozen specification hash",
                    "execution_artifact": self._execution_artifact_relative_path(
                        prediction_id
                    ),
                }]
            }
        else:
            packet["prompt"] = _load_prompt("06-human-review.md")
            packet["summary"] = self.status(include_next_action=False)
        return packet

    def submit(self, payload: dict[str, object]) -> dict[str, object]:
        action = str(self.next_action()["action"])
        if action in {"ready_for_review", "complete"}:
            raise ValueError("the workflow has no remaining automatic stage")
        if action == "validation_execution_blocked":
            raise ValueError("automatic validation execution is blocked; revise the listed test or validation setup")
        if action == "measure_validation_records":
            raise ValueError(
                "held-out measurement is performed only by the frozen measurement worker"
            )
        if action == "execute_falsification_tests":
            self._validate_execution_files(payload)
        with tempfile.TemporaryDirectory(prefix="theory-discovery-validation-") as temp_root:
            trial = TheoryDiscoveryProject(temp_root, deepcopy(self.state))
            trial._submit_for_action(action, payload)
        before = self.status(include_next_action=False)["counts"]
        self._submit_for_action(action, payload)
        return {
            "accepted_action": action,
            "counts_before": before,
            "counts_after": self.status(include_next_action=False)["counts"],
            "next_action": self.next_action(),
        }

    def _submit_for_action(self, action: str, payload: dict[str, object]) -> None:
        if action == "import_graph":
            self.import_graph(_mapping_object_list(payload, "nodes"), _mapping_object_list(payload, "edges", allow_empty=True))
            return
        if action == "discover_relationships":
            for item in _mapping_object_list(payload, "relationships"):
                self.add_relationship(
                    _mapping_text(item, "source_node_id"),
                    _mapping_text(item, "target_node_id"),
                    cast(RelationType, _mapping_text(item, "relation_type")),
                    cast(RelationDirection, _mapping_text(item, "direction")),
                    _mapping_text(item, "rationale"),
                    relationship_id=_mapping_optional_text(item, "id") or None,
                )
            return
        if action == "generate_theories":
            for item in _mapping_object_list(payload, "theories"):
                self.add_theory(
                    _mapping_text(item, "name"),
                    _mapping_text(item, "description"),
                    _required_object_list(item, "theory_statements"),
                    _required_object_list(item, "conflicting_evidence"),
                    _required_object_list(item, "unaccounted_evidence"),
                    _required_string_list(item, "new_predictions_likely"),
                    _required_string_list(item, "new_predictions_unknown"),
                    _required_string_list(item, "negative_experiments"),
                    theory_id=_mapping_optional_text(item, "id") or None,
                )
            return
        if action == "generate_predictions":
            for item in _mapping_object_list(payload, "predictions"):
                self.add_prediction(
                    _mapping_text(item, "theory_id"),
                    _mapping_text(item, "law_id"),
                    _mapping_text(item, "specific_prediction"),
                    _required_string_list(item, "operational_signals"),
                    _mapping_text(item, "strong_test_requirement"),
                    _mapping_text(item, "support_criteria"),
                    _mapping_text(item, "contradiction_criteria"),
                    prediction_id=_mapping_optional_text(item, "id") or None,
                )
            return
        if action == "generate_measurement_specifications":
            for item in _mapping_object_list(payload, "measurement_specifications"):
                self.add_measurement_specification(
                    _mapping_text(item, "construct_id"),
                    _mapping_text(item, "measurement_question"),
                    _required_string_list(item, "source_prediction_ids"),
                    specification_id=_mapping_optional_text(item, "id") or None,
                )
            return
        if action == "repair_measurement_specifications":
            self._submit_measurement_repair(payload)
            return
        if action == "compile_falsification_specifications":
            for item in _mapping_object_list(payload, "falsification_specifications"):
                self.add_falsification_specification(
                    _mapping_text(item, "prediction_id"),
                    _mapping_text(item, "template_id"),
                    _mapping_string_dict(item, "measurable_implication"),
                    _mapping_string_dict(item, "measurement"),
                    _mapping_object(item, "population"),
                    _mapping_string_dict(item, "comparison"),
                    _mapping_string_dict(item, "hypotheses"),
                    _mapping_string_dict(item, "decision_rule"),
                    _required_string_list(item, "required_fields"),
                    _required_string_list(item, "unresolved"),
                    _mapping_text(item, "readiness"),
                    specification_id=_mapping_optional_text(item, "id") or None,
                    expressions=_mapping_expression_dict(item, "expressions"),
                    missing_measurement_constructs=_mapping_string_list(
                        item, "missing_measurement_constructs"
                    ),
                )
            return
        if action == "execute_falsification_tests":
            pending_prediction_ids = self._pending_validation_prediction_ids()
            expected_prediction_ids = {pending_prediction_ids[0]} if pending_prediction_ids else set()
            results = _mapping_object_list(payload, "validation_results")
            submitted_prediction_ids = [
                _mapping_text(item, "prediction_id") for item in results
            ]
            if len(submitted_prediction_ids) != len(set(submitted_prediction_ids)):
                raise ValueError("validation_results must not repeat a prediction_id")
            if set(submitted_prediction_ids) != expected_prediction_ids:
                raise ValueError(
                    "validation_results must cover exactly the requested prediction IDs: "
                    + ", ".join(sorted(expected_prediction_ids))
                )
            for item in results:
                self.record_validation(
                    prediction_id=_mapping_text(item, "prediction_id"),
                    status=cast(ValidationStatus, _mapping_text(item, "status")),
                    summary=_mapping_text(item, "summary"),
                    evidence=_mapping_evidence_list(item, "evidence"),
                    validation_id=_mapping_optional_text(item, "id") or None,
                    execution_artifact=_mapping_text(item, "execution_artifact"),
                    falsification_specification_id=_mapping_text(
                        item, "falsification_specification_id"
                    ),
                    specification_hash=_mapping_text(item, "specification_hash"),
                )
            return
        raise ValueError(f"cannot submit output for unknown action: {action}")

    def status(self, *, include_next_action: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "study_id": self.state.config.study_id,
            "revision": self.state.revision,
            "workflow_status": self.workflow_status(),
            "counts": {
                "graph_nodes": len(self.state.graph_nodes),
                "graph_edges": len(self.state.graph_edges),
                "discovery_documents": len(self.state.discovery_documents),
                "candidate_relationships": len(self.state.relationships),
                "theories": len(self.state.theories),
                "predictions": len(self.state.predictions),
                "falsification_tests": len(self.state.falsification_tests),
                "falsification_specifications": len(
                    self.state.falsification_specifications
                ),
                "legacy_falsification_specifications": len(
                    self.state.legacy_falsification_specifications
                ),
                "measurement_specifications": len(
                    self.state.measurement_specifications
                ),
                "measurement_runs": len(self.state.measurement_runs),
                "measurement_plans": len(self.state.measurement_plans),
                "readiness_assessments": len(self.state.readiness_assessments),
                "validation_results": len(self.state.validation_results),
                "readiness_statuses": self._readiness_statuses(),
            },
        }
        if include_next_action:
            result["next_action"] = self.next_action()
        return result

    def workflow_status(self) -> WorkflowStatus:
        if (
            not self.state.predictions
            or not self.state.measurement_specifications
            or not self.state.falsification_specifications
        ):
            return "draft"
        covered_prediction_ids = {
            prediction_id
            for specification in self.state.measurement_specifications
            for prediction_id in specification.source_prediction_ids
        }
        if any(item.id not in covered_prediction_ids for item in self.state.predictions):
            return "draft"
        specified_prediction_ids = {
            item.prediction_id for item in self.state.falsification_specifications
        }
        if any(item.id not in specified_prediction_ids for item in self.state.predictions):
            return "draft"
        if any(
            item.readiness != "ready"
            for item in self.state.falsification_specifications
        ):
            return "blocked"
        if self.state.validation_data is not None and self.state.validation_data.auto_execute:
            if (
                self._pending_measurement_specification_ids()
                or self._pending_validation_prediction_ids()
            ):
                return "draft"
        return "ready"

    def _readiness_requirements(self) -> list[dict[str, str]]:
        requirements: list[dict[str, str]] = []
        for theory in self.state.theories:
            requirements.extend({"parent_type": "theory", "parent_id": theory.id, "field_key": field} for field in THEORY_FIELDS)
        for prediction in self.state.predictions:
            requirements.extend({"parent_type": "prediction", "parent_id": prediction.id, "field_key": field} for field in PREDICTION_FIELDS)
        for plan in self.state.measurement_plans:
            requirements.extend({"parent_type": "measurement", "parent_id": plan.id, "field_key": field} for field in MEASUREMENT_FIELDS)
        for specification in self.state.measurement_specifications:
            requirements.append(
                {
                    "parent_type": "measurement",
                    "parent_id": specification.id,
                    "field_key": "frozen_question",
                }
            )
        for specification in self.state.falsification_specifications:
            requirements.extend(
                {
                    "parent_type": "falsification_specification",
                    "parent_id": specification.id,
                    "field_key": field,
                }
                for field in SPECIFICATION_FIELDS
            )
        return requirements

    def _missing_readiness_requirements(self) -> list[dict[str, str]]:
        assessed = {(item.parent_type, item.parent_id, item.field_key) for item in self.state.readiness_assessments}
        return [item for item in self._readiness_requirements() if (item["parent_type"], item["parent_id"], item["field_key"]) not in assessed]

    def _readiness_statuses(self) -> dict[str, int]:
        return {status: sum(item.status == status for item in self.state.readiness_assessments) for status in ("missing", "draft", "ready", "needs_revision", "blocked")}

    def _validation_records_path(self) -> Path:
        if self.state.validation_data is None:
            raise ValueError("automatic execution requires validation_data")
        source = Path(self.state.validation_data.data_source)
        records_path = source if source.is_absolute() else self.root.parent / source
        records_path = records_path.resolve()
        if not records_path.is_file():
            raise ValueError(
                "automatic execution requires a readable validation data file: "
                + str(records_path)
            )
        return records_path

    def _validation_data_description(self) -> dict[str, object] | None:
        """Expose schema metadata to design stages, never the held-out rows."""
        if self.state.validation_data is None:
            return None
        return {
            "unit_of_analysis": self.state.validation_data.unit_of_analysis,
            "population": self.state.validation_data.population,
            "required_data_fields": self.state.validation_data.required_data_fields,
            "partition_role": self.state.validation_data.partition_role,
            "separation_note": self.state.validation_data.separation_note,
        }

    def _measurement_artifact_relative_path(self, specification_id: str) -> str:
        _require_identifier("measurement specification ID", specification_id)
        return f"{MEASUREMENT_DIRECTORY}/{specification_id}.json"

    def _measurement_artifact_path(self, specification_id: str) -> Path:
        return self.root.parent / self._measurement_artifact_relative_path(specification_id)

    def _pending_measurement_specification_ids(self) -> list[str]:
        required_ids = {
            specification_id
            for contract in self.state.falsification_specifications
            if contract.readiness == "ready"
            for specification_id in contract.measurement.values()
        }
        completed_ids = {
            item.measurement_specification_id
            for item in self.state.measurement_runs
            if any(
                specification.id == item.measurement_specification_id
                and specification.specification_hash == item.specification_hash
                for specification in self.state.measurement_specifications
            )
        }
        return sorted(required_ids - completed_ids)

    def record_measurement_run(
        self,
        measurement_specification_id: str,
        *,
        input_records_hash: str,
        output_artifact: str,
        output_hash: str,
        record_count: int,
        run_id: str | None = None,
    ) -> MeasurementRun:
        """Accept labels only if their immutable artifact matches its contract."""
        specification = self._resolve_measurement_specification(
            measurement_specification_id
        )
        if measurement_specification_id not in self._pending_measurement_specification_ids():
            raise ValueError("measurement run is not currently requested")
        _require_text("input_records_hash", input_records_hash)
        _require_text("output_hash", output_hash)
        if not isinstance(record_count, int) or record_count < 0:
            raise ValueError("record_count must be a non-negative integer")
        expected_artifact = self._measurement_artifact_relative_path(
            measurement_specification_id
        )
        if output_artifact != expected_artifact:
            raise ValueError("measurement output must use the frozen artifact path: " + expected_artifact)
        artifact_path = self._measurement_artifact_path(measurement_specification_id)
        if not artifact_path.is_file():
            raise ValueError("measurement output artifact was not created: " + str(artifact_path))
        try:
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError("measurement output artifact must be valid JSON: " + str(error)) from error
        if not isinstance(payload, dict):
            raise ValueError("measurement output artifact must contain an object")
        if payload.get("measurement_specification_id") != measurement_specification_id:
            raise ValueError("measurement artifact references the wrong specification")
        if payload.get("specification_hash") != specification.specification_hash:
            raise ValueError("measurement artifact does not match the frozen specification")
        labels = payload.get("labels")
        if not isinstance(labels, list) or len(labels) != record_count:
            raise ValueError("measurement artifact label count does not match record_count")
        seen_record_ids: set[str] = set()
        for label in labels:
            if not isinstance(label, dict):
                raise ValueError("every measurement label must be an object")
            record_id = label.get("record_id")
            value = label.get("label")
            if not isinstance(record_id, str) or not record_id.strip():
                raise ValueError("every measurement label needs a record_id")
            if record_id in seen_record_ids:
                raise ValueError("measurement artifact repeats a record_id")
            seen_record_ids.add(record_id)
            if value not in {"PRESENT", "ABSENT", "UNCERTAIN"}:
                raise ValueError("measurement labels must be PRESENT, ABSENT, or UNCERTAIN")
        expected_record_ids = self._validation_record_ids()
        if seen_record_ids != expected_record_ids:
            raise ValueError(
                "measurement artifact must label exactly the frozen validation record IDs"
            )
        actual_input_hash = sha256(self._validation_records_path().read_bytes()).hexdigest()
        if input_records_hash != actual_input_hash:
            raise ValueError("measurement input hash does not match the validation partition")
        actual_hash = sha256(artifact_path.read_bytes()).hexdigest()
        if actual_hash != output_hash:
            raise ValueError("measurement output hash does not match its artifact")
        run = MeasurementRun(
            id=run_id or _new_id("measurement_run"),
            measurement_specification_id=measurement_specification_id,
            specification_hash=specification.specification_hash,
            input_records_hash=input_records_hash,
            output_artifact=output_artifact,
            output_hash=output_hash,
            record_count=record_count,
        )
        self._require_unique_id(run.id)
        self.state.measurement_runs.append(run)
        self._commit(
            "held_out_measurement_completed",
            {
                "measurement_run_id": run.id,
                "measurement_specification_id": measurement_specification_id,
                "record_count": record_count,
            },
        )
        return run

    def _validation_record_ids(self) -> set[str]:
        records_path = self._validation_records_path()
        try:
            with records_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
        except csv.Error as error:
            raise ValueError("validation records must be readable CSV: " + str(error)) from error
        if not rows or any(not isinstance(row.get("record_id"), str) or not row["record_id"].strip() for row in rows):
            raise ValueError("validation records must contain a non-empty record_id column")
        record_ids = {str(row["record_id"]) for row in rows}
        if len(record_ids) != len(rows):
            raise ValueError("validation records must have unique record_id values")
        return record_ids

    def _pending_validation_prediction_ids(self) -> list[str]:
        completed_prediction_ids = {
            item.prediction_id
            for item in self.state.validation_results
            if item.falsification_specification_id
            and any(
                specification.id == item.falsification_specification_id
                and specification.specification_hash == item.specification_hash
                for specification in self.state.falsification_specifications
            )
        }
        return [
            item.id
            for item in self.state.predictions
            if item.id not in completed_prediction_ids
        ]

    def _execution_artifact_relative_path(self, prediction_id: str) -> str:
        _require_identifier("prediction ID", prediction_id)
        return f"{EXECUTION_DIRECTORY}/{prediction_id}.py"

    def _execution_results_relative_path(self, prediction_id: str) -> str:
        _require_identifier("prediction ID", prediction_id)
        return f"{EXECUTION_DIRECTORY}/{prediction_id}.json"

    def _execution_artifact_path(self, prediction_id: str) -> Path:
        return self.root.parent / self._execution_artifact_relative_path(prediction_id)

    def _execution_results_path(self, prediction_id: str) -> Path:
        return self.root.parent / self._execution_results_relative_path(prediction_id)

    def _validate_execution_artifact(
        self, prediction_id: str, execution_artifact: str
    ) -> None:
        expected = self._execution_artifact_relative_path(prediction_id)
        if execution_artifact != expected:
            raise ValueError(
                "automatic validation must use the prediction-specific execution artifact path: "
                + expected
            )

    def _validate_execution_files(self, payload: dict[str, object]) -> None:
        """Require a reproducible program and a returned payload it actually emitted."""
        results = _mapping_object_list(payload, "validation_results")
        if len(results) != 1:
            raise ValueError("automatic validation must submit exactly one result per stage")
        prediction_id = _mapping_text(results[0], "prediction_id")
        pending_prediction_ids = self._pending_validation_prediction_ids()
        if not pending_prediction_ids or prediction_id != pending_prediction_ids[0]:
            raise ValueError("automatic validation result does not match the requested prediction")
        artifact_path = self._execution_artifact_path(prediction_id)
        results_path = self._execution_results_path(prediction_id)
        if not artifact_path.is_file():
            raise ValueError(
                "automatic validation execution artifact was not created: "
                + str(artifact_path)
            )
        if not results_path.is_file():
            raise ValueError(
                "automatic validation result artifact was not created: "
                + str(results_path)
            )
        try:
            saved_result = json.loads(results_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(
                "automatic validation result artifact must be valid JSON: " + str(error)
            ) from error
        expected = {"validation_results": results}
        if saved_result != expected:
            raise ValueError(
                "submitted validation_results must exactly match the JSON emitted by "
                + self._execution_artifact_relative_path(prediction_id)
            )
        for result in expected["validation_results"]:
            self._validate_execution_artifact(
                prediction_id, _mapping_text(result, "execution_artifact")
            )

    def _require_unique_id(self, item_id: str) -> None:
        known = {
            *(item.id for item in self.state.graph_nodes),
            *(item.id for item in self.state.graph_edges),
            *(item.id for item in self.state.relationships),
            *(item.id for item in self.state.theories),
            *(
                statement.id
                for theory in self.state.theories
                for statement in theory.theory_statements
            ),
            *(item.id for item in self.state.predictions),
            *(item.id for item in self.state.falsification_tests),
            *(item.id for item in self.state.falsification_specifications),
            *(item.id for item in self.state.legacy_falsification_specifications),
            *(item.id for item in self.state.measurement_specifications),
            *(item.id for item in self.state.measurement_runs),
            *(item.id for item in self.state.measurement_plans),
            *(item.id for item in self.state.readiness_assessments),
            *(item.id for item in self.state.validation_results),
        }
        if item_id in known:
            raise ValueError(f"duplicate ID: {item_id}")

    def _validate_evidence_references(
        self, references: list[EvidenceReference]
    ) -> None:
        evidence_by_id = {item.id: item for item in self.state.evidence_records}
        known_evidence_ids = set(evidence_by_id)
        missing = sorted({item.id for item in references} - known_evidence_ids)
        if missing:
            raise ValueError(
                "theory references unknown evidence IDs: " + ", ".join(missing)
            )
        for reference in references:
            canonical = evidence_by_id[reference.id]
            if reference.summary != canonical.summary:
                raise ValueError(
                    "evidence summary must exactly match the supplied evidence record: "
                    + reference.id
                )
            if reference.source and reference.source != canonical.source:
                raise ValueError(
                    "evidence source must exactly match the supplied evidence record: "
                    + reference.id
                )
            if reference.locator and reference.locator != canonical.locator:
                raise ValueError(
                    "evidence locator must exactly match the supplied evidence record: "
                    + reference.id
                )

    def _require_supported_numbers(
        self, field_name: str, value: str, evidence: list[EvidenceReference]
    ) -> None:
        observed = set(NUMBER_PATTERN.findall(value))
        if not observed:
            return
        allowed = {
            number
            for item in evidence
            for number in NUMBER_PATTERN.findall(item.summary)
        }
        unsupported = sorted(observed - allowed)
        if unsupported:
            raise ValueError(
                f"{field_name} contains numerical content not present in its evidence: "
                + ", ".join(unsupported)
            )

    def _validate_required_data_fields(self, required_data: list[str]) -> None:
        if not required_data:
            return
        if self.state.validation_data is None:
            raise ValueError("required_data cannot be supplied without validation_data")
        available = set(self.state.validation_data.required_data_fields)
        missing = sorted(set(required_data) - available)
        if missing:
            raise ValueError(
                "falsification test references unavailable data fields: "
                + ", ".join(missing)
            )

    def _require_validation_data_separation(self) -> None:
        if self.state.validation_data is None:
            raise ValueError("an implementable test requires validation_data")
        _require_validation_data_separation(self.state.validation_data)

    def _commit(self, event_type: str, payload: dict[str, object]) -> None:
        self.state.revision += 1
        self.state.updated_at = utc_now()
        encoded = json.dumps(self.state.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        self.root.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=".state-", suffix=".json", dir=self.root, text=True)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, self.root / STATE_FILE)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
        event = {"revision": self.state.revision, "timestamp": self.state.updated_at, "type": event_type, "payload": payload}
        with (self.root / EVENTS_FILE).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")


def _find_by_id(items: list[object], item_id: str, label: str) -> object:
    for item in items:
        if getattr(item, "id") == item_id:
            return item
    raise ValueError(f"unknown {label} ID: {item_id}")


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


def _require_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_identifier(name: str, value: object) -> str:
    text = _require_text(name, value)
    if not ID_PATTERN.fullmatch(text):
        raise ValueError(
            f"{name} must match {ID_PATTERN.pattern!r}: {text!r}"
        )
    return text


def _parse_data_design(value: dict[str, object], name: str) -> DataDesign:
    partition_role = _mapping_optional_text(value, "partition_role") or "unknown"
    if partition_role not in {"held_out", "independent", "overlapping", "unknown"}:
        raise ValueError(
            f"{name}.partition_role must be held_out, independent, overlapping, or unknown"
        )
    auto_execute = _mapping_optional_bool(value, "auto_execute")
    raw_rules = _mapping_object_list(value, "measurement_rules", allow_empty=True)
    parsed_rules: list[MeasurementRule] = []
    known_rule_ids: set[str] = set()
    for raw_rule in raw_rules:
        rule_id = _mapping_text(raw_rule, "id")
        _require_identifier(f"{name} measurement rule ID", rule_id)
        if rule_id in known_rule_ids:
            raise ValueError(f"{name}.measurement_rules must have unique IDs")
        known_rule_ids.add(rule_id)
        source_field = _mapping_text(raw_rule, "source_field")
        kind = _mapping_text(raw_rule, "kind")
        if kind not in {"field_truthy", "contains_any"}:
            raise ValueError(
                f"{name} measurement rule kind must be field_truthy or contains_any"
            )
        values = _mapping_string_list(raw_rule, "values")
        if kind == "field_truthy" and values:
            raise ValueError("field_truthy measurement rules cannot have values")
        if kind == "contains_any" and not values:
            raise ValueError("contains_any measurement rules require one or more values")
        parsed_rules.append(
            MeasurementRule(
                id=rule_id,
                source_field=source_field,
                kind=cast("MeasurementRuleKind", kind),
                values=values,
                version=_mapping_optional_text(raw_rule, "version") or "v1",
            )
        )
    parsed = DataDesign(
        unit_of_analysis=_mapping_text(value, "unit_of_analysis"),
        population=_mapping_text(value, "population"),
        data_source=_mapping_text(value, "data_source"),
        time_window=_mapping_text(value, "time_window"),
        required_data_fields=_mapping_string_list(value, "required_data_fields"),
        partition_role=cast(ValidationPartition, partition_role),
        separation_note=_mapping_optional_text(value, "separation_note"),
        auto_execute=False if auto_execute is None else auto_execute,
        measurement_rules=parsed_rules,
    )
    if not parsed.required_data_fields:
        raise ValueError(f"{name}.required_data_fields must not be empty")
    for field_name in parsed.required_data_fields:
        _require_identifier(f"{name} field", field_name)
    for rule in parsed.measurement_rules:
        if rule.source_field not in parsed.required_data_fields:
            raise ValueError(
                f"{name} measurement rule references unavailable data field: "
                + rule.source_field
            )
    return parsed


def _parse_discovery_document(value: dict[str, object]) -> DiscoveryDocument:
    return DiscoveryDocument(
        id=_mapping_text(value, "id"),
        text=_mapping_text(value, "text"),
        source=_mapping_optional_text(value, "source"),
        locator=_mapping_optional_text(value, "locator"),
    )


def _require_validation_data_separation(data: DataDesign) -> None:
    if data.partition_role not in {"held_out", "independent"}:
        raise ValueError(
            "an implementable test requires validation_data.partition_role to be "
            "held_out or independent, not overlapping or unknown"
        )
    if not data.separation_note.strip():
        raise ValueError(
            "validation_data.separation_note must explain its separation from discovery material"
        )


def _reject_forbidden_analysis_content(field_name: str, value: str) -> None:
    match = FORBIDDEN_THRESHOLD_OR_WINDOW_PATTERN.search(value)
    if match:
        raise ValueError(
            f"{field_name} contains an unavailable threshold or sampling window: {match.group(0)!r}"
        )


def _normalized_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _mapping_text(item: dict[str, object], key: str) -> str:
    return _require_text(key, item.get(key))


def _mapping_optional_text(item: dict[str, object], key: str) -> str:
    value = item.get(key, "")
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _mapping_optional_bool(item: dict[str, object], key: str) -> bool | None:
    value = item.get(key)
    if value is not None and not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean or null")
    return value


def _mapping_object(item: dict[str, object], key: str) -> dict[str, object]:
    value = item.get(key)
    if not isinstance(value, dict) or not all(isinstance(entry_key, str) for entry_key in value):
        raise ValueError(f"{key} must be an object with string keys")
    return cast(dict[str, object], value)


def _mapping_expression_dict(
    item: dict[str, object], key: str
) -> dict[str, dict[str, object]]:
    """Read the optional v12 Boolean-expression mapping from a payload."""
    value = item.get(key, {})
    if not isinstance(value, dict) or not all(
        isinstance(name, str) and isinstance(expression, dict)
        for name, expression in value.items()
    ):
        raise ValueError(f"{key} must map names to expression objects")
    return cast(dict[str, dict[str, object]], value)


def _mapping_object_list(item: dict[str, object], key: str, *, allow_empty: bool = False) -> list[dict[str, object]]:
    value = item.get(key, [])
    if not isinstance(value, list):
        raise ValueError(f"{key} must be an array of objects")
    if not allow_empty and not value:
        raise ValueError(f"{key} must not be empty")
    if not all(isinstance(entry, dict) and all(isinstance(entry_key, str) for entry_key in entry) for entry in value):
        raise ValueError(f"{key} must contain only objects with string keys")
    return cast(list[dict[str, object]], value)


def _required_object_list(item: dict[str, object], key: str) -> list[dict[str, object]]:
    if key not in item:
        raise ValueError(f"{key} is required")
    return _mapping_object_list(item, key, allow_empty=True)


def _validated_object_list(name: str, value: list[dict[str, object]]) -> list[dict[str, object]]:
    if not isinstance(value, list) or not all(
        isinstance(item, dict) and all(isinstance(key, str) for key in item)
        for item in value
    ):
        raise ValueError(f"{name} must contain only objects with string keys")
    return value


def _mapping_string_list(item: dict[str, object], key: str) -> list[str]:
    value = item.get(key, [])
    if not isinstance(value, list):
        raise ValueError(f"{key} must be an array of strings")
    return _validated_string_list(key, value, allow_empty=True)


def _required_string_list(item: dict[str, object], key: str) -> list[str]:
    if key not in item:
        raise ValueError(f"{key} is required")
    return _mapping_string_list(item, key)


def _mapping_string_dict(item: dict[str, object], key: str) -> dict[str, str]:
    value = _mapping_object(item, key)
    return _mapping_string_dict_value(value, key)


def _mapping_string_dict_value(value: dict[str, object], key: str) -> dict[str, str]:
    if not all(isinstance(entry_key, str) and isinstance(entry_value, str) and entry_value.strip() for entry_key, entry_value in value.items()):
        raise ValueError(f"{key} must contain only non-empty string values")
    return {str(entry_key): str(entry_value) for entry_key, entry_value in value.items()}


def _validated_expression_dict(
    name: str, value: dict[str, dict[str, object]]
) -> dict[str, dict[str, object]]:
    """Validate the outer container of the frozen Boolean DSL.

    Individual trees are checked only when a template requires them so a
    compiler can still submit a concrete ``not_ready`` contract for a missing
    group definition.  The executor never evaluates a non-ready contract.
    """
    if not isinstance(value, dict) or not all(
        isinstance(key, str)
        and key.strip()
        and isinstance(expression, dict)
        for key, expression in value.items()
    ):
        raise ValueError(f"{name} must map non-empty names to expression objects")
    return deepcopy(cast(dict[str, dict[str, object]], value))


def _expression_measurement_aliases(
    expression: dict[str, object], *, _depth: int = 0
) -> set[str]:
    """Return aliases used by one closed Boolean expression tree.

    The grammar has no arbitrary values, callbacks, or source-text access:
    ``measurement``, ``not``, ``all``, ``any``, and ``at_least`` are the only
    possible operations.  Thus every group decision remains a pure function
    of labels already emitted under frozen measurement contracts.
    """
    if _depth > 16:
        raise ValueError("expression nesting exceeds the frozen depth limit")
    operation = expression.get("op")
    if not isinstance(operation, str):
        raise ValueError("expression needs a string op")
    if operation == "measurement":
        if set(expression) != {"op", "measurement"}:
            raise ValueError("measurement expressions may contain only op and measurement")
        alias = expression.get("measurement")
        if not isinstance(alias, str) or not alias.strip():
            raise ValueError("measurement expressions need a non-empty measurement alias")
        _require_identifier("expression measurement alias", alias)
        return {alias}
    if operation == "not":
        if set(expression) != {"op", "operand"}:
            raise ValueError("not expressions may contain only op and operand")
        operand = expression.get("operand")
        if not isinstance(operand, dict):
            raise ValueError("not expressions need an object operand")
        return _expression_measurement_aliases(operand, _depth=_depth + 1)
    if operation in {"all", "any", "at_least"}:
        permitted_keys = {"op", "operands"}
        if operation == "at_least":
            permitted_keys.add("minimum")
        if set(expression) != permitted_keys:
            raise ValueError(
                f"{operation} expressions must contain only "
                + ", ".join(sorted(permitted_keys))
            )
        operands = expression.get("operands")
        if not isinstance(operands, list) or len(operands) < 2 or not all(
            isinstance(operand, dict) for operand in operands
        ):
            raise ValueError(f"{operation} expressions need at least two object operands")
        if operation == "at_least":
            minimum = expression.get("minimum")
            if (
                not isinstance(minimum, int)
                or isinstance(minimum, bool)
                or minimum < 1
                or minimum > len(operands)
            ):
                raise ValueError("at_least minimum must be an integer within its operand count")
        aliases: set[str] = set()
        for operand in operands:
            aliases.update(_expression_measurement_aliases(operand, _depth=_depth + 1))
        return aliases
    raise ValueError("unsupported expression op: " + operation)


def _mapping_evidence_list(item: dict[str, object], key: str) -> list[dict[str, object]]:
    return _mapping_object_list(item, key, allow_empty=True)


def _parse_evidence(item: dict[str, object]) -> EvidenceReference:
    if not isinstance(item, dict):
        raise ValueError("every evidence item must be an object")
    role = _mapping_optional_text(item, "role") or "supporting"
    if role not in {"supporting", "conflicting", "unclear"}:
        raise ValueError("invalid evidence role")
    evidence_id = _mapping_optional_text(item, "id") or _new_id("evidence")
    _require_identifier("evidence ID", evidence_id)
    return EvidenceReference(
        id=evidence_id,
        summary=_mapping_text(item, "summary"),
        source=_mapping_optional_text(item, "source"),
        role=cast(EvidenceRole, role),
        locator=_mapping_optional_text(item, "locator"),
    )


def _validated_string_list(name: str, values: list[object], *, allow_empty: bool = False) -> list[str]:
    if not allow_empty and not values:
        raise ValueError(f"{name} must not be empty")
    if not all(isinstance(value, str) and value.strip() for value in values):
        raise ValueError(f"{name} must contain only non-empty strings")
    return [str(value) for value in values]
