"""Small, JSON-serializable models for the PRIME theory workflow.

The schema mirrors the research objects:

    Theory = Description + scoped law statements + evidence accounting
    Prediction = Falsifiable implication of one theory law
    Falsification test = Null + alternative + supplied data requirements
    Validation = Support / Contradict / No evidence
    Workflow = Draft / Ready / Needs revision / Blocked
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Literal


RelationType = Literal["association", "sequence", "mechanism", "moderation"]
RelationDirection = Literal["positive", "negative", "conditional", "unspecified"]
EvidenceRole = Literal["supporting", "conflicting", "unclear"]
ReadinessStatus = Literal["missing", "draft", "ready", "needs_revision", "blocked"]
ReadinessParentType = Literal[
    "theory", "prediction", "measurement", "falsification_specification"
]
# The first three values are retained only so projects created before schema v9
# can still be opened.  New compiler-backed executions use the latter four.
ValidationStatus = Literal[
    "support",
    "contradict",
    "no_evidence",
    "prediction_consistent",
    "observed_directional_contradiction",
    "inconclusive",
    "not_testable",
]
WorkflowStatus = Literal["draft", "ready", "needs_revision", "blocked"]
TestRelevance = Literal["direct", "limited", "none"]
TestImplementability = Literal["ready", "partial", "blocked"]
ValidationPartition = Literal["held_out", "independent", "overlapping", "unknown"]
MeasurementRuleKind = Literal["field_truthy", "contains_any"]
SpecificationReadiness = Literal["ready", "not_ready"]
MeasurementLabel = Literal["PRESENT", "ABSENT", "UNCERTAIN"]


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(slots=True)
class StudyConfig:
    study_id: str
    question: str
    graph_source: str | None = None
    created_at: str = field(default_factory=utc_now)


@dataclass(slots=True)
class GraphNode:
    id: str
    label: str
    description: str = ""
    examples: list[str] = field(default_factory=list)


@dataclass(slots=True)
class GraphEdge:
    id: str
    source_node_id: str
    target_node_id: str
    label: str


@dataclass(slots=True)
class RelationshipCandidate:
    id: str
    source_node_id: str
    target_node_id: str
    relation_type: RelationType
    direction: RelationDirection
    rationale: str
    created_at: str = field(default_factory=utc_now)


@dataclass(slots=True)
class EvidenceReference:
    """A source-backed item of support, conflict, or unresolved evidence."""

    id: str
    summary: str
    source: str = ""
    role: EvidenceRole = "supporting"
    locator: str = ""


@dataclass(slots=True)
class DiscoveryDocument:
    """Original material available while discovering candidate explanations.

    These documents are deliberately not included in a falsification-design
    packet.  They may be raw records, passages, or other source text used to
    construct the graph and evidence records.
    """

    id: str
    text: str
    source: str = ""
    locator: str = ""


@dataclass(slots=True)
class TheoryStatement:
    """One scoped law within a broader theory."""

    id: str
    law: str
    scope: str
    evidence: list[EvidenceReference] = field(default_factory=list)


@dataclass(slots=True)
class Theory:
    """An evidence-grounded theory with one or more scoped laws."""

    id: str
    name: str
    description: str
    theory_statements: list[TheoryStatement] = field(default_factory=list)
    conflicting_evidence: list[EvidenceReference] = field(default_factory=list)
    unaccounted_evidence: list[EvidenceReference] = field(default_factory=list)
    new_predictions_likely: list[str] = field(default_factory=list)
    new_predictions_unknown: list[str] = field(default_factory=list)
    negative_experiments: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)


@dataclass(slots=True)
class Prediction:
    """A falsifiable empirical implication of one scoped theory law."""

    id: str
    theory_id: str
    law_id: str
    specific_prediction: str
    operational_signals: list[str] = field(default_factory=list)
    strong_test_requirement: str = ""
    support_criteria: str = ""
    contradiction_criteria: str = ""
    created_at: str = field(default_factory=utc_now)


@dataclass(slots=True)
class FalsificationTest:
    """One implementable attempt to contradict a prediction."""

    id: str
    prediction_id: str
    test_description: str
    null_hypothesis: str
    alternative_hypothesis: str
    required_data: list[str] = field(default_factory=list)
    missing_information: list[str] = field(default_factory=list)
    relevance: TestRelevance = "direct"
    implementability: TestImplementability = "blocked"
    created_at: str = field(default_factory=utc_now)


@dataclass(slots=True)
class MeasurementRule:
    """A deterministic, prediction-independent way to derive one indicator.

    Rules belong to the validation-data declaration and must be fixed before
    the held-out records are exposed.  ``field_truthy`` needs no values;
    ``contains_any`` checks literal phrases case-insensitively.
    """

    id: str
    source_field: str
    kind: MeasurementRuleKind
    values: list[str] = field(default_factory=list)
    version: str = "v1"


@dataclass(slots=True)
class MeasurementSpecification:
    """A question frozen before any held-out text is inspected.

    The generation agent supplies only the construct and question.  The
    protocol fields below are filled and hashed by Python, so a later model
    cannot silently change labels, decoding, or how uncertainty is handled.
    """

    id: str
    construct_id: str
    measurement_question: str
    source_prediction_ids: list[str] = field(default_factory=list)
    labels: list[MeasurementLabel] = field(
        default_factory=lambda: ["PRESENT", "ABSENT", "UNCERTAIN"]
    )
    model_id: str = "qwen3.6-27b-fp8"
    model_revision: str = "qwen3.6-27b-fp8"
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 16
    prompt_version: str = "frozen-measurement-v1"
    uncertain_handling: str = "exclude"
    specification_hash: str = ""
    created_at: str = field(default_factory=utc_now)


@dataclass(slots=True)
class MeasurementRun:
    """Immutable reference to labels produced under one frozen question."""

    id: str
    measurement_specification_id: str
    specification_hash: str
    input_records_hash: str
    output_artifact: str
    output_hash: str
    record_count: int
    status: str = "completed"
    created_at: str = field(default_factory=utc_now)


@dataclass(slots=True)
class FalsificationSpecification:
    """A frozen, machine-executable contract compiled before validation.

    It deliberately stores structured rules rather than a prose test.  The
    executor must be able to use this object without interpreting the parent
    prediction or making a measurement or decision judgement.
    """

    id: str
    prediction_id: str
    template_id: str
    measurable_implication: dict[str, str]
    measurement: dict[str, str]
    population: dict[str, object]
    comparison: dict[str, str]
    hypotheses: dict[str, str]
    decision_rule: dict[str, str]
    # v12 templates can give measurements stable aliases and derive group or
    # outcome membership from a tiny frozen Boolean language.  The expression
    # tree is interpreted only by the deterministic executor, never by the
    # held-out measurement model.
    expressions: dict[str, dict[str, object]] = field(default_factory=dict)
    measurement_rule_snapshot: dict[str, MeasurementRule] = field(
        default_factory=dict
    )
    # ``legacy_lexical_v1`` is preserved only for historical v9 projects.
    # New specifications bind to frozen LLM measurement questions instead.
    measurement_protocol: str = "frozen_llm_question_v1"
    measurement_specification_snapshot: dict[str, MeasurementSpecification] = field(
        default_factory=dict
    )
    required_fields: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    # A compiler may diagnose an absent *atomic* construct before held-out
    # access.  The controller can then request exactly those questions from
    # the discovery-side generator and recompile, rather than asking a person
    # to patch a question by hand.
    missing_measurement_constructs: list[str] = field(default_factory=list)
    readiness: SpecificationReadiness = "not_ready"
    specification_hash: str = ""
    created_at: str = field(default_factory=utc_now)


@dataclass(slots=True)
class DataDesign:
    unit_of_analysis: str
    population: str
    data_source: str
    time_window: str
    required_data_fields: list[str] = field(default_factory=list)
    partition_role: ValidationPartition = "unknown"
    separation_note: str = ""
    auto_execute: bool = False
    measurement_rules: list[MeasurementRule] = field(default_factory=list)


@dataclass(slots=True)
class MeasurementPlan:
    """How one prediction becomes observable and reproducibly codable."""

    id: str
    prediction_id: str
    indicators: dict[str, str]
    operationalization: dict[str, str]
    data_design: DataDesign
    missing_information: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)


# Short alias for callers that refer to the object simply as “Measurement”.
Measurement = MeasurementPlan


@dataclass(slots=True)
class ReadinessAssessment:
    """A preflight review of one required field, not empirical validity."""

    id: str
    parent_type: ReadinessParentType
    parent_id: str
    field_key: str
    status: ReadinessStatus
    review_note: str
    indicator_available: bool | None = None
    evidence_needed: str = ""
    missing_information: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)


@dataclass(slots=True)
class ValidationResult:
    """An optional later-stage empirical result for a measurable prediction."""

    id: str
    prediction_id: str
    status: ValidationStatus
    summary: str
    evidence: list[EvidenceReference] = field(default_factory=list)
    execution_artifact: str = ""
    falsification_specification_id: str = ""
    specification_hash: str = ""
    result_semantics: str = "legacy_v1"
    created_at: str = field(default_factory=utc_now)


@dataclass(slots=True)
class ProjectState:
    schema_version: int
    revision: int
    config: StudyConfig
    evidence_records: list[EvidenceReference] = field(default_factory=list)
    discovery_documents: list[DiscoveryDocument] = field(default_factory=list)
    validation_data: DataDesign | None = None
    graph_nodes: list[GraphNode] = field(default_factory=list)
    graph_edges: list[GraphEdge] = field(default_factory=list)
    relationships: list[RelationshipCandidate] = field(default_factory=list)
    theories: list[Theory] = field(default_factory=list)
    predictions: list[Prediction] = field(default_factory=list)
    falsification_tests: list[FalsificationTest] = field(default_factory=list)
    falsification_specifications: list[FalsificationSpecification] = field(
        default_factory=list
    )
    # Earlier compiler output is retained for auditability but never advances
    # the current frozen-question validation workflow.
    legacy_falsification_specifications: list[FalsificationSpecification] = field(
        default_factory=list
    )
    # Unused contracts replaced during an automatic missing-construct repair.
    # They remain inspectable but never advance the active workflow.
    superseded_falsification_specifications: list[FalsificationSpecification] = field(
        default_factory=list
    )
    measurement_specifications: list[MeasurementSpecification] = field(
        default_factory=list
    )
    # A measurement contract can only be superseded before it has been used on
    # held-out data.  Retain it for audit rather than silently overwriting it.
    superseded_measurement_specifications: list[MeasurementSpecification] = field(
        default_factory=list
    )
    # The repair loop is intentionally bounded.  A repeated diagnosis after
    # the permitted attempts is surfaced as blocked instead of looping or
    # silently weakening a prediction.
    measurement_repair_rounds: int = 0
    falsification_repair_prediction_ids: list[str] = field(default_factory=list)
    measurement_runs: list[MeasurementRun] = field(default_factory=list)
    measurement_plans: list[MeasurementPlan] = field(default_factory=list)
    readiness_assessments: list[ReadinessAssessment] = field(default_factory=list)
    validation_results: list[ValidationResult] = field(default_factory=list)
    updated_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "ProjectState":
        evidence_records = [
            _parse_evidence_dict(entry)
            for entry in _expect_dict_list(value, "evidence_records")
        ]
        discovery_documents = [
            _parse_discovery_document_dict(entry)
            for entry in _expect_dict_list(value, "discovery_documents")
        ]
        theories: list[Theory] = []
        for item in _expect_dict_list(value, "theories"):
            raw_statements = item.get("theory_statements", [])
            if not isinstance(raw_statements, list):
                raise ValueError("theory_statements must be an array")
            statements: list[TheoryStatement] = []
            for statement in raw_statements:
                if not isinstance(statement, dict):
                    raise ValueError("every theory statement must be an object")
                raw_evidence = statement.get("evidence", [])
                if not isinstance(raw_evidence, list):
                    raise ValueError("theory statement evidence must be an array")
                statements.append(
                    TheoryStatement(
                        **{
                            **statement,
                            "evidence": [_parse_evidence_dict(entry) for entry in raw_evidence],
                        }
                    )
                )
            conflicting_evidence = [
                _parse_evidence_dict(entry)
                for entry in _expect_dict_list(item, "conflicting_evidence")
            ]
            unaccounted_evidence = [
                _parse_evidence_dict(entry)
                for entry in _expect_dict_list(item, "unaccounted_evidence")
            ]
            theories.append(
                Theory(
                    **{
                        **item,
                        "theory_statements": statements,
                        "conflicting_evidence": conflicting_evidence,
                        "unaccounted_evidence": unaccounted_evidence,
                    }
                )
            )

        validation_results: list[ValidationResult] = []
        for item in _expect_dict_list(value, "validation_results"):
            raw_evidence = item.get("evidence", [])
            if not isinstance(raw_evidence, list):
                raise ValueError("validation evidence must be an array")
            evidence = [_parse_evidence_dict(entry) for entry in raw_evidence]
            validation_results.append(ValidationResult(**{**item, "evidence": evidence}))

        raw_validation_data = value.get("validation_data")
        if raw_validation_data is None:
            raw_validation_data = value.get("available_data")
        if raw_validation_data is None:
            validation_data = None
        elif isinstance(raw_validation_data, dict):
            validation_data = _parse_data_design_dict(raw_validation_data)
        else:
            raise ValueError("validation_data must be an object or null")

        measurement_plans: list[MeasurementPlan] = []
        for item in _expect_dict_list(value, "measurement_plans"):
            raw_design = item.get("data_design")
            if not isinstance(raw_design, dict):
                raise ValueError("data_design must be an object")
            measurement_plans.append(
                MeasurementPlan(
                    **{**item, "data_design": _parse_data_design_dict(raw_design)}
                )
            )

        return cls(
            schema_version=_expect_int(value, "schema_version"),
            revision=_expect_int(value, "revision"),
            config=StudyConfig(**_expect_dict(value, "config")),
            evidence_records=evidence_records,
            discovery_documents=discovery_documents,
            validation_data=validation_data,
            graph_nodes=[GraphNode(**item) for item in _expect_dict_list(value, "graph_nodes")],
            graph_edges=[GraphEdge(**item) for item in _expect_dict_list(value, "graph_edges")],
            relationships=[
                RelationshipCandidate(**item)
                for item in _expect_dict_list(value, "relationships")
            ],
            theories=theories,
            predictions=[Prediction(**item) for item in _expect_dict_list(value, "predictions")],
            falsification_tests=[
                FalsificationTest(**item)
                for item in _expect_dict_list(value, "falsification_tests")
            ],
            falsification_specifications=[
                _parse_falsification_specification_dict(item)
                for item in _expect_dict_list(value, "falsification_specifications")
            ],
            legacy_falsification_specifications=[
                _parse_falsification_specification_dict(item)
                for item in _expect_dict_list(
                    value, "legacy_falsification_specifications"
                )
            ],
            superseded_falsification_specifications=[
                _parse_falsification_specification_dict(item)
                for item in _expect_dict_list(
                    value, "superseded_falsification_specifications"
                )
            ],
            measurement_specifications=[
                MeasurementSpecification(**item)
                for item in _expect_dict_list(value, "measurement_specifications")
            ],
            superseded_measurement_specifications=[
                MeasurementSpecification(**item)
                for item in _expect_dict_list(
                    value, "superseded_measurement_specifications"
                )
            ],
            measurement_repair_rounds=_expect_nonnegative_int(
                value, "measurement_repair_rounds"
            ),
            falsification_repair_prediction_ids=_expect_string_list(
                value, "falsification_repair_prediction_ids"
            ),
            measurement_runs=[
                MeasurementRun(**item)
                for item in _expect_dict_list(value, "measurement_runs")
            ],
            measurement_plans=measurement_plans,
            readiness_assessments=[
                ReadinessAssessment(**item)
                for item in _expect_dict_list(value, "readiness_assessments")
            ],
            validation_results=validation_results,
            updated_at=_expect_str(value, "updated_at"),
        )


def _parse_evidence_dict(value: object) -> EvidenceReference:
    if not isinstance(value, dict):
        raise ValueError("every evidence item must be an object")
    return EvidenceReference(**value)


def _parse_discovery_document_dict(value: object) -> DiscoveryDocument:
    if not isinstance(value, dict):
        raise ValueError("every discovery document must be an object")
    return DiscoveryDocument(**value)


def _parse_data_design_dict(value: dict[str, object]) -> DataDesign:
    raw_rules = value.get("measurement_rules", [])
    if not isinstance(raw_rules, list):
        raise ValueError("measurement_rules must be an array")
    rules: list[MeasurementRule] = []
    for raw_rule in raw_rules:
        if not isinstance(raw_rule, dict):
            raise ValueError("every measurement rule must be an object")
        rules.append(MeasurementRule(**raw_rule))
    return DataDesign(**{**value, "measurement_rules": rules})


def _parse_falsification_specification_dict(
    value: dict[str, object],
) -> FalsificationSpecification:
    object_fields = ("population",)
    string_dict_fields = (
        "measurable_implication",
        "measurement",
        "comparison",
        "hypotheses",
        "decision_rule",
    )
    for key in object_fields:
        if not isinstance(value.get(key), dict):
            raise ValueError(f"falsification specification {key} must be an object")
    for key in string_dict_fields:
        raw = value.get(key)
        if not isinstance(raw, dict) or not all(
            isinstance(entry_key, str) and isinstance(entry_value, str)
            for entry_key, entry_value in raw.items()
        ):
            raise ValueError(
                f"falsification specification {key} must contain string values"
            )
    raw_expressions = value.get("expressions", {})
    if not isinstance(raw_expressions, dict) or not all(
        isinstance(key, str) and isinstance(expression, dict)
        for key, expression in raw_expressions.items()
    ):
        raise ValueError("falsification specification expressions must map names to objects")
    raw_snapshot = value.get("measurement_rule_snapshot", {})
    if not isinstance(raw_snapshot, dict) or not all(
        isinstance(key, str) and isinstance(rule, dict)
        for key, rule in raw_snapshot.items()
    ):
        raise ValueError("measurement_rule_snapshot must map names to rule objects")
    snapshot = {
        key: MeasurementRule(**rule)
        for key, rule in raw_snapshot.items()
    }
    raw_question_snapshot = value.get("measurement_specification_snapshot", {})
    if not isinstance(raw_question_snapshot, dict) or not all(
        isinstance(key, str) and isinstance(specification, dict)
        for key, specification in raw_question_snapshot.items()
    ):
        raise ValueError(
            "measurement_specification_snapshot must map names to specification objects"
        )
    question_snapshot = {
        key: MeasurementSpecification(**specification)
        for key, specification in raw_question_snapshot.items()
    }
    return FalsificationSpecification(
        **{
            **value,
            "expressions": raw_expressions,
            "missing_measurement_constructs": _expect_string_list(
                value, "missing_measurement_constructs"
            ),
            "measurement_rule_snapshot": snapshot,
            "measurement_specification_snapshot": question_snapshot,
        }
    )


def _expect_dict(value: dict[str, object], key: str) -> dict[str, object]:
    item = value.get(key)
    if not isinstance(item, dict) or not all(isinstance(item_key, str) for item_key in item):
        raise ValueError(f"{key} must be an object with string keys")
    return item


def _expect_dict_list(value: dict[str, object], key: str) -> list[dict[str, object]]:
    items = value.get(key, [])
    if not isinstance(items, list):
        raise ValueError(f"{key} must be an array")
    result: list[dict[str, object]] = []
    for item in items:
        if not isinstance(item, dict) or not all(isinstance(item_key, str) for item_key in item):
            raise ValueError(f"every {key} item must be an object with string keys")
        result.append(item)
    return result


def _expect_int(value: dict[str, object], key: str) -> int:
    item = value.get(key)
    if not isinstance(item, int):
        raise ValueError(f"{key} must be an integer")
    return item


def _expect_nonnegative_int(value: dict[str, object], key: str) -> int:
    item = value.get(key, 0)
    if not isinstance(item, int) or item < 0:
        raise ValueError(f"{key} must be a non-negative integer")
    return item


def _expect_string_list(value: dict[str, object], key: str) -> list[str]:
    item = value.get(key, [])
    if not isinstance(item, list) or not all(isinstance(entry, str) for entry in item):
        raise ValueError(f"{key} must be an array of strings")
    return list(item)


def _expect_str(value: dict[str, object], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str):
        raise ValueError(f"{key} must be a string")
    return item
