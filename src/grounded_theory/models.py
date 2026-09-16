"""JSON-serializable models for the standalone Grounded Theory workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal


ComparisonDecision = Literal["SAME", "VARIATION", "NEW", "CONTRADICTION"]
GroundingStatus = Literal["well_grounded", "tentative", "insufficient_evidence"]
IntegrationStatus = Literal[
    "integrated", "no_adequately_grounded_core_category"
]


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(slots=True)
class QualitativeRecord:
    id: str
    text: str
    source: str = ""


@dataclass(slots=True)
class Evidence:
    """A verbatim span whose presence is checked against its source record."""

    record_id: str
    text_span: str
    note: str = ""


@dataclass(slots=True)
class Concept:
    id: str
    label: str
    definition: str
    evidence: list[Evidence] = field(default_factory=list)
    definition_revisions: list[dict[str, Any]] = field(default_factory=list)
    variations: list[dict[str, Any]] = field(default_factory=list)
    negative_or_boundary_cases: list[Evidence] = field(default_factory=list)
    parent_category_id: str | None = None
    status: str = "active"
    level: Literal["concept", "category"] = "concept"
    child_concept_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Relationship:
    id: str
    source_concept_id: str
    relationship: str
    target_concept_id: str
    evidence: list[Evidence] = field(default_factory=list)
    comparative_basis: str = ""
    grounding_kind: Literal[
        "explicitly_expressed", "repeated_comparison", "tentative_theoretical_inference"
    ] = "tentative_theoretical_inference"
    grounding_explanation: str = ""
    conditions: list[str] = field(default_factory=list)
    variations: list[dict[str, Any]] = field(default_factory=list)
    negative_cases: list[Evidence] = field(default_factory=list)
    status: GroundingStatus = "tentative"
    memo_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ProcessEdge:
    """One reviewed arrow in a process pathway.

    A process is not accepted merely because it names supporting records.  Its
    arrows retain the direct relationships that warrant each transition.
    """

    source_concept_id: str
    relationship: str
    target_concept_id: str
    supporting_relation_ids: list[str] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Process:
    id: str
    label: str
    description: str
    conditions: list[str] = field(default_factory=list)
    actions_interactions: list[str] = field(default_factory=list)
    consequences: list[str] = field(default_factory=list)
    subsequent_changes: list[str] = field(default_factory=list)
    alternative_pathways: list[str] = field(default_factory=list)
    edges: list[ProcessEdge] = field(default_factory=list)
    supporting_relation_ids: list[str] = field(default_factory=list)
    supporting_record_ids: list[str] = field(default_factory=list)
    negative_cases: list[Evidence] = field(default_factory=list)
    status: GroundingStatus = "tentative"


@dataclass(slots=True)
class Memo:
    id: str
    topic: str
    analytic_observation: str
    supporting_evidence: list[Evidence] = field(default_factory=list)
    comparisons: str = ""
    tentative_interpretation: str = ""
    negative_or_contradictory_cases: list[Evidence] = field(default_factory=list)
    uncertainties: str = ""
    questions_for_further_analysis: list[str] = field(default_factory=list)
    linked_concept_ids: list[str] = field(default_factory=list)
    linked_relation_ids: list[str] = field(default_factory=list)
    linked_process_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class NegativeCase:
    id: str
    target_type: Literal["concept", "relationship", "process"]
    target_id: str
    description: str
    evidence: list[Evidence] = field(default_factory=list)


@dataclass(slots=True)
class TheoreticalSamplingNeed:
    id: str
    target: str
    reason: str
    evidence_needed: str
    evidence: list[Evidence] = field(default_factory=list)
    status: str = "unresolved"


@dataclass(slots=True)
class TheoreticalProposition:
    id: str
    statement: str
    status: Literal["well_grounded", "tentative", "unresolved"]
    process_ids: list[str] = field(default_factory=list)
    relation_ids: list[str] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)


@dataclass(slots=True)
class IntegratedTheory:
    status: IntegrationStatus
    core_category_id: str | None
    account: str
    id: str = "integrated_theory_001"
    propositions: list[TheoreticalProposition] = field(default_factory=list)
    alternative_pathways: list[str] = field(default_factory=list)
    negative_case_ids: list[str] = field(default_factory=list)
    unresolved_questions: list[str] = field(default_factory=list)


@dataclass(slots=True)
class GroundedTheoryConfig:
    study_id: str
    research_question: str
    open_coding_batch_size: int = 24
    relational_batch_size: int = 48
    created_at: str = field(default_factory=utc_now)


@dataclass(slots=True)
class GroundedTheoryState:
    schema_version: int
    revision: int
    config: GroundedTheoryConfig
    analysis_metadata: dict[str, Any] = field(default_factory=dict)
    records: list[QualitativeRecord] = field(default_factory=list)
    concepts: list[Concept] = field(default_factory=list)
    relationships: list[Relationship] = field(default_factory=list)
    processes: list[Process] = field(default_factory=list)
    memos: list[Memo] = field(default_factory=list)
    negative_cases: list[NegativeCase] = field(default_factory=list)
    theoretical_sampling_needs: list[TheoreticalSamplingNeed] = field(default_factory=list)
    integrated_theory: IntegratedTheory | None = None
    open_coded_record_ids: list[str] = field(default_factory=list)
    relationally_analyzed_record_ids: list[str] = field(default_factory=list)
    pending_relational_payload: dict[str, Any] | None = None
    pending_relational_record_ids: list[str] = field(default_factory=list)
    # One independent reviewer result for each staged relationship or process edge.
    # This is durable so an interrupted review resumes at the next claim rather
    # than recreating a conversation or accepting a partly reviewed batch.
    pending_relational_review_results: list[dict[str, Any]] = field(default_factory=list)
    relational_validation_feedback: list[str] = field(default_factory=list)
    relational_validation_attempts: int = 0
    relational_validation_blocked: bool = False
