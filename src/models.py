"""JSON-serializable models for the standalone Grounded Theory workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal


ComparisonDecision = Literal["SAME", "VARIATION", "NEW", "CONTRADICTION"]
OpenCodingDisposition = Literal[
    "SUPPORTS_EXISTING",
    "VARIATION",
    "BOUNDARY",
    "POSSIBLE_NEW",
    "NO_RELEVANT_MECHANISM",
]
GroundingStatus = Literal["well_grounded", "tentative", "insufficient_evidence"]
ProcessEdgeStatus = Literal["supported", "conditional", "tentative", "rejected"]
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
class DeferredRelationalRecord:
    """A Stage-2 record deliberately returned for a later, bounded pass."""

    record_id: str
    reason: str
    defer_count: int = 1
    last_deferred_at: str = field(default_factory=utc_now)


@dataclass(slots=True)
class Evidence:
    """A verbatim span whose presence is checked against its source record."""

    record_id: str
    text_span: str
    note: str = ""


@dataclass(slots=True)
class OpenCodingJudgment:
    """One auditable primary open-coding decision for one input record.

    This is deliberately separate from a batch-level concept update.  A record
    can support an already stable concept without requiring a definition edit,
    and that support must not disappear merely because another record supplied
    the representative update for the batch.
    """

    record_id: str
    disposition: OpenCodingDisposition
    rationale: str
    concept_id: str | None = None
    evidence: Evidence | None = None
    candidate_label: str | None = None


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
    # Revision-aware recency is used only to make an adaptive context view
    # deterministic. It never changes the analytic content of a concept.
    last_modified_revision: int = 0


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

    A process edge may reuse an already validated relation, or be independently
    reviewed against the complete corpus.  It retains the audit trail for that
    decision rather than relying on a semantically plausible sequence.
    """

    source_concept_id: str
    relationship: str
    target_concept_id: str
    supporting_relation_ids: list[str] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)
    supporting_record_ids: list[str] = field(default_factory=list)
    supporting_evidence: list[Evidence] = field(default_factory=list)
    reviewer_rationale: str = ""
    boundary_conditions: list[str] = field(default_factory=list)
    negative_evidence: list[Evidence] = field(default_factory=list)
    status: ProcessEdgeStatus = "tentative"


@dataclass(slots=True)
class ProcessEdgeReview:
    """Durable full-corpus review result, keyed by a directed concept pair.

    This is distinct from a formal ``Relationship``: it records the outcome of
    an independent process-specific arrow review so a later process proposing
    the same A -> B arrow does not spend another full-corpus review on it.
    """

    source_concept_id: str
    relationship: str
    target_concept_id: str
    supporting_record_ids: list[str] = field(default_factory=list)
    supporting_evidence: list[Evidence] = field(default_factory=list)
    reviewer_rationale: str = ""
    boundary_conditions: list[str] = field(default_factory=list)
    negative_evidence: list[Evidence] = field(default_factory=list)
    status: ProcessEdgeStatus = "tentative"


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
class PromptAdaptationConfig:
    """Conservative controls for Stage-2 strategy adaptation.

    The configuration deliberately says nothing about what counts as evidence:
    that remains in the frozen Stage-2 and grounding-review prompts.  All
    values are persisted with a project so a resumed run cannot silently adopt
    a new selection rule.
    """

    enabled: bool = False
    # ``offline`` records frozen reviewer audits during Stage 2 but never
    # interrupts the analytic run with extraction, replay, or blind A/B work.
    # Those audits can be evaluated as a separate experiment after the run.
    mode: Literal["online", "offline"] = "online"
    failure_window_batches: int = 5
    failure_trigger_batches: int = 3
    failure_min_instances: int = 2
    # Extracting a prompt-methodology audit is useful but does not change the
    # analytic state.  Sampling it by completed, independently reviewed Stage-2
    # batches keeps the expensive analytic path responsive.
    failure_extraction_interval_batches: int = 1
    validation_batch_count: int = 3
    pairwise_evaluation_rounds: int = 4
    # A production run may demonstrate the online mechanism once without
    # repeatedly diverting the main analytic schedule into replay evaluation.
    # A candidate can affect the active overlay only after blind A/B validation.
    max_online_adaptation_trials: int = 1
    require_grounding_non_regression: bool = True
    require_unsupported_inference_non_regression: bool = True
    allow_full_prompt_rewrite: bool = False
    mutable_scope: str = "stage2_strategy_only"
    random_seed: int = 42


@dataclass(slots=True)
class GroundedTheoryConfig:
    study_id: str
    research_question: str
    open_coding_batch_size: int = 24
    relational_batch_size: int = 48
    # A deferred Stage-2 record is retried in the normal comparison queue up
    # to this many times.  It then receives a one-record forced-resolution
    # transaction so coverage cannot silently loop forever.
    relational_max_defer_count: int = 2
    # ``legacy`` reproduces the prior bounded packet presentation. ``adaptive``
    # begins with a full inventory and advances only through explicit,
    # durable context-escalation steps when a packet cannot fit.
    context_policy: Literal["legacy", "adaptive"] = "adaptive"
    open_context_mode: Literal["full", "indexed_retrieval"] = "full"
    open_context_detailed_k: int = 12
    open_context_recent_concepts: int = 4
    context_index_compression_level: int = 0
    novelty_review_similarity_threshold: float = 0.28
    prompt_adaptation: PromptAdaptationConfig = field(default_factory=PromptAdaptationConfig)
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
    # One completed independent full-corpus process-edge review per directed
    # concept pair. Formal validated relationships remain the preferred reuse
    # source; this is the fallback when no such relationship exists.
    process_edge_review_cache: list[ProcessEdgeReview] = field(default_factory=list)
    memos: list[Memo] = field(default_factory=list)
    negative_cases: list[NegativeCase] = field(default_factory=list)
    theoretical_sampling_needs: list[TheoreticalSamplingNeed] = field(default_factory=list)
    integrated_theory: IntegratedTheory | None = None
    open_coded_record_ids: list[str] = field(default_factory=list)
    open_coding_judgments: list[OpenCodingJudgment] = field(default_factory=list)
    relationally_analyzed_record_ids: list[str] = field(default_factory=list)
    deferred_relational_records: list[DeferredRelationalRecord] = field(default_factory=list)
    pending_relational_payload: dict[str, Any] | None = None
    pending_relational_record_ids: list[str] = field(default_factory=list)
    # One independent reviewer result for each staged relationship or process edge.
    # This is durable so an interrupted review resumes at the next claim rather
    # than recreating a conversation or accepting a partly reviewed batch.
    pending_relational_review_results: list[dict[str, Any]] = field(default_factory=list)
    relational_validation_feedback: list[str] = field(default_factory=list)
    relational_validation_attempts: int = 0
    relational_validation_blocked: bool = False
    # Adaptation is deliberately kept outside the analytic inventories.  Its
    # only effect is which *strategy overlay* renders in a later Stage-2
    # packet; it may never alter evidence, reviews, schemas, or state rules.
    prompt_adaptation_state: dict[str, Any] = field(default_factory=dict)
    pending_relational_adaptation_context: dict[str, Any] | None = None
    pending_prompt_adaptation_failure_extraction: dict[str, Any] | None = None
