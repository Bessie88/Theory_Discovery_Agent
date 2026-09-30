"""Persistent state machine for the standalone Grounded Theory front end."""

from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from .integration import apply_integration
from .context_management import (
    CONTEXT_AUDIT_KEY,
    DETAIL_K_STEPS,
    OPEN_CONTEXT_TARGET_TOKENS,
    audit_context,
    complete_identifier_catalogue,
    compact_text,
    detailed_concept_view,
    open_context_packet,
    rank_relevant_concepts,
)
from .models import GroundedTheoryConfig, GroundedTheoryState, QualitativeRecord, utc_now
from .prompt_adaptation import (
    CRITIC_ACTION,
    EDITOR_ACTION,
    FAILURE_EXTRACTION_ACTION,
    PAIRWISE_DIMENSIONS,
    VALIDATION_ACTION,
    active_overlay,
    append_failure_memory,
    begin_candidate_if_triggered,
    candidate_overlay,
    canonical_hash,
    critic_schema,
    editor_schema,
    enabled as prompt_adaptation_enabled,
    failure_extraction_schema,
    initialize_state as initialize_prompt_adaptation_state,
    normalize_critic,
    normalize_failure_extraction,
    normalize_patch,
    stage2_prompt,
    aggregate_validation,
)
from .open_coding import (
    OPEN_CODING_COVERAGE_ACTION,
    apply_open_coding,
    apply_open_coding_coverage_audit,
)
from .packets import (
    integration_output_schema,
    load_prompt,
    open_coding_coverage_audit_output_schema,
    open_coding_output_schema,
    process_edge_grounding_review_output_schema,
    relational_grounding_review_output_schema,
    relational_output_schema,
)
from .persistence import (
    EVENTS_FILE,
    SCHEMA_VERSION,
    STATE_FILE,
    record_from_value,
    render_analysis_report,
    state_from_dict,
    state_to_dict,
    validate_records,
    write_json,
)
from .relational import apply_relational_analysis, apply_relational_grounding_validation
from .review import (
    apply_review_decisions,
    build_relational_review_material,
    next_relational_review_material,
    normalize_claim_review,
)
from .socrates_context import shared_case_context_records
from .token_budget import packet_token_estimate
from .validation import object_list, remaining_record_ids


PACKET_EVIDENCE_PER_CONCEPT = 2
PACKET_CONCEPT_HISTORY_ITEMS = 1
PACKET_MEMOS = 4
OPEN_CODING_RELEVANT_MEMOS = 4
OPEN_CODING_RECENT_MEMOS = 2
PACKET_MEMO_EVIDENCE = 1
PACKET_RELATION_EVIDENCE = 2
PACKET_RELATION_HISTORY_ITEMS = 1
PACKET_PROCESS_ITEMS = 3
PACKET_PROCESS_EDGE_ITEMS = 2
PACKET_NEGATIVE_CASES = 8
PACKET_NEGATIVE_CASE_EVIDENCE = 2
# Leave headroom under the 24k runner limit for stateless request framing and
# future inventory growth. The compact structural-retrieval view keeps the
# ordinary eight-record SoCRATES batch below this target; persisted evidence
# is never truncated or deleted.
RELATIONAL_PACKET_TARGET_TOKENS = 23_000
ADAPTATION_FAILURE_EXTRACTION_TARGET_TOKENS = 18000
ADAPTATION_CRITIC_TARGET_TOKENS = 18000
ADAPTATION_EDITOR_TARGET_TOKENS = 14000
ADAPTATION_EXAMPLE_DETAIL_STEPS = (12, 8, 4, 2, 0)
COMPACT_INVENTORY_ITEMS = 12
COMPACT_PROCESS_EDGE_ITEMS = 8
COMPACT_TEXT_CHARS = 480
# Memos are analytic working notes rather than formal theory entities.  A
# complete ID index plus deterministic detailed retrieval preserves their
# auditability without allowing an unbounded collection of memo topics to
# crowd out the current incident and its evidence.
STAGE2_RETRIEVED_CONCEPTS = 6
STAGE2_RECENT_CONCEPTS = 2
STAGE2_RETRIEVED_RELATIONS = 8
STAGE2_RETRIEVED_PROCESSES = 4
STAGE2_RETRIEVED_NEGATIVE_CASES = 4
class GroundedTheoryProject:
    """Durable Corbin--Strauss stages with feedback from Stage 2 into Stage 1."""

    def __init__(self, root: str | Path, state: GroundedTheoryState):
        self.root = Path(root)
        self.state = state

    @classmethod
    def create(
        cls,
        root: str | Path,
        *,
        study_id: str,
        research_question: str,
        records: list[dict[str, str] | QualitativeRecord],
        open_coding_batch_size: int = 24,
        relational_batch_size: int = 48,
    ) -> "GroundedTheoryProject":
        root_path = Path(root)
        if (root_path / STATE_FILE).exists():
            raise FileExistsError(f"Grounded Theory state already exists: {root_path / STATE_FILE}")
        if not study_id.strip() or not research_question.strip():
            raise ValueError("study_id and research_question are required")
        if open_coding_batch_size < 1 or relational_batch_size < 1:
            raise ValueError("batch sizes must be positive")
        parsed_records = [record_from_value(item) for item in records]
        validate_records(parsed_records)
        state = GroundedTheoryState(
            schema_version=SCHEMA_VERSION,
            revision=0,
            config=GroundedTheoryConfig(
                study_id=study_id.strip(),
                research_question=research_question.strip(),
                open_coding_batch_size=open_coding_batch_size,
                relational_batch_size=relational_batch_size,
            ),
            analysis_metadata={
                "methodology": "Corbin & Strauss Grounded Theory, 4th edition",
                "stage_order": [
                    "open_coding",
                    "relational_process_analysis",
                    "theoretical_integration",
                ],
                "stage_2_revises_concept_inventory": True,
                "constant_comparison": True,
                "memo_writing": True,
            },
            records=parsed_records,
        )
        project = cls(root_path, state)
        project._write_state()
        return project

    @classmethod
    def load(cls, root: str | Path) -> "GroundedTheoryProject":
        root_path = Path(root)
        path = root_path / STATE_FILE
        if not path.exists():
            raise FileNotFoundError(f"No Grounded Theory state: {path}")
        return cls(root_path, state_from_dict(json.loads(path.read_text(encoding="utf-8"))))

    def status(self) -> dict[str, Any]:
        adaptation = self.state.prompt_adaptation_state
        pending_candidate = adaptation.get("pending_candidate") if adaptation else None
        return {
            "schema_version": self.state.schema_version,
            "revision": self.state.revision,
            "counts": {
                "records": len(self.state.records),
                "open_coded_records": len(self.state.open_coded_record_ids),
                "open_coding_judgments": len(self.state.open_coding_judgments),
                "relationally_analyzed_records": len(self.state.relationally_analyzed_record_ids),
                "concepts": len(self.state.concepts),
                "relationships": len(self.state.relationships),
                "processes": len(self.state.processes),
                "process_edge_review_cache": len(self.state.process_edge_review_cache),
                "deferred_relational_records": len(self.state.deferred_relational_records),
                "memos": len(self.state.memos),
                "negative_cases": len(self.state.negative_cases),
                "sampling_needs": len(self.state.theoretical_sampling_needs),
            },
            "prompt_adaptation": {
                "enabled": self.state.config.prompt_adaptation.enabled,
                "mode": self.state.config.prompt_adaptation.mode,
                "active_strategy_version": (
                    adaptation.get("active_overlay", {}).get("version") if adaptation else None
                ),
                "failure_batches": len(adaptation.get("failure_memory", [])) if adaptation else 0,
                "offline_failure_audits": len(adaptation.get("offline_failure_audits", [])) if adaptation else 0,
                "pending_phase": pending_candidate.get("phase") if isinstance(pending_candidate, dict) else None,
                "awaiting_failure_extraction": self.state.pending_prompt_adaptation_failure_extraction is not None,
            },
            "next_action": self.next_action(),
        }

    def update_batch_sizes(
        self,
        *,
        open_coding_batch_size: int | None = None,
        relational_batch_size: int | None = None,
    ) -> bool:
        """Persist an audited batch-size change before a resumed run."""
        requested = {
            "open_coding_batch_size": open_coding_batch_size,
            "relational_batch_size": relational_batch_size,
        }
        changes = {
            name: value for name, value in requested.items()
            if value is not None and value != getattr(self.state.config, name)
        }
        if not changes:
            return False
        if any(not isinstance(value, int) or value < 1 for value in changes.values()):
            raise ValueError("batch sizes must be positive integers")
        previous = asdict(self.state.config)
        for name, value in changes.items():
            setattr(self.state.config, name, value)
        self._commit(
            "analysis_configuration_updated",
            {"previous": previous, "current": asdict(self.state.config)},
        )
        return True

    def update_context_configuration(
        self,
        *,
        context_policy: str | None = None,
        open_context_detailed_k: int | None = None,
        open_context_recent_concepts: int | None = None,
        novelty_review_similarity_threshold: float | None = None,
    ) -> bool:
        """Persist an intentional context-policy setting change.

        This does not transform the analytic state. It only changes how the
        next stateless packet is rendered and leaves an event for reproducible
        reruns.
        """
        requested = {
            "context_policy": context_policy,
            "open_context_detailed_k": open_context_detailed_k,
            "open_context_recent_concepts": open_context_recent_concepts,
            "novelty_review_similarity_threshold": novelty_review_similarity_threshold,
        }
        if context_policy is not None and context_policy not in {"legacy", "adaptive"}:
            raise ValueError("context_policy must be legacy or adaptive")
        if open_context_detailed_k is not None and open_context_detailed_k not in DETAIL_K_STEPS:
            raise ValueError(f"open_context_detailed_k must be one of {DETAIL_K_STEPS}")
        if open_context_recent_concepts is not None and open_context_recent_concepts < 0:
            raise ValueError("open_context_recent_concepts must be non-negative")
        if novelty_review_similarity_threshold is not None and not 0 <= novelty_review_similarity_threshold <= 1:
            raise ValueError("novelty_review_similarity_threshold must be between 0 and 1")
        changes = {
            key: value for key, value in requested.items()
            if value is not None and value != getattr(self.state.config, key)
        }
        if not changes:
            return False
        previous = asdict(self.state.config)
        for key, value in changes.items():
            setattr(self.state.config, key, value)
        if "context_policy" in changes:
            self.state.config.open_context_mode = "full"
            self.state.config.open_context_detailed_k = DETAIL_K_STEPS[0]
            self.state.config.context_index_compression_level = 0
        self._commit("context_configuration_updated", {"previous": previous, "current": asdict(self.state.config)})
        return True

    def update_prompt_adaptation_configuration(self, updates: dict[str, Any]) -> bool:
        """Persist one complete, conservative adaptation configuration change.

        The mutable strategy is enabled only by an explicit configuration
        update.  Enabling freezes the exact rendered Stage-2 base prompt into
        both state and an auditable artifact; later source-prompt edits cannot
        silently affect a resumed adaptive project.
        """
        allowed = set(asdict(self.state.config.prompt_adaptation))
        unknown = set(updates) - allowed
        if unknown:
            raise ValueError(f"unknown prompt-adaptation configuration: {sorted(unknown)}")
        config = self.state.config.prompt_adaptation
        proposed = {**asdict(config), **updates}
        if type(proposed["enabled"]) is not bool:
            raise ValueError("prompt_adaptation.enabled must be boolean")
        if proposed["mode"] not in {"online", "offline"}:
            raise ValueError("prompt_adaptation.mode must be online or offline")
        for key in (
            "failure_window_batches",
            "failure_trigger_batches",
            "failure_min_instances",
            "failure_extraction_interval_batches",
            "validation_batch_count",
            "pairwise_evaluation_rounds",
            "max_online_adaptation_trials",
        ):
            if type(proposed[key]) is not int or proposed[key] < 1:
                raise ValueError(f"prompt_adaptation.{key} must be a positive integer")
        if proposed["failure_trigger_batches"] > proposed["failure_window_batches"]:
            raise ValueError("prompt_adaptation.failure_trigger_batches cannot exceed its window")
        if type(proposed["random_seed"]) is not int:
            raise ValueError("prompt_adaptation.random_seed must be an integer")
        if proposed["mutable_scope"] != "stage2_strategy_only" or proposed["allow_full_prompt_rewrite"]:
            raise ValueError("v1 permits only stage2_strategy_only and never a full prompt rewrite")
        for key in ("require_grounding_non_regression", "require_unsupported_inference_non_regression"):
            if type(proposed[key]) is not bool:
                raise ValueError(f"prompt_adaptation.{key} must be boolean")
        if proposed == asdict(config):
            return False
        previous = asdict(config)
        for key, value in proposed.items():
            setattr(config, key, value)
        if config.enabled:
            initialize_prompt_adaptation_state(self.state)
            self._write_frozen_stage2_prompt()
        self._commit(
            "prompt_adaptation_configuration_updated",
            {"previous": previous, "current": asdict(config)},
        )
        return True

    def escalate_open_context(self, *, after_batch_reduction: bool = False) -> bool:
        """Advance one documented adaptive escalation step, never silently.

        Callers first advance full->indexed, then reduce detail retrieval and
        finally compact the global index presentation. Raw batch size is never
        a context-recovery mechanism: every index entry remains represented by
        an exact ID, label, or identifier range at each escalation level.

        ``after_batch_reduction`` is retained as a no-op compatibility keyword
        for already submitted recovery scripts. New callers must not reduce a
        raw batch before entering the final identifier-retrieval tier.
        """
        config = self.state.config
        if config.context_policy != "adaptive":
            return False
        previous = asdict(config)
        if config.open_context_mode == "full":
            config.open_context_mode = "indexed_retrieval"
        elif config.open_context_detailed_k > 0:
            current = config.open_context_detailed_k
            config.open_context_detailed_k = next(
                (value for value in DETAIL_K_STEPS if value < current), 0
            )
        elif config.context_index_compression_level < 3:
            config.context_index_compression_level += 1
        else:
            return False
        self._commit(
            "open_context_escalated",
            {"previous": previous, "current": asdict(config), "after_batch_reduction": False},
        )
        return True

    def discard_blocked_relational_batch(self) -> dict[str, Any]:
        """Recover a legacy state that was blocked before skip-on-rejection."""
        if not self.state.relational_validation_blocked:
            raise ValueError("there is no blocked relational batch to discard")
        record_ids = remaining_record_ids(
            self.state.records, self.state.relationally_analyzed_record_ids
        )[: self.state.config.relational_batch_size]
        if not record_ids:
            raise ValueError("the blocked relational batch has no remaining records")
        open_coded = set(self.state.open_coded_record_ids)
        if any(record_id not in open_coded for record_id in record_ids):
            raise ValueError("a blocked relational batch must already be open coded")
        feedback = list(self.state.relational_validation_feedback)
        self.state.relationally_analyzed_record_ids.extend(record_ids)
        self.state.pending_relational_payload = None
        self.state.pending_relational_record_ids = []
        self.state.pending_relational_review_results = []
        self.state.relational_validation_feedback = []
        self.state.relational_validation_attempts = 0
        self.state.relational_validation_blocked = False
        event = {
            "record_ids": record_ids,
            "issues": feedback,
            "reason": "Recovered legacy blocked state by discarding its twice-rejected, uncommitted relational candidate batch.",
        }
        self._commit("relational_batch_discarded_after_repeated_grounding_failure", event)
        return {"discarded": event, "status": self.status()}

    def skip_failed_stage(self, action: str, reason: str) -> dict[str, Any]:
        """Record and skip one exhausted non-terminal agent stage."""
        from .recovery import skip_failed_stage

        return skip_failed_stage(self, action, reason)

    def waive_remaining_open_coding_coverage(self, reason: str) -> dict[str, Any]:
        """Allow Stage 2 with explicitly documented missing Stage-1 judgments.

        This is an operator-only exception for exhausted coverage-audit batches.
        It never fabricates judgments, removes rejected candidate artifacts, or
        changes the underlying records.  The waiver is persisted in public
        analysis metadata so downstream packets and exports disclose it.
        """
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("a non-empty coverage-waiver reason is required")
        existing = self.state.analysis_metadata.get("open_coding_coverage_waiver")
        if existing is not None:
            raise ValueError("open-coding coverage has already been waived")
        judged = {item.record_id for item in self.state.open_coding_judgments}
        missing = [record.id for record in self.state.records if record.id not in judged]
        if not missing:
            raise ValueError("there are no missing open-coding judgments to waive")
        waiver = {
            "record_ids": missing,
            "reason": reason.strip(),
            "created_at": utc_now(),
            "judgments_present": len(judged),
            "judgments_missing": len(missing),
            "rule": (
                "Stage 2 may analyze these records, but their missing Stage-1 "
                "judgments must remain disclosed in all downstream interpretation."
            ),
        }
        self.state.analysis_metadata["open_coding_coverage_waiver"] = waiver
        self._commit("open_coding_coverage_waived_by_operator", waiver)
        return {"waiver": waiver, "status": self.status()}

    def requeue_skipped_relational_batches(self) -> dict[str, Any]:
        """Restore skipped Stage 2 records after increasing runner capacity."""
        from .recovery import requeue_skipped_relational_batches

        return requeue_skipped_relational_batches(self)

    def restore_relational_coverage_after_unintended_requeue(self, reason: str) -> dict[str, Any]:
        """Restore terminal Stage-2 coverage before an integration resume.

        This is an explicit operator recovery for a legacy runner that
        requeued auditable skips while resuming final integration.  It never
        commits an unreviewed candidate: all records return to their prior
        terminal coverage status and the restoration is durably audited.
        """
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("a non-empty restoration reason is required")
        if self.state.pending_relational_payload is not None:
            raise ValueError("cannot restore relational coverage while a candidate is pending review")
        record_ids = [record.id for record in self.state.records]
        open_coded = set(self.state.open_coded_record_ids)
        if any(record_id not in open_coded for record_id in record_ids):
            raise ValueError("only fully open-coded records can be restored for integration")
        previous = set(self.state.relationally_analyzed_record_ids)
        restored = [record_id for record_id in record_ids if record_id not in previous]
        if not restored:
            raise ValueError("relational coverage is already complete")
        self.state.relationally_analyzed_record_ids = record_ids
        event = {
            "record_ids": restored,
            "reason": reason.strip(),
            "rule": (
                "Restored prior terminal Stage-2 coverage after an unintended "
                "automatic requeue during integration resume; no uncommitted "
                "candidate was promoted."
            ),
        }
        self._commit("relational_coverage_restored_after_unintended_requeue", event)
        return {"restored": event, "status": self.status()}

    def repair_pending_process_edge_review_evidence(self) -> dict[str, Any]:
        """Remove only old pending edge reviews whose quotes cannot be verified.

        New reviews are aligned when submitted.  This repair is for the small
        window before that check existed: it keeps every valid relationship and
        process-edge decision, but returns an invalid edge alone to the normal
        full-corpus reviewer instead of letting the later process commit fail.
        """
        if self.state.pending_relational_payload is None:
            return {"repaired": [], "status": self.status()}
        material = build_relational_review_material(
            self.state, self.state.pending_relational_payload
        )
        claims = {claim["claim_id"]: claim for claim in material["review_claims"]}
        record_texts = {record.id: record.text for record in self.state.records}
        retained: list[dict[str, Any]] = []
        removed: list[dict[str, str]] = []
        changed = False
        for result in self.state.pending_relational_review_results:
            claim_id = result.get("claim_id")
            claim = claims.get(claim_id)
            if not isinstance(claim_id, str) or claim is None or claim.get("claim_type") != "process_edge":
                retained.append(result)
                continue
            try:
                normalized = normalize_claim_review(
                    result,
                    claim,
                    allowed_record_ids=set(record_texts),
                    review_record_texts=record_texts,
                )
            except ValueError as error:
                removed.append({"claim_id": claim_id, "reason": str(error)})
                changed = True
                continue
            retained.append(normalized)
            changed = changed or normalized != result
        if not changed:
            return {"repaired": [], "status": self.status()}
        self.state.pending_relational_review_results = retained
        event = {
            "removed_process_edge_review_claims": removed,
            "retained_review_count": len(retained),
            "reason": "Removed only pending process-edge reviews with unverifiable evidence quotes; they will be independently reviewed again.",
        }
        self._commit("pending_process_edge_review_evidence_repaired", event)
        return {"repaired": removed, "status": self.status()}

    def requeue_invalid_pending_relational_candidate(self) -> dict[str, Any]:
        """Return a legacy staged candidate to Stage 2 if new checks reject it.

        A candidate is never accepted merely because it reached review under an
        older validator.  If its process edge points to a differently directed
        relationship update, discard only that uncommitted candidate and leave
        its original record IDs available for a fresh relational analysis.
        """
        payload = self.state.pending_relational_payload
        if payload is None:
            return {"requeued": None, "status": self.status()}
        trial = deepcopy(self.state)
        try:
            apply_relational_analysis(
                trial,
                payload,
                {"record_ids": list(self.state.pending_relational_record_ids)},
                allow_unreviewed_processes=True,
            )
        except ValueError as error:
            if "process edge supporting relationship must have the same direction" not in str(error):
                return {"requeued": None, "status": self.status()}
            record_ids = list(self.state.pending_relational_record_ids)
            self.state.pending_relational_payload = None
            self.state.pending_relational_record_ids = []
            self.state.pending_relational_review_results = []
            self.state.relational_validation_feedback = []
            self.state.relational_validation_attempts = 0
            self.state.relational_validation_blocked = False
            event = {
                "record_ids": record_ids,
                "reason": str(error),
                "action": "Returned the invalid, uncommitted candidate to relational analysis; no accepted findings were removed.",
            }
            self._commit("invalid_pending_relational_candidate_requeued", event)
            return {"requeued": event, "status": self.status()}
        return {"requeued": None, "status": self.status()}

    def next_action(self) -> dict[str, Any]:
        if self.state.relational_validation_blocked:
            return {
                "action": "relational_validation_blocked",
                "reason": "an independent grounding review rejected the same relational batch twice; revise it instead of accepting an unsupported relation",
                "feedback": list(self.state.relational_validation_feedback),
            }
        if (
            prompt_adaptation_enabled(self.state)
            and self.state.config.prompt_adaptation.mode == "online"
        ):
            pending_extraction = self.state.pending_prompt_adaptation_failure_extraction
            if pending_extraction is not None:
                return {
                    "action": FAILURE_EXTRACTION_ACTION,
                    "batch_id": pending_extraction["batch_id"],
                    "reason": "a completed, independently reviewed Stage-2 batch needs frozen methodological failure extraction",
                }
            candidate = self.state.prompt_adaptation_state.get("pending_candidate")
            if isinstance(candidate, dict):
                phase = candidate.get("phase")
                if phase == "critic":
                    return {"action": CRITIC_ACTION, "reason": "recurring methodological failures need a strategy-only diagnosis"}
                if phase == "editor":
                    return {"action": EDITOR_ACTION, "reason": "a prompt-related diagnosis needs one minimal strategy patch"}
                if phase == "validation":
                    return {"action": VALIDATION_ACTION, "reason": "a candidate strategy requires isolated held-out replay validation"}
        remaining_open = remaining_record_ids(self.state.records, self.state.open_coded_record_ids)
        if remaining_open:
            return {
                "action": "open_coding",
                "record_ids": remaining_open[: self.state.config.open_coding_batch_size],
                "reason": "records need open coding and comparison with the live concept inventory",
            }
        judged = {item.record_id for item in self.state.open_coding_judgments}
        coverage_waiver = self.state.analysis_metadata.get("open_coding_coverage_waiver")
        waived_ids = (
            set(coverage_waiver.get("record_ids", []))
            if isinstance(coverage_waiver, dict)
            else set()
        )
        missing_judgments = [
            record.id for record in self.state.records
            if record.id not in judged and record.id not in waived_ids
        ]
        if missing_judgments:
            return {
                "action": OPEN_CODING_COVERAGE_ACTION,
                "record_ids": missing_judgments[: self.state.config.open_coding_batch_size],
                "reason": (
                    "every open-coded record needs one durable, traceable open-coding "
                    "judgment before relationship analysis can continue"
                ),
            }
        if self.state.pending_relational_payload is not None:
            review_material = next_relational_review_material(
                self.state,
                self.state.pending_relational_payload,
                self.state.pending_relational_review_results,
            )
            if review_material is None:
                raise RuntimeError("pending relational review has no remaining claim")
            return {
                "action": "validate_relational_grounding",
                "record_ids": list(self.state.pending_relational_record_ids),
                "reason": (
                    "one candidate relationship or process arrow requires an independent "
                    f"counterevidence review before commit: {review_material['review_claim']['claim_id']}"
                ),
            }
        relational_action = self._next_relational_action()
        if relational_action is not None:
            return relational_action
        if self.state.integrated_theory is None:
            return {
                "action": "theoretical_integration",
                "reason": "concept, relationship, process, memo, and negative-case inventories are ready for integration",
            }
        return {"action": "complete", "reason": "Grounded Theory front-end analysis is complete"}

    def _next_relational_action(self) -> dict[str, Any] | None:
        """Select queued deferrals first, with a one-record forced fallback."""
        remaining = remaining_record_ids(
            self.state.records, self.state.relationally_analyzed_record_ids
        )
        if not remaining:
            return None
        remaining_ids = set(remaining)
        queued = [
            item for item in self.state.deferred_relational_records
            if item.record_id in remaining_ids
        ]
        max_defers = self.state.config.relational_max_defer_count
        forced = [item for item in queued if item.defer_count >= max_defers]
        if forced:
            item = forced[0]
            return {
                "action": "relational_process_analysis",
                "record_ids": [item.record_id],
                "forced_resolution_record_ids": [item.record_id],
                "deferred_record_context": [{
                    "record_id": item.record_id,
                    "reason": item.reason,
                    "defer_count": item.defer_count,
                }],
                "reason": (
                    "a repeatedly deferred record requires forced resolution; it must be "
                    "processed with either grounded candidates or an explicit empty finding"
                ),
            }
        queued_ids = [item.record_id for item in queued]
        queue_set = set(queued_ids)
        ordered = [*queued_ids, *[record_id for record_id in remaining if record_id not in queue_set]]
        # Throughput is fixed by the study configuration. Context management
        # must compact the historical presentation and retrieval detail; it
        # must never quietly shorten an ordinary Stage-2 batch because one
        # packet is large. A final exact Qwen-token preflight refuses an
        # impossible request before network I/O without changing this batch.
        record_ids = ordered[: self.state.config.relational_batch_size]
        queued_context = [
            {"record_id": item.record_id, "reason": item.reason, "defer_count": item.defer_count}
            for item in queued if item.record_id in set(record_ids)
        ]
        return {
            "action": "relational_process_analysis",
            "record_ids": record_ids,
            "deferred_record_context": queued_context,
            "reason": "coded records need relational and process analysis against the live, Stage-2-revisable concept inventory",
        }

    def task_packet(self) -> dict[str, Any]:
        action_info = self.next_action()
        action = action_info["action"]
        packet = self._base_task_packet(action)
        if action == "open_coding":
            return self._open_packet(packet, action_info["record_ids"])
        if action == OPEN_CODING_COVERAGE_ACTION:
            return self._open_coding_coverage_packet(packet, action_info["record_ids"])
        if action == "relational_process_analysis":
            return self._relational_packet(packet, action_info)
        if action == FAILURE_EXTRACTION_ACTION:
            return self._failure_extraction_packet(packet)
        if action == CRITIC_ACTION:
            return self._critic_packet(packet)
        if action == EDITOR_ACTION:
            return self._editor_packet(packet)
        if action == VALIDATION_ACTION:
            return self._validation_packet(packet)
        if action == "validate_relational_grounding":
            return self._grounding_review_packet(packet, action_info["record_ids"])
        if action == "theoretical_integration":
            packet.update(
                {
                    "concept_inventory": self._packet_concept_inventory(),
                    "relation_inventory": self._packet_relation_inventory(),
                    "process_inventory": self._packet_process_inventory(),
                    "memos": self._packet_memos(),
                    "negative_cases": self._packet_negative_cases(),
                    "prompt": load_prompt("gt-03-theoretical-integration.md"),
                    "expected_output": integration_output_schema(),
                }
            )
        return packet

    def _base_task_packet(self, action: str) -> dict[str, Any]:
        """Build the common model envelope without inspecting next action.

        Stage 2 uses this same envelope to choose a packet-safe raw-record
        chunk before the model is called. Keeping the probe and real packet
        identical prevents one unusually long incident from permanently
        shrinking every later relational batch.
        """
        return {
            # Private metadata pins a context view to the accepted state that
            # rendered it. It is copied to the receipt, never sent to Qwen.
            "_context_snapshot": {
                "id": f"state-revision-{self.state.revision:08d}",
                "revision": self.state.revision,
            },
            "action": action,
            "study": asdict(self.state.config),
            "analysis_metadata": deepcopy(self.state.analysis_metadata),
            "rule": "Complete only this stage. Findings must remain grounded in supplied records; submit evidence using a supplied record ID and a short quote; do not use external knowledge.",
            "validation": {
                "evidence_rule": "Every evidence item must use a supplied record ID and a short quote.",
                "comparison_values": ["SAME", "VARIATION", "NEW", "CONTRADICTION"],
                "no_cooccurrence_rule": "Co-occurrence alone is not a relationship.",
                "stage_feedback_rule": "Stage 2 may submit evidence-grounded concept_updates to revise the live Stage 1 concept/category inventory.",
            },
        }

    def submit_agent_result(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("agent result must be a JSON object")
        action = str(self.next_action()["action"])
        if action == "complete":
            raise ValueError("Grounded Theory analysis is already complete")
        if action == "relational_validation_blocked":
            raise ValueError("relational grounding is blocked for review; do not bypass its feedback")
        trial = deepcopy(self.state)
        pending_adaptation_context = deepcopy(self.state.pending_relational_adaptation_context)
        pending_relational_payload = deepcopy(self.state.pending_relational_payload)
        if action == "open_coding":
            event = apply_open_coding(trial, payload, self.next_action())
        elif action == OPEN_CODING_COVERAGE_ACTION:
            event = apply_open_coding_coverage_audit(trial, payload, self.next_action())
        elif action == "relational_process_analysis":
            event, action = self._stage_or_commit_relations(trial, payload, action)
        elif action == "validate_relational_grounding":
            event = apply_relational_grounding_validation(trial, payload)
            if (
                event.get("review_status") == "COMPLETE"
                and prompt_adaptation_enabled(self.state)
                and pending_adaptation_context is not None
                and pending_relational_payload is not None
            ):
                self._queue_failure_extraction(
                    trial,
                    pending_adaptation_context,
                    pending_relational_payload,
                    event,
                )
        elif action == FAILURE_EXTRACTION_ACTION:
            event = self._accept_failure_extraction(trial, payload)
        elif action == CRITIC_ACTION:
            event = self._accept_prompt_critic(trial, payload)
        elif action == EDITOR_ACTION:
            event = self._accept_prompt_editor(trial, payload)
        elif action == VALIDATION_ACTION:
            raise ValueError("prompt-adaptation validation must be executed by the isolated runner")
        elif action == "theoretical_integration":
            event = apply_integration(trial, payload)
        else:
            raise ValueError(f"unsupported action: {action}")
        self.state = trial
        self._commit(action, event)
        return {"accepted_action": action, "status": self.status()}

    def submit_prompt_adaptation_validation(self, validation: list[dict[str, Any]]) -> dict[str, Any]:
        """Atomically accept or roll back one already-blinded candidate overlay."""
        if not prompt_adaptation_enabled(self.state):
            raise ValueError("prompt adaptation is not enabled")
        candidate = self.state.prompt_adaptation_state.get("pending_candidate")
        if not isinstance(candidate, dict) or candidate.get("phase") != "validation":
            raise ValueError("there is no candidate strategy awaiting validation")
        expected = list(candidate["validation_batches"])
        if [item.get("batch_id") for item in validation] != expected:
            raise ValueError("validation results must cover the fixed selected batches in stable order")
        normalized: list[dict[str, Any]] = []
        for item, snapshot_id in zip(validation, candidate["validation_snapshot_ids"]):
            if item.get("snapshot_id") != snapshot_id or not isinstance(item.get("rounds"), list):
                raise ValueError("validation result does not match its frozen snapshot")
            votes = {name: {"candidate": 0, "parent": 0, "tie": 0} for name in PAIRWISE_DIMENSIONS}
            overall_votes = {"candidate": 0, "parent": 0, "tie": 0}
            for round_result in item["rounds"]:
                if not isinstance(round_result, dict):
                    raise ValueError("a pairwise round must be an object")
                overall = round_result.get("overall")
                dimensions = round_result.get("dimensions")
                if overall not in overall_votes or not isinstance(dimensions, dict):
                    raise ValueError("a pairwise round has invalid mapped votes")
                overall_votes[overall] += 1
                for name, side in dimensions.items():
                    if name not in votes or side not in votes[name]:
                        raise ValueError("a pairwise round names an invalid methodological dimension")
                    votes[name][side] += 1
            overall = (
                "candidate" if overall_votes["candidate"] > overall_votes["parent"]
                else "parent" if overall_votes["parent"] > overall_votes["candidate"]
                else "inconclusive"
            )
            normalized.append({
                "batch_id": item["batch_id"],
                "snapshot_id": snapshot_id,
                "rounds": deepcopy(item["rounds"]),
                "overall": overall,
                "pairwise_overall_votes": overall_votes,
                "dimension_votes": votes,
            })
        trial = deepcopy(self.state)
        decision = aggregate_validation(normalized, trial.config.prompt_adaptation)
        candidate = trial.prompt_adaptation_state["pending_candidate"]
        history = {
            "version": candidate["candidate_overlay"]["version"],
            "parent_version": candidate["parent_overlay"]["version"],
            "trigger_failure": candidate["trigger_failure"],
            "trigger_batches": candidate["trigger_batches"],
            "critic_diagnosis": candidate["critic"],
            "patch": candidate["patch"],
            "validation_batches": candidate["validation_batches"],
            "validation_snapshot_ids": candidate["validation_snapshot_ids"],
            "evaluation_results": {"batches": normalized, "decision": decision},
            "accepted": decision["accepted"],
            "created_at": candidate["created_at"],
            "completed_at": utc_now(),
        }
        if decision["accepted"]:
            trial.prompt_adaptation_state["active_overlay"] = deepcopy(candidate["candidate_overlay"])
        trial.prompt_adaptation_state["history"].append(history)
        trial.prompt_adaptation_state["pending_candidate"] = None
        self.state = trial
        self._commit("prompt_adaptation_validation_completed", {"history": history})
        return {"accepted": decision["accepted"], "decision": decision, "status": self.status()}

    def submit_relational_review_group_atomically(
        self, payloads: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Commit a fully reviewed fan-out group once, or not at all.

        The independent reviewers may finish in any order. Their observations
        are validated before this method is called; applying them to a deep copy
        keeps an interruption or an invalid later claim from leaving a durable
        prefix of the group in the project state.
        """
        if not payloads:
            raise ValueError("a relational review group must contain at least one result")
        trial = deepcopy(self.state)
        pending_adaptation_context = deepcopy(self.state.pending_relational_adaptation_context)
        pending_relational_payload = deepcopy(self.state.pending_relational_payload)
        events: list[dict[str, Any]] = []
        claim_ids: list[str] = []
        for payload in payloads:
            if trial.pending_relational_payload is None:
                raise ValueError("relational review group contains more results than pending claims")
            event = apply_relational_grounding_validation(trial, payload)
            events.append(event)
            claim_id = payload.get("claim_id")
            if not isinstance(claim_id, str):
                raise ValueError("relational review result is missing claim_id")
            claim_ids.append(claim_id)
        if (
            events
            and events[-1].get("review_status") == "COMPLETE"
            and prompt_adaptation_enabled(self.state)
            and pending_adaptation_context is not None
            and pending_relational_payload is not None
        ):
            self._queue_failure_extraction(
                trial,
                pending_adaptation_context,
                pending_relational_payload,
                events[-1],
            )
        self.state = trial
        self._commit(
            "relational_grounding_review_group_committed",
            {"claim_ids": claim_ids, "review_events": events, "atomic_group_size": len(payloads)},
        )
        return {"accepted_action": "relational_grounding_review_group_committed", "status": self.status()}

    def export_outputs(self, output_dir: str | Path | None = None) -> Path:
        destination = Path(output_dir) if output_dir is not None else self.root / "outputs"
        destination.mkdir(parents=True, exist_ok=True)
        objects: dict[str, Any] = {
            "concepts.json": [asdict(item) for item in self.state.concepts],
            "relationships.json": [asdict(item) for item in self.state.relationships],
            "processes.json": [asdict(item) for item in self.state.processes],
            "process_edge_review_cache.json": [
                asdict(item) for item in self.state.process_edge_review_cache
            ],
            "memos.json": [asdict(item) for item in self.state.memos],
            "negative_cases.json": [asdict(item) for item in self.state.negative_cases],
            "theoretical_sampling_needs.json": [asdict(item) for item in self.state.theoretical_sampling_needs],
            "integrated_theory.json": asdict(self.state.integrated_theory) if self.state.integrated_theory else {"status": "not_yet_integrated"},
            "analysis_state.json": state_to_dict(self.state),
        }
        for name, value in objects.items():
            write_json(destination / name, value)
        workspace = self.root / "integration_workspace.json"
        if workspace.exists():
            try:
                write_json(
                    destination / "integration_workspace.json",
                    json.loads(workspace.read_text(encoding="utf-8")),
                )
            except (OSError, json.JSONDecodeError):
                # A malformed workspace must not be silently used for final
                # synthesis; the runner validates it before use. Export simply
                # leaves it out rather than replacing analytic outputs.
                pass
        adaptation_dir = self.root / "prompt_adaptation"
        if adaptation_dir.exists():
            for name in (
                "failure_memory.jsonl",
                "prompt_history.jsonl",
                "context_audit.jsonl",
                "frozen-stage2-base.md",
            ):
                source = adaptation_dir / name
                if source.is_file():
                    destination.joinpath(name).write_bytes(source.read_bytes())
        (destination / "analysis_report.md").write_text(render_analysis_report(self.state), encoding="utf-8")
        return destination

    def _open_packet(self, packet: dict[str, Any], record_ids: list[str]) -> dict[str, Any]:
        records, case_contexts, context_stats = self._model_records_for_ids(record_ids)
        memos, memo_retrieval = self._open_coding_memos(records)
        packet.update({
            "records": records,
            # Memos are durable analytic state, but sending every historical
            # memo on every stateless Stage-1 call makes a one-record packet
            # grow without bound.  The complete state remains persisted; this
            # view is a deterministic retrieval of the material most useful
            # for comparing the supplied incident.
            "memos": memos,
            "memo_retrieval": memo_retrieval,
            "prompt": load_prompt("gt-01-open-coding.md"),
            "expected_output": open_coding_output_schema(record_ids),
        })
        if case_contexts:
            packet["case_context_mode"] = "shared_case_contexts"
            packet["case_contexts"] = case_contexts
        if self.state.config.context_policy == "legacy":
            packet["concept_inventory"] = self._packet_concept_inventory()
        packet = open_context_packet(
            self.state,
            packet,
            packet["records"],
            target_tokens=OPEN_CONTEXT_TARGET_TOKENS,
        )
        if context_stats and isinstance(packet.get(CONTEXT_AUDIT_KEY), dict):
            details = packet[CONTEXT_AUDIT_KEY].setdefault("estimate_breakdown", {})
            if isinstance(details, dict):
                details["shared_case_context"] = context_stats
        if isinstance(packet.get(CONTEXT_AUDIT_KEY), dict):
            details = packet[CONTEXT_AUDIT_KEY].setdefault("estimate_breakdown", {})
            if isinstance(details, dict):
                details["memo_retrieval"] = memo_retrieval
        return packet

    def _open_coding_coverage_packet(
        self, packet: dict[str, Any], record_ids: list[str]
    ) -> dict[str, Any]:
        """Render a bounded, one-judgment-per-record audit packet.

        It uses the same lossless SoCRATES incident view as open coding, but
        has no batch-level discovery fields that could let a model substitute a
        few representative findings for record coverage.
        """
        records, case_contexts, context_stats = self._model_records_for_ids(record_ids)
        packet.update({
            "records": records,
            "memos": [],
            "memo_retrieval": {
                "mode": "none",
                "reason": "record-level coverage audit uses the live concept inventory and source incidents",
            },
            "prompt": load_prompt("gt-01c-open-coding-coverage-audit.md"),
            "expected_output": open_coding_coverage_audit_output_schema(record_ids),
        })
        if case_contexts:
            packet["case_context_mode"] = "shared_case_contexts"
            packet["case_contexts"] = case_contexts
        packet = open_context_packet(
            self.state,
            packet,
            packet["records"],
            target_tokens=OPEN_CONTEXT_TARGET_TOKENS,
        )
        if context_stats and isinstance(packet.get(CONTEXT_AUDIT_KEY), dict):
            details = packet[CONTEXT_AUDIT_KEY].setdefault("estimate_breakdown", {})
            if isinstance(details, dict):
                details["shared_case_context"] = context_stats
        return packet

    def _relational_packet(self, packet: dict[str, Any], action_info: dict[str, Any]) -> dict[str, Any]:
        record_ids = list(action_info["record_ids"])
        prompt, prompt_metadata = stage2_prompt(self.state)
        records, case_contexts, context_stats = self._model_records_for_ids(record_ids)
        packet.update({
            "records": records,
            "concept_inventory": self._packet_concept_inventory(),
            "relation_inventory": self._packet_relation_inventory(),
            "process_inventory": self._packet_process_inventory(),
            "memos": self._packet_memos(),
            "negative_cases": self._packet_negative_cases(),
            "validator_feedback": list(self.state.relational_validation_feedback),
            "prompt": prompt,
            "prompt_adaptation": prompt_metadata,
            "expected_output": relational_output_schema(record_ids),
            "stage2_response_policy": {
                "output_maxima": {
                    "concept_updates": 2,
                    "relationship_updates": 3,
                    "process_updates": 1,
                    "memo_updates": 2,
                },
                "quota_rule": "Maximums, not targets; return fewer when evidence is weak.",
                "deferral_rule": "Every input is processed or deferred. If an un-emitted finding needs follow-up, defer its record with a brief reason; do not cite a deferred record in this response.",
            },
        })
        if action_info.get("forced_resolution_record_ids"):
            packet["forced_resolution_record_ids"] = list(
                action_info["forced_resolution_record_ids"]
            )
        if action_info.get("deferred_record_context"):
            packet["deferred_record_context"] = deepcopy(action_info["deferred_record_context"])
        if case_contexts:
            packet["case_context_mode"] = "shared_case_contexts"
            packet["case_contexts"] = case_contexts
        packet = self._fit_relational_packet_budget(packet)
        if context_stats and isinstance(packet.get(CONTEXT_AUDIT_KEY), dict):
            details = packet[CONTEXT_AUDIT_KEY].setdefault("estimate_breakdown", {})
            if isinstance(details, dict):
                details["shared_case_context"] = context_stats
        return packet

    def capture_prompt_adaptation_snapshot(self, packet: dict[str, Any]) -> dict[str, Any] | None:
        """Write an immutable pre-Stage-2 replay snapshot without mutating state.

        Raw records are not duplicated: they are immutable project input and
        are referenced by a hash.  The snapshot instead keeps the exact live
        analysis state and rendered packet context that a later old/candidate
        replay needs.
        """
        if not prompt_adaptation_enabled(self.state) or packet.get("action") != "relational_process_analysis":
            return None
        metadata = packet.get("prompt_adaptation")
        if not isinstance(metadata, dict) or metadata.get("strategy_version") is None:
            raise ValueError("an adaptive Stage-2 packet is missing strategy provenance")
        raw_state = state_to_dict(self.state)
        records = raw_state.pop("records")
        record_ids = [item["id"] for item in packet.get("records", []) if isinstance(item, dict) and isinstance(item.get("id"), str)]
        # The snapshot must identify the rendered task envelope as well as the
        # analysis state.  Runtime controls (for example the output-token
        # budget) live in the packet's context audit and can change without a
        # state revision.  Omitting them would make an old immutable snapshot
        # collide with a materially different request on recovery.
        input_packet_sha256 = canonical_hash(packet)
        identity = canonical_hash({
            "revision": self.state.revision,
            "record_ids": record_ids,
            "strategy": metadata,
            "state": raw_state,
            "input_packet_sha256": input_packet_sha256,
        })[:20]
        snapshot_id = f"stage2-{self.state.revision:06d}-{identity}"
        snapshot = {
            "snapshot_id": snapshot_id,
            "created_at": utc_now(),
            "source_revision": self.state.revision,
            "record_ids": record_ids,
            "record_set_sha256": canonical_hash(records),
            "state_without_records": raw_state,
            "input_packet": deepcopy(packet),
            "input_packet_sha256": input_packet_sha256,
            "prompt_provenance": deepcopy(metadata),
            "stage2_context_audit": deepcopy(packet.get(CONTEXT_AUDIT_KEY, {})),
        }
        path = self.root / "prompt_adaptation" / "snapshots" / f"{snapshot_id}.json"
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing.get("input_packet_sha256") != snapshot["input_packet_sha256"]:
                raise ValueError("a prompt-adaptation snapshot ID was reused with different input")
        else:
            write_json(path, snapshot)
        return {
            "snapshot_id": snapshot_id,
            "batch_id": snapshot_id,
            "record_ids": record_ids,
            "prompt_version": metadata["strategy_version"],
            "prompt_sha256": metadata["strategy_sha256"],
            "snapshot_file": str(path.relative_to(self.root)),
        }

    def attach_prompt_adaptation_snapshot(self, context: dict[str, Any], *, staged: bool) -> None:
        """Attach a pre-state snapshot only after the corresponding result is valid."""
        if not prompt_adaptation_enabled(self.state):
            return
        if staged:
            if self.state.pending_relational_payload is None:
                raise ValueError("a staged relational snapshot requires a pending relational payload")
            self.state.pending_relational_adaptation_context = deepcopy(context)
            self._commit("prompt_adaptation_snapshot_attached", {"context": context})
            return
        # A no-claim Stage-2 batch has no normal reviewer outcome, so it is
        # deliberately auditable but ineligible to trigger or validate a patch.
        initialize_prompt_adaptation_state(self.state)
        self.state.prompt_adaptation_state["failure_memory"].append({
            **deepcopy(context),
            "methodological_failures": {},
            "examples": {},
            "eligible": False,
            "created_at": utc_now(),
            "reason": "Stage-2 batch contained no independently reviewable relationship or process claim.",
        })
        self._commit("prompt_adaptation_unreviewed_batch_recorded", {"context": context})

    def _failure_extraction_packet(self, packet: dict[str, Any]) -> dict[str, Any]:
        pending = self.state.pending_prompt_adaptation_failure_extraction
        if pending is None:
            raise RuntimeError("no completed Stage-2 batch awaits failure extraction")
        audit = pending["audit"]
        # This is deliberately an audit projection, rather than the original
        # Stage-2 response.  It preserves every supplied claim/review/evidence
        # ID but removes duplicate corpus quotations: the extractor only
        # classifies an already reviewed batch and may cite IDs, not reanalyse
        # source text.
        batch = {
            "batch_id": audit["batch_id"],
            "proposal_summary": self._compact_adaptation_value(audit["proposal_summary"]),
            "review_claims": self._compact_adaptation_value(audit["review_claims"]),
            "review_decisions": self._compact_adaptation_value(audit["review_decisions"]),
            "retrieval_rule": compact_text(str(audit["retrieval_rule"]), COMPACT_TEXT_CHARS),
            "known_evidence_record_ids": list(audit["known_evidence_record_ids"]),
            "concept_index": [
                {
                    "id": item["id"],
                    "label": compact_text(str(item["label"]), COMPACT_TEXT_CHARS),
                    "definition": compact_text(str(item["definition"]), COMPACT_TEXT_CHARS),
                }
                for item in audit["concept_index"]
            ],
            "constraint": audit["constraint"],
        }
        packet.update({
            "batch": batch,
            "prompt": load_prompt("gt-02c-extract-methodological-failures.md"),
            "expected_output": failure_extraction_schema(pending["batch_id"]),
        })
        self._audit_adaptation_packet(
            packet,
            stage=FAILURE_EXTRACTION_ACTION,
            mode="completed_batch_audit_index",
            target_tokens=ADAPTATION_FAILURE_EXTRACTION_TARGET_TOKENS,
            counts={
                "review_claims": len(audit["review_claims"]),
                "review_decisions": len(audit["review_decisions"]),
                "known_evidence_record_ids": len(audit["known_evidence_record_ids"]),
                "concept_index": len(audit["concept_index"]),
            },
            details={
                "raw_corpus_included": False,
                "text_compression_chars": COMPACT_TEXT_CHARS,
                "preserves_all_claim_and_evidence_ids": True,
            },
        )
        return packet

    def _critic_packet(self, packet: dict[str, Any]) -> dict[str, Any]:
        candidate = self._pending_adaptation_candidate("critic")
        memory = self.state.prompt_adaptation_state["failure_memory"]
        relevant = [
            item for item in memory
            if item["batch_id"] in candidate["trigger_window"]
        ]
        # Keep a complete, compact index for the whole trigger window.  Only
        # event-linked original records are retrieved in detail, and detail is
        # reduced deterministically before any context limit can be crossed.
        memory_index = [self._compact_failure_memory_item(item) for item in relevant]
        all_examples = self._ordered_failure_examples(relevant)
        chosen_detail_k = 0
        selected_examples: list[dict[str, Any]] = []
        for detail_k in ADAPTATION_EXAMPLE_DETAIL_STEPS:
            candidate_examples = deepcopy(all_examples[:detail_k])
            evidence_ids = sorted({
                record_id
                for item in candidate_examples
                for record_id in item.get("evidence_record_ids", [])
                if isinstance(record_id, str)
            })
            proposed = {
                **packet,
                "frozen_stage2_instructions": self.state.prompt_adaptation_state["frozen_base_prompt"],
                "adaptive_strategy_overlay": active_overlay(self.state),
                "trigger": {
                    "failure_type": candidate["trigger_failure"],
                    "batches": candidate["trigger_batches"],
                },
                "failure_memory": deepcopy(memory_index),
                "representative_failure_examples": candidate_examples,
                "evidence_records": self._records_for_ids(evidence_ids),
                "prompt": load_prompt("gt-02d-prompt-critic.md"),
                "expected_output": critic_schema(),
            }
            selected_examples = candidate_examples
            chosen_detail_k = detail_k
            if packet_token_estimate(proposed) <= ADAPTATION_CRITIC_TARGET_TOKENS:
                packet = proposed
                break
        else:  # pragma: no cover - the loop always includes the zero-detail projection.
            packet = proposed
        if not packet.get("prompt"):
            # The last, zero-detail candidate is still a complete index.  The
            # runner will produce an explicit protected-budget diagnostic if a
            # single unreducible index itself is too large.
            packet = proposed
        self._audit_adaptation_packet(
            packet,
            stage=CRITIC_ACTION,
            mode="complete_failure_index_with_deterministic_event_retrieval",
            target_tokens=ADAPTATION_CRITIC_TARGET_TOKENS,
            counts={
                "failure_batches": len(relevant),
                "failure_examples_total": len(all_examples),
                "failure_examples_detailed": len(selected_examples),
                "failure_examples_compact_only": len(all_examples) - len(selected_examples),
                "evidence_records_retrieved": len(packet["evidence_records"]),
            },
            details={
                "detail_k": chosen_detail_k,
                "detail_k_steps": list(ADAPTATION_EXAMPLE_DETAIL_STEPS),
                "retrieval_rule": "Only evidence_record_ids named by selected methodological-failure events are retrieved as raw records.",
                "complete_failure_counts_and_reviewer_decisions_retained": True,
            },
        )
        return packet

    def _editor_packet(self, packet: dict[str, Any]) -> dict[str, Any]:
        candidate = self._pending_adaptation_candidate("editor")
        packet.update({
            "frozen_stage2_instructions": self.state.prompt_adaptation_state["frozen_base_prompt"],
            "adaptive_strategy_overlay": active_overlay(self.state),
            "critic_diagnosis": deepcopy(candidate["critic"]),
            "prompt": load_prompt("gt-02e-prompt-editor.md"),
            "expected_output": editor_schema(candidate["parent_overlay"]["version"], candidate["trigger_failure"]),
        })
        self._audit_adaptation_packet(
            packet,
            stage=EDITOR_ACTION,
            mode="minimal_strategy_inputs",
            target_tokens=ADAPTATION_EDITOR_TARGET_TOKENS,
            counts={
                "parent_strategy_rules": len(active_overlay(self.state)["rules"]),
                "critic_evidence_items": len(candidate["critic"].get("evidence", [])),
            },
            details={
                "raw_corpus_included": False,
                "scope": "frozen base, active overlay, and accepted critic diagnosis only",
            },
        )
        return packet

    @staticmethod
    def _compact_adaptation_value(value: Any, *, text_limit: int = COMPACT_TEXT_CHARS) -> Any:
        """Compress prose while retaining every structured audit item and ID."""
        if isinstance(value, str):
            return compact_text(value, text_limit)
        if isinstance(value, list):
            return [GroundedTheoryProject._compact_adaptation_value(item, text_limit=text_limit) for item in value]
        if isinstance(value, dict):
            return {
                key: GroundedTheoryProject._compact_adaptation_value(item, text_limit=text_limit)
                for key, item in value.items()
            }
        return deepcopy(value)

    def _compact_failure_memory_item(self, item: dict[str, Any]) -> dict[str, Any]:
        return {
            "batch_id": item["batch_id"],
            "snapshot_id": item["snapshot_id"],
            "prompt_version": item["prompt_version"],
            "methodological_failures": deepcopy(item.get("methodological_failures", {})),
            # The critic's complete failure-window index retains the actual
            # reviewer decisions/rationales.  Detail reduction below applies
            # only to extra event examples and their raw-record retrieval;
            # it must not hide the evidence that made a failure count valid.
            "reviewer_audit": deepcopy(item.get("reviewer_audit", {})),
            "eligible": bool(item.get("eligible")),
        }

    @staticmethod
    def _ordered_failure_examples(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        examples: list[dict[str, Any]] = []
        for item in sorted(items, key=lambda value: (str(value.get("batch_id", "")), str(value.get("snapshot_id", "")))):
            for failure_type in sorted(item.get("examples", {})):
                raw_examples = item["examples"].get(failure_type, [])
                if not isinstance(raw_examples, list):
                    continue
                for example in raw_examples:
                    if isinstance(example, dict):
                        examples.append({
                            "batch_id": item["batch_id"],
                            "failure_type": failure_type,
                            **GroundedTheoryProject._compact_adaptation_value(example),
                        })
        return sorted(examples, key=lambda value: (
            str(value.get("batch_id", "")),
            str(value.get("failure_type", "")),
            str(value.get("claim_id", "")),
        ))

    @staticmethod
    def _audit_adaptation_packet(
        packet: dict[str, Any], *, stage: str, mode: str, target_tokens: int,
        counts: dict[str, int], details: dict[str, Any],
    ) -> None:
        packet[CONTEXT_AUDIT_KEY] = audit_context(
            packet,
            stage=stage,
            mode=mode,
            retrieval_level=0,
            counts=counts,
            details={
                **details,
                "target_tokens": target_tokens,
                "estimated_before_runner_budget_check": packet_token_estimate(packet),
            },
        )

    def _validation_packet(self, packet: dict[str, Any]) -> dict[str, Any]:
        candidate = self._pending_adaptation_candidate("validation")
        packet.update({
            "candidate_id": candidate["id"],
            "parent_version": candidate["parent_overlay"]["version"],
            "candidate_version": candidate["candidate_overlay"]["version"],
            "validation_batches": candidate["validation_batches"],
            "validation_snapshot_ids": candidate["validation_snapshot_ids"],
            "rule": "This action is executed only by the isolated validation runner; it cannot be submitted as one model response.",
        })
        return packet

    def _queue_failure_extraction(
        self,
        trial: GroundedTheoryState,
        context: dict[str, Any],
        candidate: dict[str, Any],
        completed_review: dict[str, Any],
    ) -> None:
        """Save a compact, reviewer-grounded audit for the frozen extractor."""
        if trial.config.prompt_adaptation.mode == "online":
            initialize_prompt_adaptation_state(trial)
            adaptation = trial.prompt_adaptation_state
            prior_count = adaptation.get("reviewed_stage2_batch_count", 0)
            if type(prior_count) is not int or prior_count < 0:
                raise RuntimeError("invalid reviewed Stage-2 batch count")
            reviewed_count = prior_count + 1
            adaptation["reviewed_stage2_batch_count"] = reviewed_count
            interval = trial.config.prompt_adaptation.failure_extraction_interval_batches
            if reviewed_count % interval != 0:
                adaptation.setdefault("skipped_failure_extraction_batches", []).append({
                    "batch_id": context["batch_id"],
                    "snapshot_id": context.get("snapshot_id"),
                    "record_ids": list(context.get("record_ids", [])),
                    "reviewed_batch_number": reviewed_count,
                    "reason": f"scheduled every {interval} independently reviewed Stage-2 batches",
                    "created_at": utc_now(),
                })
                trial.pending_relational_adaptation_context = None
                return
        material = build_relational_review_material(self.state, candidate)
        review = completed_review.get("review", {})
        relation_reviews = {
            f"relationship:{item.get('relationship_update_index')}": item
            for item in review.get("relationship_reviews", [])
            if isinstance(item, dict)
        }
        edge_reviews = {
            f"process:{item.get('process_update_index')}:edge:{item.get('edge_index')}": item
            for item in review.get("process_edge_reviews", [])
            if isinstance(item, dict)
        }
        review_decisions = {**relation_reviews, **edge_reviews}
        known_record_ids = {
            record_id
            for claim in material["review_claims"]
            for cohort in claim.get("evidence_sets", {}).values()
            for record_id in cohort
            if isinstance(record_id, str)
        }
        compact_concepts = [
            {"id": concept.id, "label": concept.label, "definition": concept.definition}
            for concept in self.state.concepts
        ]
        proposal_summary: dict[str, dict[str, Any]] = {}
        for index, update in enumerate(candidate.get("relationship_updates", [])):
            if isinstance(update, dict):
                proposal_summary[f"relationship:{index}"] = self._compact_candidate_claim(update)
        for process_index, process in enumerate(candidate.get("process_updates", [])):
            if not isinstance(process, dict):
                continue
            for edge_index, edge in enumerate(process.get("edges", [])):
                if isinstance(edge, dict):
                    proposal_summary[f"process:{process_index}:edge:{edge_index}"] = self._compact_candidate_claim(edge)
        audit = {
            "batch_id": context["batch_id"],
            "candidate_submission": deepcopy(candidate),
            "review_claims": deepcopy(material["review_claims"]),
            "review_decisions": review_decisions,
            "proposal_summary": proposal_summary,
            "retrieval_rule": material["retrieval_rule"],
            "known_evidence_record_ids": sorted(known_record_ids),
            "concept_index": compact_concepts,
            "constraint": (
                "Classify only prompt-addressable methodological failures supported by this audit. "
                "Do not report malformed JSON, infrastructure, token, checkpoint, ID, or quote-index errors. "
                "A missing intermediate mechanism requires evidence of a specific intermediate concept; "
                "a reviewer rejection alone is not sufficient."
            ),
        }
        if trial.config.prompt_adaptation.mode == "offline":
            initialize_prompt_adaptation_state(trial)
            trial.prompt_adaptation_state.setdefault("offline_failure_audits", []).append({
                **deepcopy(context),
                "audit": audit,
                "recorded_at": utc_now(),
            })
            trial.pending_relational_adaptation_context = None
            return
        trial.pending_prompt_adaptation_failure_extraction = {
            **deepcopy(context),
            "audit": audit,
        }
        trial.pending_relational_adaptation_context = None

    @staticmethod
    def _compact_candidate_claim(claim: dict[str, Any]) -> dict[str, Any]:
        """Keep proposal provenance without duplicating raw corpus text."""
        fields = (
            "comparison", "source_concept_id", "relationship", "target_concept_id",
            "comparative_basis", "grounding_kind", "grounding_explanation", "status",
            "conditions", "boundary_conditions",
        )
        compact = {name: deepcopy(claim[name]) for name in fields if name in claim}
        compact["evidence_record_ids"] = [
            item["record_id"] for item in claim.get("evidence", [])
            if isinstance(item, dict) and isinstance(item.get("record_id"), str)
        ]
        return compact

    def _accept_failure_extraction(
        self, trial: GroundedTheoryState, payload: dict[str, Any]
    ) -> dict[str, Any]:
        pending = trial.pending_prompt_adaptation_failure_extraction
        if pending is None:
            raise ValueError("there is no Stage-2 batch awaiting failure extraction")
        claims = {
            item["claim_id"] for item in pending["audit"]["review_claims"]
            if isinstance(item.get("claim_id"), str)
        }
        record_ids = set(pending["audit"]["known_evidence_record_ids"])
        extraction = normalize_failure_extraction(payload, pending["batch_id"], claims, record_ids)
        memory_record = append_failure_memory(trial, pending, extraction)
        candidate = (
            begin_candidate_if_triggered(trial)
            if trial.config.prompt_adaptation.mode == "online"
            else None
        )
        return {
            "failure_memory": memory_record,
            "candidate_started": {
                "id": candidate["id"], "failure_type": candidate["trigger_failure"]
            } if candidate else None,
        }

    def _pending_adaptation_candidate(self, phase: str) -> dict[str, Any]:
        candidate = self.state.prompt_adaptation_state.get("pending_candidate")
        if not isinstance(candidate, dict) or candidate.get("phase") != phase:
            raise RuntimeError(f"no prompt-adaptation candidate is awaiting {phase}")
        return candidate

    def _accept_prompt_critic(
        self, trial: GroundedTheoryState, payload: dict[str, Any]
    ) -> dict[str, Any]:
        candidate = trial.prompt_adaptation_state.get("pending_candidate")
        if not isinstance(candidate, dict) or candidate.get("phase") != "critic":
            raise ValueError("there is no candidate awaiting a prompt critic")
        memory = trial.prompt_adaptation_state["failure_memory"]
        known_batches = {item["batch_id"] for item in memory}
        known_claims = {
            example["claim_id"]
            for item in memory
            for examples in item.get("examples", {}).values()
            for example in examples
            if isinstance(example, dict) and isinstance(example.get("claim_id"), str)
        }
        critic = normalize_critic(payload, candidate["trigger_failure"], known_batches, known_claims)
        if not critic["prompt_related"]:
            history = {
                "version": None,
                "parent_version": candidate["parent_overlay"]["version"],
                "trigger_failure": candidate["trigger_failure"],
                "trigger_batches": candidate["trigger_batches"],
                "critic_diagnosis": critic,
                "patch": None,
                "validation_batches": [],
                "evaluation_results": {},
                "accepted": False,
                "outcome": "not_prompt_related",
                "created_at": candidate["created_at"],
                "completed_at": utc_now(),
            }
            trial.prompt_adaptation_state["history"].append(history)
            trial.prompt_adaptation_state["pending_candidate"] = None
            return {"history": history}
        candidate["critic"] = critic
        candidate["phase"] = "editor"
        return {"candidate_id": candidate["id"], "critic": critic, "next_phase": "editor"}

    def _accept_prompt_editor(
        self, trial: GroundedTheoryState, payload: dict[str, Any]
    ) -> dict[str, Any]:
        candidate = trial.prompt_adaptation_state.get("pending_candidate")
        if not isinstance(candidate, dict) or candidate.get("phase") != "editor":
            raise ValueError("there is no candidate awaiting a prompt editor")
        patch = normalize_patch(payload, candidate["parent_overlay"], candidate["trigger_failure"])
        candidate["patch"] = patch
        memory = trial.prompt_adaptation_state
        version_number = int(memory.get("next_strategy_version", 2))
        candidate["candidate_overlay"] = candidate_overlay(
            candidate["parent_overlay"], patch, version_number=version_number
        )
        memory["next_strategy_version"] = version_number + 1
        candidate["phase"] = "validation"
        return {
            "candidate_id": candidate["id"],
            "candidate_version": candidate["candidate_overlay"]["version"],
            "next_phase": "validation",
        }

    def _fit_relational_packet_budget(self, packet: dict[str, Any]) -> dict[str, Any]:
        """Fit Stage 2 context before the runner reaches its hard limit.

        The normal packet remains preferred. If analytic inventories have grown
        too large, switch only their *presentation* to compact forms that keep
        identifiers, directed structures, statuses, and item counts. The full
        audit trail stays in state and is still available to deterministic
        review/commit logic and final outputs.
        """
        if packet_token_estimate(packet) <= RELATIONAL_PACKET_TARGET_TOKENS:
            packet[CONTEXT_AUDIT_KEY] = audit_context(
                packet,
                stage="relational_process_analysis",
                mode="complete_inventory",
                retrieval_level=0,
                counts={
                    "concepts": len(self.state.concepts),
                    "relationships": len(self.state.relationships),
                    "processes": len(self.state.processes),
                    "memos": len(self.state.memos),
                    "negative_cases": len(self.state.negative_cases),
                },
                details={"target_tokens": RELATIONAL_PACKET_TARGET_TOKENS},
            )
            return packet
        packet["concept_inventory"] = self._compact_concept_inventory()
        packet["relation_inventory"] = self._compact_relation_inventory()
        packet["process_inventory"] = self._compact_process_inventory()
        packet["memos"] = self._compact_memos()
        packet["negative_cases"] = self._compact_negative_cases()
        packet["context_compaction"] = {
            "mode": "compact_inventory",
            "reason": "Full persisted audit trails remain in project state; packet inventories were summarized to fit the protected context budget.",
        }
        if packet_token_estimate(packet) <= RELATIONAL_PACKET_TARGET_TOKENS:
            return packet

        # A full compact inventory can still grow with thousands of memos and
        # relations.  Before shrinking the raw analytical batch, retain every
        # object as a structural ID/label catalogue and expand only the pieces
        # deterministically relevant to this batch.  This is the Stage-2
        # analogue of open-coding indexed retrieval: no state or evidence is
        # removed, and future integration still receives the whole corpus.
        self._apply_stage2_structural_retrieval(packet)
        if packet_token_estimate(packet) <= RELATIONAL_PACKET_TARGET_TOKENS:
            return packet

        # A study can accumulate hundreds of formal relations or process
        # records after Stage 2 begins.  Structural cards then repeat enough
        # field names to overflow even though their detailed content is
        # already bounded.  Retain every ID with an exact range/literal
        # catalogue and keep the same deterministic detail retrieval.  This
        # is a presentation change only, not an arbitrary item slice.
        self._apply_stage2_identifier_retrieval(packet)
        if packet_token_estimate(packet) <= RELATIONAL_PACKET_TARGET_TOKENS:
            return packet

        # No arbitrary representative slice is permitted here. A packet that
        # exceeds the protected target even after the complete identifier
        # catalogue remains explicit so the runner can reduce raw records or
        # request intervention without silently changing analytic evidence.
        packet[CONTEXT_AUDIT_KEY] = audit_context(
            packet,
            stage="relational_process_analysis",
            mode="identifier_index_with_retrieval",
            retrieval_level=2,
            counts={
                "concepts": len(self.state.concepts),
                "relationships": len(self.state.relationships),
                "processes": len(self.state.processes),
                "memos": len(self.state.memos),
                "negative_cases": len(self.state.negative_cases),
            },
            details={"target_tokens": RELATIONAL_PACKET_TARGET_TOKENS},
        )
        return packet

    def _apply_stage2_structural_retrieval(self, packet: dict[str, Any]) -> None:
        """Replace verbose Stage-2 inventories with complete structural cards.

        The cards retain every stable ID, label/direction/status and count.
        The additional retrieved view supplies full analytic detail only for
        concepts and linked objects that lexical comparison with the current
        raw batch identifies as relevant, plus recent concepts and memos.
        """
        records = packet.get("records", [])
        model_records = records if isinstance(records, list) else []
        ranked = rank_relevant_concepts(self.state, model_records)
        relevant_concepts = [concept for concept, _ in ranked[:STAGE2_RETRIEVED_CONCEPTS]]
        recent_concepts = sorted(
            self.state.concepts,
            key=lambda item: (-item.last_modified_revision, item.id),
        )[:STAGE2_RECENT_CONCEPTS]
        selected_ids = {concept.id for concept in [*relevant_concepts, *recent_concepts]}
        selected_concepts = [
            concept for concept in self.state.concepts if concept.id in selected_ids
        ]
        selected_relation_ids = {
            relation.id
            for relation in self.state.relationships
            if relation.source_concept_id in selected_ids
            or relation.target_concept_id in selected_ids
        }
        selected_relation_ids = set(sorted(selected_relation_ids)[:STAGE2_RETRIEVED_RELATIONS])
        selected_process_ids = {
            process.id
            for process in self.state.processes
            if any(
                edge.source_concept_id in selected_ids or edge.target_concept_id in selected_ids
                for edge in process.edges
            )
        }
        selected_process_ids = set(sorted(selected_process_ids)[:STAGE2_RETRIEVED_PROCESSES])
        selected_negative_ids = {
            case.id
            for case in self.state.negative_cases
            if case.target_id in selected_ids
            or case.target_id in selected_relation_ids
            or case.target_id in selected_process_ids
        }
        selected_negative_ids = set(sorted(selected_negative_ids)[:STAGE2_RETRIEVED_NEGATIVE_CASES])
        _, memo_retrieval = self._open_coding_memos(model_records)
        retrieved_memo_ids = set(memo_retrieval["selected_memo_ids"])

        packet["concept_inventory"] = self._structural_concept_index()
        packet["relation_inventory"] = self._structural_relation_index()
        packet["process_inventory"] = self._structural_process_index()
        packet["memos"] = self._structural_memo_index()
        packet["negative_cases"] = self._structural_negative_case_index()
        packet["retrieved_context"] = {
            "mode": "structural_catalog_with_deterministic_retrieval",
            "concept_details": [detailed_concept_view(concept) for concept in selected_concepts],
            # At this compaction tier, the analyst only needs selected
            # relations/processes' direction, wording, status, conditions,
            # and bounded process-edge topology. Re-sending historical quotes,
            # variations, and full process audits here consumed most of the
            # Stage-2 window and forced all future batches to shrink. Those
            # complete audits remain durable and are retrieved in full for
            # independent grounding review and final integration.
            "relation_details": [
                item for item in self._compact_relation_inventory()
                if item["id"] in selected_relation_ids
            ],
            "process_details": [
                item for item in self._compact_process_inventory()
                if item["id"] in selected_process_ids
            ],
            # Stage 2 only needs bounded analytic memo text here.  The full
            # memo/evidence remains durable state and is recovered by the
            # hierarchical integration evidence-on-demand path.
            "memo_details": [
                item for item in self._compact_memos()
                if item["id"] in retrieved_memo_ids
            ],
            "negative_case_details": [
                item for item in self._packet_negative_cases()
                if item["id"] in selected_negative_ids
            ],
            "selected_concept_ids": [concept.id for concept in selected_concepts],
            "memo_retrieval": memo_retrieval,
            "rule": (
                "Every object remains in its structural catalogue. Detailed concepts and "
                "memos are deterministically selected from the current records and recency. "
                "Relation/process details retain direction, status, conditions, and bounded "
                "topology; their full audit evidence remains durable for independent review. "
                "Do not treat an absent older detail view as evidence that the object does not exist."
            ),
        }
        packet["context_compaction"] = {
            "mode": "structural_index_with_retrieval",
            "reason": (
                "Every accepted object remains visible through an ID/label/direction/status "
                "catalogue. Full persisted text and evidence are supplied only for deterministic "
                "batch-relevant retrieval and later hierarchical integration."
            ),
        }
        packet[CONTEXT_AUDIT_KEY] = audit_context(
            packet,
            stage="relational_process_analysis",
            mode="structural_index_with_retrieval",
            retrieval_level=1,
            counts={
                "concepts": len(self.state.concepts),
                "relationships": len(self.state.relationships),
                "processes": len(self.state.processes),
                "memos": len(self.state.memos),
                "negative_cases": len(self.state.negative_cases),
                "retrieved_concepts": len(selected_concepts),
                "retrieved_relations": len(selected_relation_ids),
                "retrieved_processes": len(selected_process_ids),
                "retrieved_memos": len(retrieved_memo_ids),
                "retrieved_negative_cases": len(selected_negative_ids),
            },
            details={"target_tokens": RELATIONAL_PACKET_TARGET_TOKENS},
        )

    def _apply_stage2_identifier_retrieval(self, packet: dict[str, Any]) -> None:
        """Apply the final lossless, bounded Stage-2 context presentation.

        Each accepted object remains exactly addressable through an inclusive
        ID range or literal ID.  The detailed context selected in the prior
        structural-retrieval step is intentionally retained unchanged, so the
        current analytic comparison does not lose its evidence-aware objects.
        """
        packet["concept_inventory"] = self._complete_identifier_catalogue(
            [concept.id for concept in self.state.concepts]
        )
        packet["relation_inventory"] = self._complete_identifier_catalogue(
            [relation.id for relation in self.state.relationships]
        )
        packet["process_inventory"] = self._complete_identifier_catalogue(
            [process.id for process in self.state.processes]
        )
        # Memo topics and links can grow with every Stage-2 transaction. At
        # this final presentation tier, all memo IDs remain losslessly
        # addressable while the fixed deterministic memo retrieval below
        # retains the only analytic detail needed for the current eight
        # records. Keeping a full structural memo card for every past memo
        # would eventually defeat a fixed-size batch.
        packet["memos"] = self._complete_identifier_catalogue(
            [memo.id for memo in self.state.memos]
        )
        packet["negative_cases"] = self._complete_identifier_catalogue(
            [case.id for case in self.state.negative_cases]
        )
        retrieved = packet.get("retrieved_context")
        if isinstance(retrieved, dict):
            retrieved["mode"] = "identifier_catalog_with_deterministic_retrieval"
            retrieved["rule"] = (
                "Every accepted object is represented exactly in its identifier catalogue. "
                "Detailed views are the deterministic, evidence-aware retrieval for the "
                "current records and recency; a non-retrieved object is not evidence of absence."
            )
        packet["context_compaction"] = {
            "mode": "identifier_index_with_retrieval",
            "reason": (
                "The complete structural catalogue exceeded the protected context target. "
                "Every object remains exactly addressable by an inclusive ID range or literal "
                "ID; unchanged deterministic detail retrieval supplies analytic text and evidence."
            ),
        }
        packet[CONTEXT_AUDIT_KEY] = audit_context(
            packet,
            stage="relational_process_analysis",
            mode="identifier_index_with_retrieval",
            retrieval_level=2,
            counts={
                "concepts": len(self.state.concepts),
                "relationships": len(self.state.relationships),
                "processes": len(self.state.processes),
                "memos": len(self.state.memos),
                "negative_cases": len(self.state.negative_cases),
                "retrieved_concepts": len(retrieved.get("concept_details", [])) if isinstance(retrieved, dict) else 0,
                "retrieved_relations": len(retrieved.get("relation_details", [])) if isinstance(retrieved, dict) else 0,
                "retrieved_processes": len(retrieved.get("process_details", [])) if isinstance(retrieved, dict) else 0,
                "retrieved_memos": len(retrieved.get("memo_details", [])) if isinstance(retrieved, dict) else 0,
                "retrieved_negative_cases": len(retrieved.get("negative_case_details", [])) if isinstance(retrieved, dict) else 0,
            },
            details={"target_tokens": RELATIONAL_PACKET_TARGET_TOKENS},
        )

    def _structural_concept_index(self) -> list[dict[str, Any]]:
        return [
            {
                "id": concept.id,
                "label": concept.label,
                "level": concept.level,
                "status": concept.status,
                "parent_category_id": concept.parent_category_id,
                "evidence_count": len(concept.evidence),
            }
            for concept in self.state.concepts
        ]

    def _structural_relation_index(self) -> list[dict[str, Any]]:
        return [
            {
                "id": relation.id,
                "source_concept_id": relation.source_concept_id,
                "relationship": relation.relationship,
                "target_concept_id": relation.target_concept_id,
                "status": relation.status,
                "grounding_kind": relation.grounding_kind,
                "evidence_count": len(relation.evidence),
                "negative_case_count": len(relation.negative_cases),
            }
            for relation in self.state.relationships
        ]

    def _structural_process_index(self) -> list[dict[str, Any]]:
        return [
            {
                "id": process.id,
                "label": process.label,
                "status": process.status,
                "edge_count": len(process.edges),
                "supporting_relation_count": len(process.supporting_relation_ids),
                "negative_case_count": len(process.negative_cases),
            }
            for process in self.state.processes
        ]

    def _structural_memo_index(self) -> dict[str, Any]:
        """Encode every Stage-2 memo in a compact, self-describing catalogue.

        The model needs memo *content* only when it is deterministically
        relevant to the current records.  Listing every historical topic is
        both distracting and unbounded; it can consume the context window
        before the evidence for the current incident arrives.  This index
        therefore keeps every durable memo ID visible, while
        ``retrieved_context.memo_details`` supplies the bounded, lexically
        selected analytic text.  Full topics, linkage, evidence, and memo text
        remain in durable project state for audit and later integration.
        """
        catalogue = self._complete_identifier_catalogue(
            [memo.id for memo in self.state.memos]
        )
        catalogue["rule"] = (
            "Every durable memo ID is represented exactly by an inclusive numeric range "
            "or literal ID. Use retrieved_context.memo_details for deterministically "
            "selected analytic content; a non-retrieved memo is not evidence of absence."
        )
        return catalogue

    @staticmethod
    def _complete_identifier_catalogue(ids: list[str]) -> dict[str, Any]:
        """Losslessly encode stable IDs without repeated JSON string overhead.

        Memo IDs are generated sequentially (for example ``memo_001`` through
        ``memo_665``).  Listing each one makes context grow linearly despite
        carrying no analytic content.  Inclusive ranges are an exact, stable
        representation of every ID.  Non-numeric identifiers remain literal
        entries, so this is not dependent on a particular ID convention.
        """
        return complete_identifier_catalogue(ids)

    def _structural_negative_case_index(self) -> list[dict[str, Any]]:
        return [
            {
                "id": case.id,
                "target_type": case.target_type,
                "target_id": case.target_id,
                "evidence_count": len(case.evidence),
            }
            for case in self.state.negative_cases
        ]

    def _grounding_review_packet(self, packet: dict[str, Any], record_ids: list[str]) -> dict[str, Any]:
        review_material = next_relational_review_material(
            self.state,
            self.state.pending_relational_payload or {},
            self.state.pending_relational_review_results,
        )
        if review_material is None:
            raise RuntimeError("pending relational review has no remaining claim")
        packet.update({
            **review_material,
            "prompt": load_prompt("gt-02b-validate-relational-grounding.md"),
            "expected_output": (
                process_edge_grounding_review_output_schema()
                if review_material["review_claim"].get("claim_type") == "process_edge"
                else relational_grounding_review_output_schema()
            ),
        })
        return packet

    def _stage_or_commit_relations(
        self, trial: GroundedTheoryState, payload: dict[str, Any], action: str
    ) -> tuple[dict[str, Any], str]:
        preview = deepcopy(self.state)
        # Process arrows without a pre-existing relation are intentionally
        # allowed to reach an independent full-corpus review.  The preview
        # still validates the candidate's records, concepts, and references;
        # it simply does not commit an unreviewed process.
        preview_event = apply_relational_analysis(
            preview,
            payload,
            self.next_action(),
            allow_unreviewed_processes=True,
        )
        relation_updates = object_list(payload, "relationship_updates", allow_empty=True)
        process_updates = object_list(payload, "process_updates", allow_empty=True)
        if not relation_updates and not process_updates:
            return apply_relational_analysis(trial, payload, self.next_action()), action
        review_material = build_relational_review_material(self.state, payload)
        if not review_material["review_claims"]:
            # Every process edge reused a relation that was independently
            # validated in an earlier transaction, so no new review is needed.
            reviewed, review_event = apply_review_decisions(
                trial,
                payload,
                {
                    "review_status": "COMPLETE",
                    "relationship_reviews": [],
                    "process_edge_reviews": [],
                    "issues": [],
                },
            )
            committed = apply_relational_analysis(trial, reviewed, self.next_action())
            return {"review_status": "REUSED_VALIDATED_RELATIONS", "review": review_event, "committed": committed}, action
        trial.pending_relational_payload = deepcopy(payload)
        trial.pending_relational_record_ids = list(self.next_action()["record_ids"])
        trial.pending_relational_review_results = []
        return (
            {
                "record_ids": trial.pending_relational_record_ids,
                "relationship_updates": len(relation_updates),
                "process_updates": len(process_updates),
                "validation_preview": preview_event,
            },
            "relational_analysis_staged",
        )

    def _records_for_ids(self, record_ids: list[str]) -> list[dict[str, Any]]:
        records = {record.id: record for record in self.state.records}
        return [asdict(records[record_id]) for record_id in record_ids]

    def _model_records_for_ids(
        self, record_ids: list[str]
    ) -> tuple[list[dict[str, Any]], list[dict[str, str]], dict[str, int] | None]:
        """Render raw records or the opt-in, lossless SoCRATES shared-case view."""
        records = self._records_for_ids(record_ids)
        if os.environ.get("GT_SHARED_CASE_CONTEXT") != "1":
            return records, [], None
        compacted, contexts, stats = shared_case_context_records(records)
        if not contexts:
            return records, [], None
        return compacted, contexts, stats

    def _packet_concept_inventory(self) -> list[dict[str, Any]]:
        """Keep model comparison context bounded without changing audit data."""
        inventory: list[dict[str, Any]] = []
        for concept in self.state.concepts:
            item = asdict(concept)
            item["evidence_count"] = len(item["evidence"])
            item["evidence"] = self._representative_items(
                item["evidence"], PACKET_EVIDENCE_PER_CONCEPT
            )
            for field_name in (
                "definition_revisions",
                "variations",
                "negative_or_boundary_cases",
            ):
                item[f"{field_name}_count"] = len(item[field_name])
                item[field_name] = self._representative_items(
                    item[field_name], PACKET_CONCEPT_HISTORY_ITEMS
                )
            inventory.append(item)
        return inventory

    def _packet_memos(self) -> list[dict[str, Any]]:
        return [self._memo_packet_item(memo) for memo in self.state.memos]

    def _memo_packet_item(self, memo: Any) -> dict[str, Any]:
        """Render one memo with bounded evidence detail for a model packet."""
        item = asdict(memo)
        for field_name in (
            "supporting_evidence",
            "negative_or_contradictory_cases",
        ):
            item[f"{field_name}_count"] = len(item[field_name])
            item[field_name] = self._representative_items(
                item[field_name], PACKET_MEMO_EVIDENCE
            )
        return item

    def _open_coding_memos(
        self, records: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Retrieve a bounded, inspectable memo view for Stage 1.

        Stage 1's global comparison contract is carried by the complete
        concept index.  Memos are analytic working notes, so continually
        resending every old note is neither necessary nor scalable.  We retain
        the most lexically relevant notes and the most recent notes, record
        their IDs, and leave all original memo text/evidence in project state
        for Stage 2 and hierarchical integration.
        """
        all_memos = self.state.memos
        if not all_memos:
            return [], {
                "mode": "relevant_and_recent",
                "total_memos": 0,
                "selected_memo_ids": [],
                "relevant_memo_ids": [],
                "recent_memo_ids": [],
            }

        query = self._memo_terms(" ".join(str(record.get("text", "")) for record in records))
        ranked: list[tuple[Any, float]] = []
        for memo in all_memos:
            memo_text = " ".join(
                [
                    memo.topic,
                    memo.analytic_observation,
                    memo.comparisons,
                    memo.tentative_interpretation,
                    memo.uncertainties,
                    " ".join(memo.questions_for_further_analysis),
                ]
            )
            memo_terms = self._memo_terms(memo_text)
            score = len(query & memo_terms) / max(1, len(query | memo_terms))
            ranked.append((memo, score))
        ranked.sort(key=lambda item: (-item[1], item[0].id))
        relevant = [memo for memo, _ in ranked[:OPEN_CODING_RELEVANT_MEMOS]]
        recent = list(reversed(all_memos[-OPEN_CODING_RECENT_MEMOS:]))
        selected_ids = {memo.id for memo in [*relevant, *recent]}
        selected = [memo for memo in all_memos if memo.id in selected_ids]
        return [self._memo_packet_item(memo) for memo in selected], {
            "mode": "relevant_and_recent",
            "total_memos": len(all_memos),
            "selected_memo_ids": [memo.id for memo in selected],
            "relevant_memo_ids": [memo.id for memo in relevant],
            "recent_memo_ids": [memo.id for memo in recent],
            "rule": (
                "Complete memo state remains persisted. Stage 1 receives the "
                "deterministically retrieved relevant and recent memo details; "
                "the complete concept index remains available for every comparison."
            ),
        }

    @staticmethod
    def _memo_terms(value: str) -> set[str]:
        return {
            token.lower()
            for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", value)
        }

    def _packet_relation_inventory(self) -> list[dict[str, Any]]:
        inventory: list[dict[str, Any]] = []
        for relation in self.state.relationships:
            item = asdict(relation)
            for field_name, limit in (
                ("evidence", PACKET_RELATION_EVIDENCE),
                ("negative_cases", PACKET_RELATION_EVIDENCE),
                ("variations", PACKET_RELATION_HISTORY_ITEMS),
                ("memo_ids", PACKET_PROCESS_ITEMS),
            ):
                item[f"{field_name}_count"] = len(item[field_name])
                item[field_name] = self._representative_items(item[field_name], limit)
            inventory.append(item)
        return inventory

    def _packet_process_inventory(self) -> list[dict[str, Any]]:
        """Summarize prior process audits without re-sending their full corpus evidence.

        The full edge audit trail remains in persisted state and final outputs.
        Stage 2 only needs the direction, status, support scope, and a few
        record IDs to avoid duplicating an already reviewed process; embedding
        every quote and rationale again eventually consumes the whole context
        window even for a one-record analytical batch.
        """
        inventory: list[dict[str, Any]] = []
        for process in self.state.processes:
            item = asdict(process)
            for field_name in (
                "conditions",
                "actions_interactions",
                "consequences",
                "subsequent_changes",
                "alternative_pathways",
                "supporting_relation_ids",
                "supporting_record_ids",
                "negative_cases",
            ):
                item[f"{field_name}_count"] = len(item[field_name])
                item[field_name] = self._representative_items(
                    item[field_name], PACKET_PROCESS_ITEMS
                )
            item["edge_count"] = len(item["edges"])
            item["edges"] = [
                self._packet_process_edge(edge) for edge in item["edges"]
            ]
            inventory.append(item)
        return inventory

    def _packet_process_edge(self, edge: dict[str, Any]) -> dict[str, Any]:
        summary = {
            key: edge[key]
            for key in (
                "source_concept_id",
                "relationship",
                "target_concept_id",
                "status",
            )
        }
        for field_name in (
            "supporting_relation_ids",
            "conditions",
            "supporting_record_ids",
            "boundary_conditions",
        ):
            summary[f"{field_name}_count"] = len(edge[field_name])
            summary[field_name] = self._representative_items(
                edge[field_name], PACKET_PROCESS_EDGE_ITEMS
            )
        summary["supporting_evidence_count"] = len(edge["supporting_evidence"])
        summary["negative_evidence_count"] = len(edge["negative_evidence"])
        return summary

    def _packet_negative_cases(self) -> list[dict[str, Any]]:
        cases: list[dict[str, Any]] = []
        for negative_case in self.state.negative_cases:
            item = asdict(negative_case)
            item["evidence_count"] = len(item["evidence"])
            item["evidence"] = self._representative_items(
                item["evidence"], PACKET_NEGATIVE_CASE_EVIDENCE
            )
            cases.append(item)
        return cases

    def _compact_concept_inventory(self) -> list[dict[str, Any]]:
        return [
            {
                "id": concept.id,
                "label": concept.label,
                "definition": self._compact_text(concept.definition),
                "level": concept.level,
                "status": concept.status,
                "parent_category_id": concept.parent_category_id,
                "child_concept_count": len(concept.child_concept_ids),
                "evidence_count": len(concept.evidence),
                "definition_revision_count": len(concept.definition_revisions),
                "variation_count": len(concept.variations),
                "negative_or_boundary_case_count": len(concept.negative_or_boundary_cases),
            }
            for concept in self.state.concepts
        ]

    def _compact_relation_inventory(self) -> list[dict[str, Any]]:
        return [
            {
                "id": relation.id,
                "source_concept_id": relation.source_concept_id,
                "relationship": relation.relationship,
                "target_concept_id": relation.target_concept_id,
                "status": relation.status,
                "grounding_kind": relation.grounding_kind,
                "conditions": self._representative_items(relation.conditions, 2),
                "condition_count": len(relation.conditions),
                "evidence_count": len(relation.evidence),
                "variation_count": len(relation.variations),
                "negative_case_count": len(relation.negative_cases),
            }
            for relation in self.state.relationships
        ]

    def _compact_process_inventory(self) -> list[dict[str, Any]]:
        inventory: list[dict[str, Any]] = []
        for process in self.state.processes:
            edges = [
                {
                    "source_concept_id": edge.source_concept_id,
                    "relationship": edge.relationship,
                    "target_concept_id": edge.target_concept_id,
                    "status": edge.status,
                }
                for edge in self._representative_items(process.edges, COMPACT_PROCESS_EDGE_ITEMS)
            ]
            inventory.append(
                {
                    "id": process.id,
                    "label": process.label,
                    "description": self._compact_text(process.description),
                    "status": process.status,
                    "edge_count": len(process.edges),
                    "edges": edges,
                    "supporting_relation_count": len(process.supporting_relation_ids),
                    "supporting_record_count": len(process.supporting_record_ids),
                    "negative_case_count": len(process.negative_cases),
                }
            )
        return inventory

    def _compact_memos(self) -> list[dict[str, Any]]:
        return [
            {
                "id": memo.id,
                "topic": memo.topic,
                "analytic_observation": self._compact_text(memo.analytic_observation),
                "tentative_interpretation": self._compact_text(memo.tentative_interpretation),
                "linked_concept_ids": memo.linked_concept_ids,
                "linked_relation_ids": memo.linked_relation_ids,
                "linked_process_ids": memo.linked_process_ids,
                "supporting_evidence_count": len(memo.supporting_evidence),
                "negative_or_contradictory_case_count": len(memo.negative_or_contradictory_cases),
            }
            for memo in self.state.memos
        ]

    def _compact_negative_cases(self) -> list[dict[str, Any]]:
        return [
            {
                "id": case.id,
                "target_type": case.target_type,
                "target_id": case.target_id,
                "description": self._compact_text(case.description),
                "evidence_count": len(case.evidence),
            }
            for case in self.state.negative_cases
        ]

    @staticmethod
    def _compact_text(value: str, limit: int = COMPACT_TEXT_CHARS) -> str:
        return value if len(value) <= limit else value[: limit - 1].rstrip() + "…"

    @staticmethod
    def _representative_items(items: list[Any], limit: int) -> list[Any]:
        if len(items) <= limit:
            return items
        first_count = limit // 2
        return items[:first_count] + items[-(limit - first_count):]

    def _commit(self, event_type: str, payload: dict[str, Any]) -> None:
        self.state.revision += 1
        self.state.analysis_metadata["last_action"] = event_type
        self.state.analysis_metadata["updated_at"] = utc_now()
        self._write_state()
        with (self.root / EVENTS_FILE).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"revision": self.state.revision, "timestamp": utc_now(), "type": event_type, "payload": payload}, ensure_ascii=False, sort_keys=True) + "\n")
        if event_type == FAILURE_EXTRACTION_ACTION and isinstance(payload.get("failure_memory"), dict):
            self._append_prompt_adaptation_jsonl("failure_memory.jsonl", payload["failure_memory"])
        if event_type in {CRITIC_ACTION, "prompt_adaptation_validation_completed"} and isinstance(payload.get("history"), dict):
            self._append_prompt_adaptation_jsonl("prompt_history.jsonl", payload["history"])

    def _write_state(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        write_json(self.root / STATE_FILE, state_to_dict(self.state))

    def _write_frozen_stage2_prompt(self) -> None:
        initialize_prompt_adaptation_state(self.state)
        path = self.root / "prompt_adaptation" / "frozen-stage2-base.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        value = self.state.prompt_adaptation_state["frozen_base_prompt"] + "\n"
        if path.exists() and path.read_text(encoding="utf-8") != value:
            raise ValueError("refusing to overwrite the frozen adaptive Stage-2 base prompt")
        if not path.exists():
            path.write_text(value, encoding="utf-8")

    def _append_prompt_adaptation_jsonl(self, name: str, value: dict[str, Any]) -> None:
        path = self.root / "prompt_adaptation" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def run_grounded_theory_pipeline(
    project: GroundedTheoryProject,
    complete_packet: Callable[[dict[str, Any]], dict[str, Any]],
) -> GroundedTheoryProject:
    while project.next_action()["action"] != "complete":
        if project.next_action()["action"] == "relational_validation_blocked":
            raise RuntimeError(project.next_action()["reason"])
        project.submit_agent_result(complete_packet(project.task_packet()))
    project.export_outputs()
    return project
