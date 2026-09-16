"""Persistent state machine for the standalone Grounded Theory front end."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from .integration import apply_integration
from .models import GroundedTheoryConfig, GroundedTheoryState, QualitativeRecord, utc_now
from .open_coding import apply_open_coding
from .packets import (
    integration_output_schema,
    load_prompt,
    open_coding_output_schema,
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
from .review import next_relational_review_material
from .validation import object_list, remaining_record_ids


PACKET_EVIDENCE_PER_CONCEPT = 2
PACKET_CONCEPT_HISTORY_ITEMS = 1
PACKET_MEMOS = 4
PACKET_MEMO_EVIDENCE = 1
PACKET_RELATION_EVIDENCE = 2
PACKET_RELATION_HISTORY_ITEMS = 1
PACKET_PROCESS_ITEMS = 3
PACKET_NEGATIVE_CASES = 8
PACKET_NEGATIVE_CASE_EVIDENCE = 2


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
        return {
            "schema_version": self.state.schema_version,
            "revision": self.state.revision,
            "counts": {
                "records": len(self.state.records),
                "open_coded_records": len(self.state.open_coded_record_ids),
                "relationally_analyzed_records": len(self.state.relationally_analyzed_record_ids),
                "concepts": len(self.state.concepts),
                "relationships": len(self.state.relationships),
                "processes": len(self.state.processes),
                "memos": len(self.state.memos),
                "negative_cases": len(self.state.negative_cases),
                "sampling_needs": len(self.state.theoretical_sampling_needs),
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

    def requeue_skipped_relational_batches(self) -> dict[str, Any]:
        """Restore skipped Stage 2 records after increasing runner capacity."""
        from .recovery import requeue_skipped_relational_batches

        return requeue_skipped_relational_batches(self)

    def next_action(self) -> dict[str, Any]:
        if self.state.relational_validation_blocked:
            return {
                "action": "relational_validation_blocked",
                "reason": "an independent grounding review rejected the same relational batch twice; revise it instead of accepting an unsupported relation",
                "feedback": list(self.state.relational_validation_feedback),
            }
        remaining_open = remaining_record_ids(self.state.records, self.state.open_coded_record_ids)
        if remaining_open:
            return {
                "action": "open_coding",
                "record_ids": remaining_open[: self.state.config.open_coding_batch_size],
                "reason": "records need open coding and comparison with the live concept inventory",
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
        remaining_relational = remaining_record_ids(
            self.state.records, self.state.relationally_analyzed_record_ids
        )
        if remaining_relational:
            return {
                "action": "relational_process_analysis",
                "record_ids": remaining_relational[: self.state.config.relational_batch_size],
                "reason": "coded records need relational and process analysis against the live, Stage-2-revisable concept inventory",
            }
        if self.state.integrated_theory is None:
            return {
                "action": "theoretical_integration",
                "reason": "concept, relationship, process, memo, and negative-case inventories are ready for integration",
            }
        return {"action": "complete", "reason": "Grounded Theory front-end analysis is complete"}

    def task_packet(self) -> dict[str, Any]:
        action_info = self.next_action()
        action = action_info["action"]
        packet: dict[str, Any] = {
            "action": action,
            "study": asdict(self.state.config),
            "analysis_metadata": deepcopy(self.state.analysis_metadata),
            "rule": "Complete only this stage. Findings must remain grounded in supplied records and verbatim evidence spans; do not use external knowledge.",
            "validation": {
                "evidence_rule": "Every text_span must be a verbatim substring of the supplied record text.",
                "comparison_values": ["SAME", "VARIATION", "NEW", "CONTRADICTION"],
                "no_cooccurrence_rule": "Co-occurrence alone is not a relationship.",
                "stage_feedback_rule": "Stage 2 may submit evidence-grounded concept_updates to revise the live Stage 1 concept/category inventory.",
            },
        }
        if action == "open_coding":
            return self._open_packet(packet, action_info["record_ids"])
        if action == "relational_process_analysis":
            return self._relational_packet(packet, action_info["record_ids"])
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

    def submit_agent_result(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("agent result must be a JSON object")
        action = str(self.next_action()["action"])
        if action == "complete":
            raise ValueError("Grounded Theory analysis is already complete")
        if action == "relational_validation_blocked":
            raise ValueError("relational grounding is blocked for review; do not bypass its feedback")
        trial = deepcopy(self.state)
        if action == "open_coding":
            event = apply_open_coding(trial, payload, self.next_action())
        elif action == "relational_process_analysis":
            event, action = self._stage_or_commit_relations(trial, payload, action)
        elif action == "validate_relational_grounding":
            event = apply_relational_grounding_validation(trial, payload)
        elif action == "theoretical_integration":
            event = apply_integration(trial, payload)
        else:
            raise ValueError(f"unsupported action: {action}")
        self.state = trial
        self._commit(action, event)
        return {"accepted_action": action, "status": self.status()}

    def export_outputs(self, output_dir: str | Path | None = None) -> Path:
        destination = Path(output_dir) if output_dir is not None else self.root / "outputs"
        destination.mkdir(parents=True, exist_ok=True)
        objects: dict[str, Any] = {
            "concepts.json": [asdict(item) for item in self.state.concepts],
            "relationships.json": [asdict(item) for item in self.state.relationships],
            "processes.json": [asdict(item) for item in self.state.processes],
            "memos.json": [asdict(item) for item in self.state.memos],
            "negative_cases.json": [asdict(item) for item in self.state.negative_cases],
            "theoretical_sampling_needs.json": [asdict(item) for item in self.state.theoretical_sampling_needs],
            "integrated_theory.json": asdict(self.state.integrated_theory) if self.state.integrated_theory else {"status": "not_yet_integrated"},
            "analysis_state.json": state_to_dict(self.state),
        }
        for name, value in objects.items():
            write_json(destination / name, value)
        (destination / "analysis_report.md").write_text(render_analysis_report(self.state), encoding="utf-8")
        return destination

    def _open_packet(self, packet: dict[str, Any], record_ids: list[str]) -> dict[str, Any]:
        packet.update({
            "records": self._records_for_ids(record_ids),
            "concept_inventory": self._packet_concept_inventory(),
            "memos": self._packet_memos(),
            "prompt": load_prompt("gt-01-open-coding.md"),
            "expected_output": open_coding_output_schema(record_ids),
        })
        return packet

    def _relational_packet(self, packet: dict[str, Any], record_ids: list[str]) -> dict[str, Any]:
        packet.update({
            "records": self._records_for_ids(record_ids),
            "concept_inventory": self._packet_concept_inventory(),
            "relation_inventory": self._packet_relation_inventory(),
            "process_inventory": self._packet_process_inventory(),
            "memos": self._packet_memos(),
            "negative_cases": self._packet_negative_cases(),
            "validator_feedback": list(self.state.relational_validation_feedback),
            "prompt": load_prompt("gt-02-relational-process.md"),
            "expected_output": relational_output_schema(record_ids),
        })
        return packet

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
            "expected_output": relational_grounding_review_output_schema(),
        })
        return packet

    def _stage_or_commit_relations(
        self, trial: GroundedTheoryState, payload: dict[str, Any], action: str
    ) -> tuple[dict[str, Any], str]:
        preview = deepcopy(self.state)
        preview_event = apply_relational_analysis(preview, payload, self.next_action())
        relation_updates = object_list(payload, "relationship_updates", allow_empty=True)
        process_updates = object_list(payload, "process_updates", allow_empty=True)
        if not relation_updates and not process_updates:
            return apply_relational_analysis(trial, payload, self.next_action()), action
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
        memos: list[dict[str, Any]] = []
        for memo in self._representative_items(self.state.memos, PACKET_MEMOS):
            item = asdict(memo)
            for field_name in (
                "supporting_evidence",
                "negative_or_contradictory_cases",
            ):
                item[f"{field_name}_count"] = len(item[field_name])
                item[field_name] = self._representative_items(
                    item[field_name], PACKET_MEMO_EVIDENCE
                )
            memos.append(item)
        return memos

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
            inventory.append(item)
        return inventory

    def _packet_negative_cases(self) -> list[dict[str, Any]]:
        cases: list[dict[str, Any]] = []
        for negative_case in self._representative_items(
            self.state.negative_cases, PACKET_NEGATIVE_CASES
        ):
            item = asdict(negative_case)
            item["evidence_count"] = len(item["evidence"])
            item["evidence"] = self._representative_items(
                item["evidence"], PACKET_NEGATIVE_CASE_EVIDENCE
            )
            cases.append(item)
        return cases

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

    def _write_state(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        write_json(self.root / STATE_FILE, state_to_dict(self.state))


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
