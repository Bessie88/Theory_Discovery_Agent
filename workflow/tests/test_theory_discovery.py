from __future__ import annotations

import json
import asyncio
import tempfile
import hashlib
import unittest
from pathlib import Path

from theory_discovery import TheoryDiscoveryProject, run
from theory_discovery.frozen_execution import (
    _evaluate_expression,
    evaluate_falsification_specification,
)


class TheoryDiscoveryProjectTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name) / "study"
        self.project = TheoryDiscoveryProject.create(
            self.root,
            study_id="ai-reskilling",
            question="Why does perceived AI employment threat lead to reskilling?",
            evidence_records=[
                {
                    "id": "ev_1",
                    "source": "Example paper",
                    "summary": "A source-backed finding about control and reskilling.",
                }
            ],
            graph_source="discovery-graph.json",
            discovery_documents=[
                {
                    "id": "source_1",
                    "text": "A public post describes AI employment threat and a loss of career control.",
                    "source": "Example source corpus",
                    "locator": "record 1",
                }
            ],
            validation_data={
                "unit_of_analysis": "One public post",
                "population": "English-language AI-employment posts",
                "data_source": "Held-out evaluation corpus",
                "time_window": "Pre-specified window",
                "required_data_fields": [
                    "record_id",
                    "text",
                    "loss_of_control",
                    "reskilling",
                ],
                "partition_role": "held_out",
                "separation_note": "These records were held out before graph construction and evidence selection.",
            },
        )

    def import_graph(self) -> None:
        self.project.import_graph(
            nodes=[
                {"id": "threat", "label": "AI employment threat"},
                {"id": "control", "label": "Career control"},
                {"id": "learning", "label": "Reskilling"},
            ]
        )

    def add_theory_and_prediction(self):
        self.project.add_relationship(
            "threat",
            "control",
            "mechanism",
            "negative",
            "Threat may reduce perceived control.",
        )
        theory = self.project.add_theory(
            name="Control restoration",
            description="Threat can be linked to active reskilling through reduced career control.",
            theory_statements=[
                {
                    "id": "law_control_restoration",
                    "law": "AI employment threat increases reskilling when it reduces career control.",
                    "scope": "People with access to learning resources.",
                    "evidence": [
                        {
                            "id": "ev_1",
                            "summary": "A source-backed finding about control and reskilling.",
                        }
                    ],
                }
            ],
            conflicting_evidence=[],
            unaccounted_evidence=[],
            new_predictions_likely=["Reduced-control groups report more active reskilling."],
            new_predictions_unknown=[],
            negative_experiments=["No difference after a control-restoration intervention."],
        )
        prediction = self.project.add_prediction(
            theory_id=theory.id,
            law_id="law_control_restoration",
            specific_prediction="Loss-of-control posts contain more active reskilling reports.",
            operational_signals=["loss-of-control language", "active reskilling report"],
            strong_test_requirement="Compare threat posts with and without loss-of-control language.",
            support_criteria="The loss-of-control group has more active reskilling reports.",
            contradiction_criteria="The loss-of-control group has no more active reskilling reports.",
        )
        return theory, prediction

    def falsification_test_payload(self, prediction_id: str) -> dict[str, object]:
        return {
            "prediction_id": prediction_id,
            "test_description": "Compare active reskilling reports in posts with and without loss-of-control language.",
            "null_hypothesis": "The two groups have the same observed active-reskilling pattern.",
            "alternative_hypothesis": "Posts with loss-of-control language show more active reskilling.",
            "required_data": ["text", "loss_of_control", "reskilling"],
            "missing_information": [],
            "relevance": "direct",
            "implementability": "ready",
        }

    def falsification_specification_payload(
        self, prediction_id: str
    ) -> dict[str, object]:
        return {
            "prediction_id": prediction_id,
            "template_id": "directional_prevalence_contrast_v1",
            "measurable_implication": {
                "condition": "condition",
                "outcome": "outcome",
                "relation": "higher_prevalence",
            },
            "measurement": {
                "condition": "measure_loss_of_control",
                "outcome": "measure_reskilling",
            },
            "population": {"include": "all_valid_records", "exclude": []},
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
            "required_fields": ["record_id", "text"],
            "unresolved": [],
            "readiness": "ready",
        }

    def explicit_group_specification_payload(
        self, prediction_id: str, *, majority: bool = False
    ) -> dict[str, object]:
        template_id = (
            "majority_group_prevalence_contrast_v1"
            if majority
            else "explicit_group_prevalence_contrast_v1"
        )
        relation = (
            "higher_prevalence_with_group_a_majority"
            if majority
            else "higher_prevalence"
        )
        expected = (
            "observed_difference_positive_and_group_a_outcome_rate_above_one_half"
            if majority
            else "observed_difference_positive"
        )
        opposite = (
            "observed_difference_non_positive_or_group_a_outcome_rate_not_above_one_half"
            if majority
            else "observed_difference_non_positive"
        )
        decision_type = (
            "explicit_group_majority_directional_prevalence"
            if majority
            else "explicit_group_directional_prevalence"
        )
        return {
            "prediction_id": prediction_id,
            "template_id": template_id,
            "measurable_implication": {
                "condition": "condition",
                "outcome": "outcome",
                "relation": relation,
            },
            "measurement": {
                "condition": "measure_loss_of_control",
                "outcome": "measure_reskilling",
            },
            "expressions": {
                "group_a": {"op": "measurement", "measurement": "condition"},
                "group_b": {
                    "op": "not",
                    "operand": {"op": "measurement", "measurement": "condition"},
                },
                "outcome": {"op": "measurement", "measurement": "outcome"},
            },
            "population": {"include": "all_valid_records", "exclude": []},
            "comparison": {
                "group_a": "expression_group_a_is_true_and_group_b_is_false",
                "group_b": "expression_group_b_is_true_and_group_a_is_false",
                "estimand": (
                    "outcome_prevalence_difference_and_group_a_majority"
                    if majority
                    else "outcome_prevalence_difference"
                ),
            },
            "hypotheses": {"expected": expected, "opposite": opposite},
            "decision_rule": {
                "type": decision_type,
                "consistent_when": expected,
                "contradicted_when": opposite,
                "not_testable_when": "comparison_group_empty_or_required_field_missing",
            },
            "required_fields": ["record_id", "text"],
            "unresolved": [],
            "readiness": "ready",
        }

    def freeze_measurements(self, prediction_id: str) -> None:
        self.project.add_measurement_specification(
            "loss_of_control",
            "Does the text express a loss of personal career control?",
            [prediction_id],
            specification_id="measure_loss_of_control",
        )
        self.project.add_measurement_specification(
            "reskilling",
            "Does the text describe active reskilling or learning activity?",
            [prediction_id],
            specification_id="measure_reskilling",
        )

    def complete_frozen_measurements(self) -> None:
        records_path = self.root.parent / "validation.csv"
        input_hash = hashlib.sha256(records_path.read_bytes()).hexdigest()
        for specification in self.project.state.measurement_specifications:
            artifact_path = self.root.parent / "validation-measurements" / f"{specification.id}.json"
            artifact_path.parent.mkdir(exist_ok=True)
            artifact = {
                "measurement_specification_id": specification.id,
                "specification_hash": specification.specification_hash,
                "input_records_hash": input_hash,
                "labels": [
                    {"record_id": "1", "label": "PRESENT", "parse_status": "valid_json_label"}
                ],
            }
            artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
            self.project.record_measurement_run(
                specification.id,
                input_records_hash=input_hash,
                output_artifact=artifact_path.relative_to(self.root.parent).as_posix(),
                output_hash=hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
                record_count=1,
            )

    def test_five_object_stages_reach_ready(self) -> None:
        self.assertEqual(self.project.next_action()["action"], "import_graph")
        self.import_graph()
        self.assertEqual(self.project.next_action()["action"], "discover_relationships")
        theory, prediction = self.add_theory_and_prediction()
        self.assertEqual(self.project.next_action()["action"], "generate_measurement_specifications")
        self.freeze_measurements(prediction.id)
        self.assertEqual(self.project.next_action()["action"], "compile_falsification_specifications")

        self.project.submit(
            {
                "falsification_specifications": [
                    self.falsification_specification_payload(prediction.id)
                ]
            }
        )
        self.assertEqual(self.project.next_action()["action"], "ready_for_review")
        self.assertEqual(self.project.workflow_status(), "ready")

        reopened = TheoryDiscoveryProject.open(self.root)
        self.assertEqual(reopened.state.revision, self.project.state.revision)
        self.assertEqual(
            reopened.state.theories[0].theory_statements[0].law,
            theory.theory_statements[0].law,
        )
        self.assertEqual(len(reopened.state.falsification_specifications), 1)
        self.assertTrue((self.root / "events.jsonl").exists())

    def test_validation_is_separate_from_readiness(self) -> None:
        self.import_graph()
        _, prediction = self.add_theory_and_prediction()
        self.freeze_measurements(prediction.id)
        specification = self.project.add_falsification_specification(
            **self.falsification_specification_payload(prediction.id)
        )
        result = self.project.record_validation(
            prediction.id,
            "not_testable",
            "The corpus was not collected yet.",
            [],
            falsification_specification_id=specification.id,
            specification_hash=specification.specification_hash,
        )
        self.assertEqual(result.status, "not_testable")
        self.assertEqual(self.project.workflow_status(), "ready")
        self.assertEqual(self.project.status()["counts"]["validation_results"], 1)

    def test_automatic_execution_requires_matching_reproducible_artifacts(self) -> None:
        self.import_graph()
        _, prediction = self.add_theory_and_prediction()
        self.freeze_measurements(prediction.id)
        specification = self.project.add_falsification_specification(
            **self.falsification_specification_payload(prediction.id)
        )

        (self.root.parent / "validation.csv").write_text(
            "record_id,text,loss_of_control,reskilling\n1,example,yes,yes\n",
            encoding="utf-8",
        )
        assert self.project.state.validation_data is not None
        self.project.state.validation_data.data_source = "validation.csv"
        self.project.configure_validation_execution(True)

        self.assertEqual(self.project.next_action()["action"], "measure_validation_records")
        packet = self.project.task_packet()
        self.assertNotIn("discovery_documents", packet)
        self.assertNotIn("evidence_records", packet)
        self.assertNotIn("graph", packet)
        self.assertNotIn("predictions", packet)
        self.assertTrue(packet["validation_records_path"].endswith("validation.csv"))
        self.assertEqual(packet["measurement_specification"]["id"], "measure_loss_of_control")
        self.complete_frozen_measurements()
        self.assertEqual(self.project.next_action()["action"], "execute_falsification_tests")
        packet = self.project.task_packet()
        self.assertNotIn("validation_records_path", packet)

        execution_dir = self.root.parent / "validation-execution"
        execution_dir.mkdir()
        artifact_name = f"{prediction.id}.py"
        (execution_dir / artifact_name).write_text(
            "# Deterministic execution artifact for this test.\n",
            encoding="utf-8",
        )
        output = {
            "validation_results": [
                {
                    "id": "validation_auto_1",
                    "prediction_id": prediction.id,
                    "status": "not_testable",
                    "summary": "The frozen comparison group is empty.",
                    "evidence": [],
                    "falsification_specification_id": specification.id,
                    "specification_hash": specification.specification_hash,
                    "execution_artifact": f"validation-execution/{artifact_name}",
                }
            ]
        }
        (execution_dir / f"{prediction.id}.json").write_text(
            json.dumps(output), encoding="utf-8"
        )
        result = self.project.submit(output)
        self.assertEqual(result["accepted_action"], "execute_falsification_tests")
        self.assertEqual(result["next_action"]["action"], "complete")
        self.assertEqual(self.project.workflow_status(), "ready")
        self.assertEqual(
            self.project.state.validation_results[0].execution_artifact,
            f"validation-execution/{artifact_name}",
        )

    def test_falsification_specification_can_be_recompiled_in_place(self) -> None:
        self.import_graph()
        _, prediction = self.add_theory_and_prediction()
        self.freeze_measurements(prediction.id)
        first = self.project.add_falsification_specification(
            **self.falsification_specification_payload(prediction.id)
        )
        revised = self.project.add_falsification_specification(
            **{
                **self.falsification_specification_payload(prediction.id),
                "specification_id": first.id,
            }
        )
        self.assertEqual(revised.id, first.id)

    def test_task_packet_uses_theory_and_prediction_schema(self) -> None:
        self.import_graph()
        self.assertEqual(self.project.task_packet()["action"], "discover_relationships")
        self.project.add_relationship(
            "threat", "control", "association", "negative", "A candidate association."
        )
        packet = self.project.task_packet()
        self.assertEqual(packet["action"], "generate_theories")
        expected = packet["expected_output"]["theories"][0]
        self.assertIn("description", expected)
        self.assertIn("theory_statements", expected)
        self.assertIn("conflicting_evidence", expected)
        self.assertIn("ev_1", {item["id"] for item in packet["evidence_records"]})
        self.assertIn("source_1", {item["id"] for item in packet["discovery_documents"]})

    def test_falsification_packet_excludes_discovery_material(self) -> None:
        self.import_graph()
        _, prediction = self.add_theory_and_prediction()
        self.freeze_measurements(prediction.id)
        packet = self.project.task_packet()
        self.assertEqual(packet["action"], "compile_falsification_specifications")
        self.assertIn("validation_data", packet)
        self.assertNotIn("discovery_documents", packet)
        self.assertNotIn("evidence_records", packet)
        self.assertNotIn("graph", packet)
        self.assertNotIn("graph_source", packet["study"])
        self.assertEqual(packet["falsification_isolation"]["partition_role"], "held_out")

    def test_measurement_generation_is_discovery_only_and_freezes_contract(self) -> None:
        self.import_graph()
        _, prediction = self.add_theory_and_prediction()
        packet = self.project.task_packet()
        self.assertEqual(packet["action"], "generate_measurement_specifications")
        self.assertIn("discovery_documents", packet)
        self.assertNotIn("validation_data", packet)
        self.assertNotIn("validation_records_path", packet)
        self.assertNotIn("validation_results", packet)
        self.assertEqual(packet["frozen_measurement_protocol"]["temperature"], 0.0)
        self.assertEqual(
            packet["frozen_measurement_protocol"]["model_id"],
            "qwen3.6-27b-fp8",
        )
        self.freeze_measurements(prediction.id)
        with self.assertRaisesRegex(ValueError, "measurement specifications are frozen"):
            self.project.add_measurement_specification(
                "loss_of_control",
                "Does the text describe a different construct?",
                [prediction.id],
                specification_id="measure_loss_of_control",
            )

    def test_deterministic_executor_uses_only_frozen_labels(self) -> None:
        self.import_graph()
        _, prediction = self.add_theory_and_prediction()
        self.freeze_measurements(prediction.id)
        specification = self.project.add_falsification_specification(
            **self.falsification_specification_payload(prediction.id)
        )
        status, _, audit = evaluate_falsification_specification(
            specification,
            {
                "measure_loss_of_control": {"a": "PRESENT", "b": "ABSENT", "c": "UNCERTAIN"},
                "measure_reskilling": {"a": "PRESENT", "b": "ABSENT", "c": "PRESENT"},
            },
        )
        self.assertEqual(status, "prediction_consistent")
        self.assertEqual(audit["usable_records"], 2)

    def test_explicit_group_executor_preserves_frozen_boolean_groups(self) -> None:
        self.import_graph()
        _, prediction = self.add_theory_and_prediction()
        self.freeze_measurements(prediction.id)
        specification = self.project.add_falsification_specification(
            **self.explicit_group_specification_payload(prediction.id)
        )
        status, _, audit = evaluate_falsification_specification(
            specification,
            {
                "measure_loss_of_control": {
                    "a": "PRESENT",
                    "b": "PRESENT",
                    "c": "ABSENT",
                    "d": "ABSENT",
                    "e": "UNCERTAIN",
                },
                "measure_reskilling": {
                    "a": "PRESENT",
                    "b": "PRESENT",
                    "c": "ABSENT",
                    "d": "ABSENT",
                    "e": "PRESENT",
                },
            },
        )
        self.assertEqual(status, "prediction_consistent")
        self.assertEqual(audit["group_a_records"], 2)
        self.assertEqual(audit["group_b_records"], 2)
        self.assertEqual(audit["uncertain_records_excluded"], 1)

    def test_majority_template_requires_a_strict_majority(self) -> None:
        self.import_graph()
        _, prediction = self.add_theory_and_prediction()
        self.freeze_measurements(prediction.id)
        specification = self.project.add_falsification_specification(
            **self.explicit_group_specification_payload(prediction.id, majority=True)
        )
        status, _, audit = evaluate_falsification_specification(
            specification,
            {
                "measure_loss_of_control": {
                    "a": "PRESENT",
                    "b": "PRESENT",
                    "c": "ABSENT",
                },
                "measure_reskilling": {
                    "a": "PRESENT",
                    "b": "ABSENT",
                    "c": "ABSENT",
                },
            },
        )
        self.assertEqual(status, "observed_directional_contradiction")
        self.assertEqual(audit["outcome_prevalence_group_a"], 0.5)

    def test_boolean_expression_language_propagates_uncertainty(self) -> None:
        labels = {
            "first": {"present": "PRESENT", "uncertain": "PRESENT", "absent": "PRESENT"},
            "second": {"present": "PRESENT", "uncertain": "UNCERTAIN", "absent": "ABSENT"},
            "third": {"present": "ABSENT", "uncertain": "ABSENT", "absent": "ABSENT"},
        }
        at_least_two = {
            "op": "at_least",
            "minimum": 2,
            "operands": [
                {"op": "measurement", "measurement": "first"},
                {"op": "measurement", "measurement": "second"},
                {"op": "measurement", "measurement": "third"},
            ],
        }
        self.assertEqual(_evaluate_expression(at_least_two, "present", labels), "PRESENT")
        self.assertEqual(_evaluate_expression(at_least_two, "uncertain", labels), "UNCERTAIN")
        self.assertEqual(_evaluate_expression(at_least_two, "absent", labels), "ABSENT")
        any_or_not = {
            "op": "any",
            "operands": [
                {"op": "measurement", "measurement": "third"},
                {
                    "op": "not",
                    "operand": {"op": "measurement", "measurement": "second"},
                },
            ],
        }
        self.assertEqual(_evaluate_expression(any_or_not, "absent", labels), "PRESENT")

    def test_ready_test_requires_an_isolated_validation_partition(self) -> None:
        self.import_graph()
        _, prediction = self.add_theory_and_prediction()
        self.freeze_measurements(prediction.id)
        assert self.project.state.validation_data is not None
        self.project.state.validation_data.partition_role = "overlapping"
        with self.assertRaisesRegex(ValueError, "held_out or independent"):
            self.project.add_falsification_test(
                **self.falsification_test_payload(prediction.id)
            )

    def test_compiler_marks_an_unregistered_measurement_rule_not_ready(self) -> None:
        self.import_graph()
        _, prediction = self.add_theory_and_prediction()
        self.freeze_measurements(prediction.id)
        payload = self.falsification_specification_payload(prediction.id)
        payload.update(
            {
                "measurement": {
                    "condition": "missing_condition_rule",
                    "outcome": "measure_reskilling",
                },
                "required_fields": ["record_id", "text"],
                "readiness": "not_ready",
            }
        )
        specification = self.project.add_falsification_specification(**payload)
        self.assertEqual(specification.readiness, "not_ready")
        self.assertIn(
            "unknown_measurement_specification:missing_condition_rule",
            specification.unresolved,
        )
        assessment = self.project.state.readiness_assessments[-1]
        self.assertEqual(assessment.parent_type, "falsification_specification")
        self.assertEqual(assessment.status, "blocked")

    def test_compiler_rejects_a_false_ready_claim(self) -> None:
        self.import_graph()
        _, prediction = self.add_theory_and_prediction()
        self.freeze_measurements(prediction.id)
        payload = self.falsification_specification_payload(prediction.id)
        payload.update(
            {
                "measurement": {
                    "condition": "missing_condition_rule",
                    "outcome": "measure_reskilling",
                },
                "required_fields": ["record_id", "text"],
            }
        )
        with self.assertRaisesRegex(ValueError, "deterministic verifier: not_ready"):
            self.project.add_falsification_specification(**payload)

    def test_missing_atomic_construct_triggers_bounded_repair_then_recompile(self) -> None:
        self.import_graph()
        _, prediction = self.add_theory_and_prediction()
        self.freeze_measurements(prediction.id)
        blocked = self.falsification_specification_payload(prediction.id)
        blocked.update(
            {
                "measurement": {
                    "condition": "measure_loss_of_control",
                    "outcome": "missing_workload_pressure",
                },
                "unresolved": [
                    "No frozen measurement specification exists for workload_pressure."
                ],
                "missing_measurement_constructs": ["workload_pressure"],
                "readiness": "not_ready",
            }
        )
        self.project.add_falsification_specification(**blocked)

        repair = self.project.next_action()
        self.assertEqual(repair["action"], "repair_measurement_specifications")
        self.assertEqual(repair["repair_round"], 1)
        packet = self.project.task_packet()
        self.assertNotIn("validation_records_path", packet)
        self.assertEqual(
            packet["measurement_repair_requests"][0]["construct_id"],
            "workload_pressure",
        )
        response = self.project.submit(
            {
                "measurement_specifications": [
                    {
                        "id": "measure_workload_pressure",
                        "construct_id": "workload_pressure",
                        "measurement_question": "Does the text describe pressure from academic workload or deadlines?",
                        "source_prediction_ids": [prediction.id],
                    }
                ]
            }
        )
        self.assertEqual(response["accepted_action"], "repair_measurement_specifications")
        self.assertEqual(self.project.state.measurement_repair_rounds, 1)
        self.assertEqual(
            self.project.next_action()["action"], "compile_falsification_specifications"
        )

        repaired = self.falsification_specification_payload(prediction.id)
        repaired["measurement"]["outcome"] = "measure_workload_pressure"
        self.project.add_falsification_specification(**repaired)

        self.assertEqual(self.project.state.falsification_repair_prediction_ids, [])
        self.assertEqual(len(self.project.state.superseded_falsification_specifications), 1)
        self.assertEqual(self.project.next_action()["action"], "ready_for_review")

    def test_measurement_repair_rejects_extra_or_renamed_constructs(self) -> None:
        self.import_graph()
        _, prediction = self.add_theory_and_prediction()
        self.freeze_measurements(prediction.id)
        blocked = self.falsification_specification_payload(prediction.id)
        blocked.update(
            {
                "measurement": {
                    "condition": "measure_loss_of_control",
                    "outcome": "missing_workload_pressure",
                },
                "unresolved": ["workload_pressure is absent"],
                "missing_measurement_constructs": ["workload_pressure"],
                "readiness": "not_ready",
            }
        )
        self.project.add_falsification_specification(**blocked)

        with self.assertRaisesRegex(ValueError, "exactly the compiler-diagnosed"):
            self.project.submit(
                {
                    "measurement_specifications": [
                        {
                            "id": "measure_extra",
                            "construct_id": "extra_construct",
                            "measurement_question": "Does the text express an extra construct?",
                            "source_prediction_ids": [prediction.id],
                        }
                    ]
                }
            )

    def test_prediction_revision_invalidates_only_its_unused_contract(self) -> None:
        self.import_graph()
        _, prediction = self.add_theory_and_prediction()
        self.freeze_measurements(prediction.id)
        self.project.add_falsification_specification(
            **self.falsification_specification_payload(prediction.id)
        )

        self.project.revise_prediction_before_measurement(
            prediction.id,
            "Loss-of-control posts contain more active reskilling reports after revision.",
            ["loss-of-control language", "active reskilling report"],
            "Compare threat posts with and without loss-of-control language.",
            "The loss-of-control group has more active reskilling reports.",
            "The loss-of-control group has no more active reskilling reports.",
        )

        self.assertEqual(self.project.state.measurement_runs, [])
        self.assertEqual(self.project.state.falsification_specifications, [])
        self.assertEqual(len(self.project.state.superseded_falsification_specifications), 1)
        self.assertEqual(
            self.project.next_action()["action"], "compile_falsification_specifications"
        )
        self.assertEqual(
            self.project.next_action()["prediction_ids"], [prediction.id]
        )

    def test_new_theory_and_prediction_contract_submits(self) -> None:
        self.import_graph()
        self.project.add_relationship(
            "threat", "control", "association", "negative", "A candidate association."
        )
        self.project.submit(
            {
                "theories": [
                    {
                        "id": "theory_submitted",
                        "name": "Submitted theory",
                        "description": "A theory submitted using the new multi-law contract.",
                        "theory_statements": [
                            {
                                "id": "law_submitted",
                                "law": "Threat changes perceived control.",
                                "scope": "People in the supplied study population.",
                                "evidence": [
                                    {
                                        "id": "ev_1",
                                        "summary": "A source-backed finding about control and reskilling.",
                                    }
                                ],
                            }
                        ],
                        "conflicting_evidence": [],
                        "unaccounted_evidence": [],
                        "new_predictions_likely": [],
                        "new_predictions_unknown": [],
                        "negative_experiments": [],
                    }
                ]
            }
        )
        self.assertEqual(self.project.next_action()["action"], "generate_predictions")
        self.project.submit(
            {
                "predictions": [
                    {
                        "id": "prediction_submitted",
                        "theory_id": "theory_submitted",
                        "law_id": "law_submitted",
                        "specific_prediction": "Threat reports include lower perceived control.",
                        "operational_signals": [
                            "threat language",
                            "perceived-control language",
                        ],
                        "strong_test_requirement": "Compare threat reports with a defined baseline.",
                        "support_criteria": "Threat reports show lower perceived control.",
                        "contradiction_criteria": "Threat reports do not show lower perceived control.",
                    }
                ]
            }
        )
        self.assertEqual(self.project.next_action()["action"], "generate_measurement_specifications")

    def test_predictions_must_reference_a_law_in_their_theory(self) -> None:
        self.import_graph()
        theory, _ = self.add_theory_and_prediction()
        with self.assertRaisesRegex(ValueError, "does not belong"):
            self.project.add_prediction(
                theory_id=theory.id,
                law_id="missing_law",
                specific_prediction="A testable claim.",
                operational_signals=["observable signal"],
                strong_test_requirement="Compare a defined condition with a baseline.",
                support_criteria="The expected observable pattern appears.",
                contradiction_criteria="The expected observable pattern does not appear.",
            )

    def test_theories_cannot_cite_unknown_evidence(self) -> None:
        self.import_graph()
        self.project.add_relationship(
            "threat", "control", "association", "negative", "A candidate association."
        )
        with self.assertRaisesRegex(ValueError, "unknown evidence IDs"):
            self.project.add_theory(
                name="Unsupported theory",
                description="A theory with an invalid evidence reference.",
                theory_statements=[
                    {
                        "law": "A condition changes an outcome.",
                        "scope": "The supplied domain.",
                        "evidence": [{"id": "invented", "summary": "Not supplied."}],
                    }
                ],
                conflicting_evidence=[],
                unaccounted_evidence=[],
                new_predictions_likely=[],
                new_predictions_unknown=[],
                negative_experiments=[],
            )

    def test_evidence_summaries_must_be_canonical(self) -> None:
        self.import_graph()
        self.project.add_relationship(
            "threat", "control", "association", "negative", "A candidate association."
        )
        with self.assertRaisesRegex(ValueError, "summary must exactly match"):
            self.project.add_theory(
                name="Altered evidence theory",
                description="A theory with a rewritten evidence summary.",
                theory_statements=[
                    {
                        "law": "A condition changes an outcome.",
                        "scope": "The supplied domain.",
                        "evidence": [{"id": "ev_1", "summary": "A rewritten claim."}],
                    }
                ],
                conflicting_evidence=[],
                unaccounted_evidence=[],
                new_predictions_likely=[],
                new_predictions_unknown=[],
                negative_experiments=[],
            )

    def test_theories_reject_numerical_claims_not_in_evidence(self) -> None:
        self.import_graph()
        self.project.add_relationship(
            "threat", "control", "association", "negative", "A candidate association."
        )
        with self.assertRaisesRegex(ValueError, "numerical content not present"):
            self.project.add_theory(
                name="Unsupported quantitative theory",
                description="A theory with an unsupported number.",
                theory_statements=[
                    {
                        "law": "A condition increases an outcome by 50%.",
                        "scope": "The supplied domain.",
                        "evidence": [
                            {
                                "id": "ev_1",
                                "summary": "A source-backed finding about control and reskilling.",
                            }
                        ],
                    }
                ],
                conflicting_evidence=[],
                unaccounted_evidence=[],
                new_predictions_likely=[],
                new_predictions_unknown=[],
                negative_experiments=[],
            )

    def test_falsification_tests_reject_unavailable_data_and_thresholds(self) -> None:
        self.import_graph()
        _, prediction = self.add_theory_and_prediction()
        with self.assertRaisesRegex(ValueError, "unavailable data fields"):
            self.project.add_falsification_test(
                **{
                    **self.falsification_test_payload(prediction.id),
                    "required_data": ["invented_variable"],
                }
            )
        with self.assertRaisesRegex(ValueError, "unavailable threshold or sampling window"):
            self.project.add_falsification_test(
                **{
                    **self.falsification_test_payload(prediction.id),
                    "null_hypothesis": "The observed difference has p < 0.05.",
                }
            )
        method_test = self.project.add_falsification_test(
            **{
                **self.falsification_test_payload(prediction.id),
                "test_description": "Use logistic regression to compare the available records.",
            }
        )
        self.assertEqual(method_test.implementability, "ready")

    def test_submit_returns_retry_signal_after_a_regex_failure(self) -> None:
        response = json.loads(
            asyncio.run(
                run(
                    str(self.root),
                    action="submit",
                    payload_json=json.dumps(
                        {"nodes": [{"id": "invalid id", "label": "Invalid"}]}
                    ),
                )
            )
        )
        self.assertFalse(response["accepted"])
        self.assertTrue(response["retry"])
        self.assertIn("graph node ID", response["errors"][0])

    def test_run_init_accepts_separate_discovery_and_validation_inputs(self) -> None:
        root = Path(self.temporary_directory.name) / "run-init-study"
        response = json.loads(
            asyncio.run(
                run(
                    str(root),
                    action="init",
                    payload_json=json.dumps(
                        {
                            "study_id": "run-init",
                            "question": "What changes the outcome?",
                            "evidence_records": [
                                {"id": "ev_run", "summary": "A source-backed finding."}
                            ],
                            "discovery_documents": [
                                {"id": "doc_run", "text": "Original discovery text."}
                            ],
                            "validation_data": {
                                "unit_of_analysis": "One record",
                                "population": "Records held out from discovery",
                                "data_source": "Held-out corpus",
                                "time_window": "Pre-specified period",
                                "required_data_fields": ["record_id", "outcome"],
                                "partition_role": "held_out",
                                "separation_note": "Held out before graph construction.",
                            },
                        }
                    ),
                )
            )
        )
        self.assertEqual(response["counts"]["discovery_documents"], 1)
        self.assertEqual(response["next_action"]["action"], "import_graph")

    def test_invalid_batch_does_not_partially_save(self) -> None:
        self.import_graph()
        with self.assertRaisesRegex(ValueError, "unknown node IDs"):
            self.project.submit(
                {
                    "relationships": [
                        {
                            "source_node_id": "threat",
                            "target_node_id": "control",
                            "relation_type": "association",
                            "direction": "negative",
                            "rationale": "valid first item",
                        },
                        {
                            "source_node_id": "missing",
                            "target_node_id": "control",
                            "relation_type": "association",
                            "direction": "negative",
                            "rationale": "invalid second item",
                        },
                    ]
                }
            )
        self.assertEqual(self.project.state.relationships, [])


if __name__ == "__main__":
    unittest.main()
