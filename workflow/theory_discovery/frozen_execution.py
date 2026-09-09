"""Deterministic execution for a frozen falsification contract.

This module deliberately accepts labels, not source text.  Text interpretation
ends in the frozen measurement worker; this code only excludes UNCERTAIN and
performs the comparison selected before the validation partition was read.
"""

from __future__ import annotations

from collections.abc import Mapping

from .models import FalsificationSpecification


LabelMaps = Mapping[str, Mapping[str, str]]


def _negate(label: str) -> str:
    if label == "PRESENT":
        return "ABSENT"
    if label == "ABSENT":
        return "PRESENT"
    return "UNCERTAIN"


def _evaluate_expression(
    expression: Mapping[str, object], record_id: str, labels_by_alias: LabelMaps
) -> str:
    """Evaluate the closed, three-valued expression language.

    The rules deliberately propagate uncertainty: a composite is PRESENT or
    ABSENT only when that conclusion follows from its frozen component labels.
    """
    operation = expression.get("op")
    if operation == "measurement":
        alias = expression.get("measurement")
        if not isinstance(alias, str):
            return "UNCERTAIN"
        return labels_by_alias.get(alias, {}).get(record_id, "UNCERTAIN")
    if operation == "not":
        operand = expression.get("operand")
        if not isinstance(operand, Mapping):
            return "UNCERTAIN"
        return _negate(_evaluate_expression(operand, record_id, labels_by_alias))
    operands = expression.get("operands")
    if operation not in {"all", "any", "at_least"} or not isinstance(operands, list):
        return "UNCERTAIN"
    if not all(isinstance(operand, Mapping) for operand in operands):
        return "UNCERTAIN"
    labels = [_evaluate_expression(operand, record_id, labels_by_alias) for operand in operands]
    if operation == "all":
        if any(label == "ABSENT" for label in labels):
            return "ABSENT"
        return "PRESENT" if all(label == "PRESENT" for label in labels) else "UNCERTAIN"
    if operation == "any":
        if any(label == "PRESENT" for label in labels):
            return "PRESENT"
        return "ABSENT" if all(label == "ABSENT" for label in labels) else "UNCERTAIN"
    minimum = expression.get("minimum")
    if not isinstance(minimum, int) or isinstance(minimum, bool):
        return "UNCERTAIN"
    present = sum(label == "PRESENT" for label in labels)
    possible = present + sum(label == "UNCERTAIN" for label in labels)
    if present >= minimum:
        return "PRESENT"
    if possible < minimum:
        return "ABSENT"
    return "UNCERTAIN"


def _evaluate_explicit_group_prevalence(
    specification: FalsificationSpecification,
    labels_by_measurement_id: LabelMaps,
) -> tuple[str, str, dict[str, int | float]]:
    required_expressions = ("group_a", "group_b", "outcome")
    if any(name not in specification.expressions for name in required_expressions):
        return "not_testable", "Frozen contract is missing an explicit group or outcome expression.", {}
    labels_by_alias: dict[str, Mapping[str, str]] = {}
    for alias, measurement_id in specification.measurement.items():
        labels = labels_by_measurement_id.get(measurement_id)
        if labels is None:
            return "not_testable", "A required frozen measurement artifact is unavailable.", {}
        labels_by_alias[alias] = labels
    if not labels_by_alias:
        return "not_testable", "Frozen contract has no measurement bindings.", {}
    common_ids = sorted(set.intersection(*(set(labels) for labels in labels_by_alias.values())))
    if not common_ids:
        return "not_testable", "No records are shared by the required frozen measurement artifacts.", {
            "usable_records": 0
        }

    group_a: list[str] = []
    group_b: list[str] = []
    uncertain_records = 0
    neither_or_overlap_records = 0
    for record_id in common_ids:
        group_a_label = _evaluate_expression(
            specification.expressions["group_a"], record_id, labels_by_alias
        )
        group_b_label = _evaluate_expression(
            specification.expressions["group_b"], record_id, labels_by_alias
        )
        outcome_label = _evaluate_expression(
            specification.expressions["outcome"], record_id, labels_by_alias
        )
        if "UNCERTAIN" in {group_a_label, group_b_label, outcome_label}:
            uncertain_records += 1
            continue
        # The template freezes a two-group contrast.  Records satisfying both
        # groups or neither group are not in either comparison group; this is
        # mechanical, not a data-dependent coding decision.
        if group_a_label == "PRESENT" and group_b_label == "ABSENT":
            group_a.append(record_id)
        elif group_b_label == "PRESENT" and group_a_label == "ABSENT":
            group_b.append(record_id)
        else:
            neither_or_overlap_records += 1
    usable_records = len(common_ids) - uncertain_records
    audit_base: dict[str, int | float] = {
        "common_records": len(common_ids),
        "usable_records": usable_records,
        "uncertain_records_excluded": uncertain_records,
        "neither_or_overlap_records_excluded": neither_or_overlap_records,
        "group_a_records": len(group_a),
        "group_b_records": len(group_b),
    }
    if not group_a or not group_b:
        return (
            "not_testable",
            "The frozen explicit-group comparison has an empty group after UNCERTAIN labels are excluded.",
            audit_base,
        )
    outcome_expression = specification.expressions["outcome"]
    group_a_outcomes = sum(
        _evaluate_expression(outcome_expression, record_id, labels_by_alias) == "PRESENT"
        for record_id in group_a
    )
    group_b_outcomes = sum(
        _evaluate_expression(outcome_expression, record_id, labels_by_alias) == "PRESENT"
        for record_id in group_b
    )
    group_a_rate = group_a_outcomes / len(group_a)
    group_b_rate = group_b_outcomes / len(group_b)
    difference = group_a_rate - group_b_rate
    is_majority_template = specification.template_id == "majority_group_prevalence_contrast_v1"
    is_consistent = difference > 0 and (not is_majority_template or group_a_rate > 0.5)
    status = "prediction_consistent" if is_consistent else "observed_directional_contradiction"
    majority_note = " and above one half" if is_majority_template else ""
    return (
        status,
        "Frozen explicit-group comparison: outcome prevalence was "
        f"{group_a_rate:.6f} in group A and {group_b_rate:.6f} in group B"
        f" (difference {difference:.6f}){majority_note}.",
        {
            **audit_base,
            "outcome_present_in_group_a": group_a_outcomes,
            "outcome_present_in_group_b": group_b_outcomes,
            "outcome_prevalence_group_a": group_a_rate,
            "outcome_prevalence_group_b": group_b_rate,
            "outcome_prevalence_difference": difference,
        },
    )


def evaluate_falsification_specification(
    specification: FalsificationSpecification,
    labels_by_measurement_id: LabelMaps,
) -> tuple[str, str, dict[str, int | float]]:
    """Return deterministic status, plain summary, and audit counts.

    The only decision is the template's already-frozen comparison.  There are
    intentionally no p-values, effect-size thresholds, statistical methods,
    or post-hoc exclusions.
    """
    template_id = specification.template_id
    if template_id in {
        "explicit_group_prevalence_contrast_v1",
        "majority_group_prevalence_contrast_v1",
    }:
        return _evaluate_explicit_group_prevalence(specification, labels_by_measurement_id)

    condition_id = specification.measurement.get("condition")
    outcome_id = specification.measurement.get("outcome")
    if not condition_id or not outcome_id:
        return "not_testable", "Frozen contract is missing a condition or outcome binding.", {}
    conditions = labels_by_measurement_id.get(condition_id)
    outcomes = labels_by_measurement_id.get(outcome_id)
    if conditions is None or outcomes is None:
        return "not_testable", "A required frozen measurement artifact is unavailable.", {}
    common_ids = sorted(set(conditions) & set(outcomes))
    usable = [
        record_id
        for record_id in common_ids
        if conditions[record_id] != "UNCERTAIN" and outcomes[record_id] != "UNCERTAIN"
    ]
    if not usable:
        return "not_testable", "No records have non-UNCERTAIN labels for both frozen measures.", {"usable_records": 0}

    if template_id in {"directional_prevalence_contrast_v1", "cooccurrence_v1"}:
        condition_present = [
            record_id for record_id in usable if conditions[record_id] == "PRESENT"
        ]
        condition_absent = [
            record_id for record_id in usable if conditions[record_id] == "ABSENT"
        ]
        if not condition_present or not condition_absent:
            return (
                "not_testable",
                "The frozen comparison has an empty condition group after UNCERTAIN labels are excluded.",
                {
                    "usable_records": len(usable),
                    "condition_present_records": len(condition_present),
                    "condition_absent_records": len(condition_absent),
                },
            )
        present_outcomes = sum(outcomes[record_id] == "PRESENT" for record_id in condition_present)
        absent_outcomes = sum(outcomes[record_id] == "PRESENT" for record_id in condition_absent)
        present_rate = present_outcomes / len(condition_present)
        absent_rate = absent_outcomes / len(condition_absent)
        difference = present_rate - absent_rate
        status = "prediction_consistent" if difference > 0 else "observed_directional_contradiction"
        return (
            status,
            "Frozen directional comparison: outcome prevalence was "
            f"{present_rate:.6f} when the condition was PRESENT and "
            f"{absent_rate:.6f} when it was ABSENT.",
            {
                "usable_records": len(usable),
                "condition_present_records": len(condition_present),
                "condition_absent_records": len(condition_absent),
                "outcome_present_when_condition_present": present_outcomes,
                "outcome_present_when_condition_absent": absent_outcomes,
                "outcome_prevalence_difference": difference,
            },
        )

    if template_id == "necessary_condition_v1":
        outcome_present = [
            record_id for record_id in usable if outcomes[record_id] == "PRESENT"
        ]
        if not outcome_present:
            return (
                "not_testable",
                "The frozen necessary-condition comparison has no outcome-PRESENT records.",
                {"usable_records": len(usable), "outcome_present_records": 0},
            )
        condition_absent = sum(
            conditions[record_id] == "ABSENT" for record_id in outcome_present
        )
        status = "prediction_consistent" if condition_absent == 0 else "observed_directional_contradiction"
        return (
            status,
            "Frozen necessary-condition comparison found "
            f"{condition_absent} condition-ABSENT records among "
            f"{len(outcome_present)} outcome-PRESENT records.",
            {
                "usable_records": len(usable),
                "outcome_present_records": len(outcome_present),
                "condition_absent_among_outcome_present": condition_absent,
            },
        )

    return "not_testable", "The frozen template is not implemented by the deterministic executor.", {}
