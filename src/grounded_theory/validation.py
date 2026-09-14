"""Deterministic parsing, reference checks, and evidence validation."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, TypeVar

from .models import ComparisonDecision, Evidence, GroundedTheoryState


T = TypeVar("T")


def text(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def optional_text(raw: dict[str, Any], key: str) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string or null")
    return value.strip() or None


def choice(
    raw: dict[str, Any], key: str, values: set[str], default: str | None = None
) -> str:
    value = raw.get(key, default)
    if value not in values:
        raise ValueError(f"{key} must be one of {sorted(values)}")
    return value


def optional_choice(raw: dict[str, Any], key: str, values: set[str]) -> str | None:
    if key not in raw or raw[key] is None:
        return None
    return choice(raw, key, values)


def object_value(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object")
    return value


def object_list(
    raw: dict[str, Any], key: str, allow_empty: bool = False
) -> list[dict[str, Any]]:
    value = raw.get(key, [] if allow_empty else None)
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError(f"{key} must be an array of objects")
    return value


def string_list(raw: dict[str, Any], key: str, allow_empty: bool = False) -> list[str]:
    value = raw.get(key, [] if allow_empty else None)
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(f"{key} must be an array of non-empty strings")
    return [item.strip() for item in value]


def evidence_from_dict(raw: dict[str, Any]) -> Evidence:
    return Evidence(
        record_id=text(raw, "record_id"),
        text_span=text(raw, "text_span"),
        note=optional_text(raw, "note") or "",
    )


def evidence_list(
    raw: dict[str, Any],
    state: GroundedTheoryState,
    *,
    allowed_record_ids: set[str] | None = None,
    allow_empty: bool = False,
) -> list[Evidence]:
    return evidence_list_from_key(
        raw,
        "evidence",
        state,
        allowed_record_ids=allowed_record_ids,
        allow_empty=allow_empty,
    )


def evidence_list_from_key(
    raw: dict[str, Any],
    key: str,
    state: GroundedTheoryState,
    allowed_record_ids: set[str] | None = None,
    allow_empty: bool = False,
) -> list[Evidence]:
    records = {record.id: record for record in state.records}
    result: list[Evidence] = []
    for value in object_list(raw, key, allow_empty=(key != "evidence") or allow_empty):
        evidence = evidence_from_dict(value)
        if allowed_record_ids is not None and evidence.record_id not in allowed_record_ids:
            raise ValueError(
                f"{key} references record outside the requested batch: {evidence.record_id}"
            )
        record = records.get(evidence.record_id)
        if record is None:
            raise ValueError(f"{key} references unknown record: {evidence.record_id}")
        if evidence.text_span not in record.text:
            raise ValueError(
                f"evidence text_span is not a verbatim substring of record {evidence.record_id}"
            )
        result.append(evidence)
    result = dedupe_evidence(result)
    if not result and key == "evidence" and not allow_empty:
        raise ValueError("every concept or relationship update needs at least one evidence span")
    return result


def comparison(raw: dict[str, Any]) -> ComparisonDecision:
    return choice(raw, "comparison", {"SAME", "VARIATION", "NEW", "CONTRADICTION"})  # type: ignore[return-value]


def expected_records(payload: dict[str, Any], action: dict[str, Any]) -> list[str]:
    expected = list(action["record_ids"])
    received = string_list(payload, "processed_record_ids")
    if received != expected:
        raise ValueError(
            f"processed_record_ids must exactly equal the requested batch: {expected}"
        )
    return expected


def by_id(items: list[T], identifier: str, item_name: str) -> T:
    for item in items:
        if getattr(item, "id") == identifier:
            return item
    raise ValueError(f"unknown {item_name} ID: {identifier}")


def existing_ids(items: list[T], identifiers: list[str], item_name: str) -> list[str]:
    for identifier in identifiers:
        by_id(items, identifier, item_name)
    return list(dict.fromkeys(identifiers))


def next_id(items: list[T], prefix: str) -> str:
    existing = {getattr(item, "id") for item in items}
    index = 1
    while f"{prefix}_{index:03d}" in existing:
        index += 1
    return f"{prefix}_{index:03d}"


def dedupe_evidence(items: list[Evidence]) -> list[Evidence]:
    seen: set[tuple[str, str, str]] = set()
    result: list[Evidence] = []
    for item in items:
        key = (item.record_id, item.text_span, item.note)
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def extend_evidence(destination: list[Evidence], additions: list[Evidence]) -> None:
    destination[:] = dedupe_evidence([*destination, *additions])


def extend_unique(destination: list[str], additions: list[str]) -> None:
    for item in additions:
        if item not in destination:
            destination.append(item)


def remaining_record_ids(records: list[Any], done: list[str]) -> list[str]:
    done_set = set(done)
    return [record.id for record in records if record.id not in done_set]


def evidence_as_dict(items: list[Evidence]) -> list[dict[str, Any]]:
    return [asdict(item) for item in items]
