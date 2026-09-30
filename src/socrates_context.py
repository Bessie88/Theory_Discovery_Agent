"""Lossless shared-case packet views for stateless SoCRATES model calls."""

from __future__ import annotations

import hashlib
import re
from copy import deepcopy
from typing import Any


INCIDENT_MARKER = "INTERACTION / REASONING INCIDENT:"
DYNAMIC_CONTEXT_MARKER = "LOCAL PRECEDING PUBLIC TURN:"
CASE_ID_RE = re.compile(r"^CASE\s+(?P<case_id>[^\s|]+)", re.MULTILINE)


def shared_case_context_records(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]], dict[str, int]]:
    """Move repeated static case text into one referenced context per packet.

    Project state still retains every original full record. This is only the
    model-facing rendering: each incident retains its dynamic trace, utterance,
    and feedback and references a complete static context in the same packet.
    An unparseable record is passed through unchanged.
    """
    contexts: dict[str, dict[str, str]] = {}
    compacted: list[dict[str, Any]] = []
    shared_records = 0
    raw_chars = 0
    rendered_chars = 0
    for record in records:
        item = deepcopy(record)
        text = item.get("text")
        if not isinstance(text, str) or INCIDENT_MARKER not in text:
            compacted.append(item)
            if isinstance(text, str):
                raw_chars += len(text)
                rendered_chars += len(text)
            continue
        before_incident, incident = text.split(INCIDENT_MARKER, 1)
        # In the source view the fixed case description is followed by the
        # immediately preceding public turn. That turn changes for every
        # incident and must stay attached to its record rather than becoming a
        # distinct faux "case context" or being dropped during compression.
        if DYNAMIC_CONTEXT_MARKER in before_incident:
            static, preceding_turn = before_incident.split(DYNAMIC_CONTEXT_MARKER, 1)
            dynamic_prefix = f"{DYNAMIC_CONTEXT_MARKER}\n{preceding_turn.strip()}\n"
        else:
            static = before_incident
            dynamic_prefix = ""
        match = CASE_ID_RE.search(static)
        if not match or not static.strip() or not incident.strip():
            compacted.append(item)
            raw_chars += len(text)
            rendered_chars += len(text)
            continue
        case_id = match.group("case_id")
        static = static.strip()
        # Preserve every static variant if a source uses different rendering
        # for the same case identifier.
        context_id = f"{case_id}:{hashlib.sha256(static.encode('utf-8')).hexdigest()[:12]}"
        contexts.setdefault(
            context_id,
            {"context_id": context_id, "case_id": case_id, "static_context": static},
        )
        item["case_context_id"] = context_id
        item["text"] = f"{dynamic_prefix}{INCIDENT_MARKER}\n{incident.strip()}"
        compacted.append(item)
        shared_records += 1
        raw_chars += len(text)
        rendered_chars += len(item["text"])
    rendered_chars += sum(len(item["static_context"]) for item in contexts.values())
    return compacted, list(contexts.values()), {
        "records_with_shared_context": shared_records,
        "shared_case_contexts": len(contexts),
        "raw_record_characters": raw_chars,
        "rendered_record_characters": rendered_chars,
        "saved_record_characters": raw_chars - rendered_chars,
    }
