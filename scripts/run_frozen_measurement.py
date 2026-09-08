#!/usr/bin/env python3
"""Apply exactly one already-frozen question to the held-out records.

This worker is intentionally not an agent task.  Its model prompt receives
only one record's text and the frozen measurement question; it never receives
a theory, prediction, expected direction, decision rule, or prior result.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from theory_discovery import TheoryDiscoveryProject


LABEL_PATTERN = re.compile(
    r'"label"\s*:\s*"(PRESENT|ABSENT|UNCERTAIN)"', re.IGNORECASE
)
SYSTEM_PROMPT = """You are a fixed construct-measurement classifier.
Classify only the supplied text against the supplied measurement question.
Output exactly one JSON object: {\"label\":\"PRESENT\"},
{\"label\":\"ABSENT\"}, or {\"label\":\"UNCERTAIN\"}.
Do not explain. Do not use any hypothesis, expected result, theory, or context
outside this question and this text."""


def measurement_concurrency() -> int:
    """Return the bounded number of independent classifier requests in flight.

    SGLang continuously batches simultaneous OpenAI-compatible requests.  We
    intentionally send one record per request: putting multiple texts or
    constructs into a single prompt would weaken the frozen atomic-measurement
    boundary.  This only overlaps independent, identically frozen calls.
    """
    value = os.environ.get("MEASUREMENT_CONCURRENCY", "16")
    try:
        concurrency = int(value)
    except ValueError as error:
        raise SystemExit("MEASUREMENT_CONCURRENCY must be an integer") from error
    if not 1 <= concurrency <= 32:
        raise SystemExit("MEASUREMENT_CONCURRENCY must be between 1 and 32")
    return concurrency


def classify(
    endpoint: str, question: str, text: str, specification: dict[str, object]
) -> tuple[str, str]:
    model_id = specification.get("model_id")
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("frozen measurement specification has no model_id")
    payload = {
        "model": model_id,
        "temperature": specification["temperature"],
        "top_p": specification["top_p"],
        "max_tokens": specification["max_tokens"],
        # The frozen classifier contract requires a direct JSON label. Qwen's
        # default reasoning mode can consume the entire 16-token budget before
        # producing final content, so turn it off explicitly.
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Measurement question:\n{question}\n\nText:\n{text}",
            },
        ],
    }
    request = Request(
        endpoint.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urlopen(request, timeout=180) as response:
                decoded = json.loads(response.read().decode("utf-8"))
            content = decoded["choices"][0]["message"].get("content")
            if not isinstance(content, str):
                # A syntactically valid response without final content is an
                # unclassifiable record, not a transport failure. Preserve the
                # frozen UNCERTAIN=exclude policy and continue the run.
                return "UNCERTAIN", "missing_final_model_content"
            match = LABEL_PATTERN.search(content)
            if match:
                return match.group(1).upper(), "valid_json_label"
            return "UNCERTAIN", "invalid_model_label"
        except (URLError, TimeoutError, ValueError, KeyError, json.JSONDecodeError) as error:
            last_error = error
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
    raise RuntimeError("frozen measurement request failed after 3 attempts") from last_error


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: run_frozen_measurement.py PROJECT_DIR QWEN_ENDPOINT")
    project = TheoryDiscoveryProject.open(sys.argv[1])
    packet = project.task_packet()
    if packet["action"] != "measure_validation_records":
        raise SystemExit("frozen measurement is not currently requested: " + repr(packet["action"]))
    specification = packet["measurement_specification"]
    if not isinstance(specification, dict):
        raise SystemExit("measurement packet has no specification")
    records_path = Path(str(packet["validation_records_path"]))
    with records_path.open("r", encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    text_field = packet.get("text_field")
    if not isinstance(text_field, str):
        raise SystemExit("measurement packet has no frozen text field")
    if not records or text_field not in records[0] or "record_id" not in records[0]:
        raise SystemExit("validation CSV must contain record_id and text_review")
    question = specification.get("measurement_question")
    if not isinstance(question, str):
        raise SystemExit("measurement specification has no question")
    labels: list[dict[str, str] | None] = [None] * len(records)
    concurrency = measurement_concurrency()
    print(
        f"measuring {len(records)} records with {concurrency} independent requests in flight",
        flush=True,
    )

    def measure_record(index: int, record: dict[str, str]) -> tuple[int, dict[str, str]]:
        record_id = record.get("record_id", "")
        text = record.get(text_field, "")
        if not record_id or not text:
            raise SystemExit("every validation record needs record_id and text_review")
        label, parse_status = classify(sys.argv[2], question, text, specification)
        return index, {
            "record_id": record_id,
            "label": label,
            "parse_status": parse_status,
        }

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(measure_record, index, record)
            for index, record in enumerate(records)
        ]
        for completed, future in enumerate(as_completed(futures), start=1):
            index, result = future.result()
            labels[index] = result
            if completed % 25 == 0 or completed == len(records):
                print(f"measured {completed}/{len(records)} records", flush=True)

    if any(label is None for label in labels):
        raise RuntimeError("measurement worker did not produce every label")
    completed_labels = [label for label in labels if label is not None]
    artifact = {
        "measurement_specification_id": specification["id"],
        "specification_hash": specification["specification_hash"],
        "input_records_hash": hashlib.sha256(records_path.read_bytes()).hexdigest(),
        "labels": completed_labels,
    }
    artifact_path = Path(str(packet["measurement_artifact_path"]))
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    project.record_measurement_run(
        str(specification["id"]),
        input_records_hash=str(artifact["input_records_hash"]),
        output_artifact=artifact_path.relative_to(project.root.parent).as_posix(),
        output_hash=hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
        record_count=len(completed_labels),
    )


if __name__ == "__main__":
    main()
