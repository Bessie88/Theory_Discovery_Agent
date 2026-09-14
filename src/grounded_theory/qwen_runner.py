"""Bounded, validator-backed execution of one Grounded Theory model packet."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .persistence import write_json
from .project import GroundedTheoryProject


DEFAULT_OUTPUT_TOKENS = {
    "open_coding": 6144,
    "relational_process_analysis": 6144,
    "validate_relational_grounding": 2048,
    "theoretical_integration": 8192,
}

SYSTEM_PROMPT = """You are a careful Grounded Theory analyst working on one isolated transaction.
The user packet is the complete and only analytic input. Follow its prompt, evidence rules,
and expected output exactly. Work silently, then return exactly one JSON object and no prose.
Never invent an evidence span, record ID, concept ID, relation ID, process ID, or causal link.
When the supplied records do not justify an update, use an empty update list rather than
inventing a finding. Do not claim saturation. This response is checked by a deterministic
validator before it can be committed."""


class CompletionError(RuntimeError):
    """A local model request or its response could not produce a usable object."""


Completion = Callable[[dict[str, Any], str | None], dict[str, Any]]


def build_chat_request(
    packet: dict[str, Any],
    *,
    model: str,
    max_tokens: int,
    repair_feedback: str | None = None,
) -> dict[str, Any]:
    """Build a stateless request; retries never inherit hidden agent context."""
    public_packet = dict(packet)
    public_packet.pop("grounded_theory_results_path", None)
    instruction = (
        "Complete the following Grounded Theory packet. Return only the complete JSON object "
        "that matches expected_output.\n\nPACKET:\n"
        + json.dumps(public_packet, ensure_ascii=False, separators=(",", ":"))
    )
    if repair_feedback is not None:
        instruction += (
            "\n\nA previous candidate was rejected before any state changed. Regenerate the "
            "entire JSON object, correcting the reported issue.\nVALIDATOR FEEDBACK:\n"
            + repair_feedback
        )
    return {
        "model": model,
        "temperature": 0,
        "top_p": 1,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
        ],
    }


def parse_completion(response: dict[str, Any]) -> dict[str, Any]:
    """Extract the one JSON object from an OpenAI-compatible chat response."""
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise CompletionError("local completion response has no choices")
    choice = choices[0]
    if choice.get("finish_reason") == "length":
        raise CompletionError("local completion reached its bounded output-token limit")
    message = choice.get("message")
    if not isinstance(message, dict):
        raise CompletionError("local completion response has no message")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise CompletionError("local completion response has no JSON content")
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as error:
        raise CompletionError(f"local completion returned invalid JSON: {error.msg}") from error
    if not isinstance(payload, dict):
        raise CompletionError("local completion result must be one JSON object")
    return payload


class LocalQwenClient:
    """A small OpenAI-compatible client for the runner-owned local SGLang server."""

    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        timeout_seconds: int,
        output_tokens: dict[str, int] | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.output_tokens = output_tokens or DEFAULT_OUTPUT_TOKENS

    def complete(self, packet: dict[str, Any], repair_feedback: str | None = None) -> dict[str, Any]:
        action = packet.get("action")
        max_tokens = self.output_tokens.get(action) if isinstance(action, str) else None
        if max_tokens is None:
            raise CompletionError(f"no output-token budget for action: {action!r}")
        request_body = build_chat_request(
            packet,
            model=self.model,
            max_tokens=max_tokens,
            repair_feedback=repair_feedback,
        )
        try:
            response = self._post(request_body)
        except CompletionError as error:
            # Older SGLang builds can lack JSON mode. The deterministic validator
            # remains the authoritative contract in that compatibility fallback.
            if "response_format" not in str(error):
                raise
            request_body.pop("response_format", None)
            response = self._post(request_body)
        return parse_completion(response)

    def _post(self, request_body: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(request_body, ensure_ascii=False).encode("utf-8")
        request = Request(
            self.endpoint,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise CompletionError(f"local completion HTTP {error.code}: {detail}") from error
        except URLError as error:
            raise CompletionError(f"local completion connection error: {error.reason}") from error
        except TimeoutError as error:
            raise CompletionError("local completion request timed out") from error
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as error:
            raise CompletionError("local completion server returned non-JSON data") from error
        if not isinstance(value, dict):
            raise CompletionError("local completion server returned a non-object response")
        return value


def execute_stage(
    project: GroundedTheoryProject,
    complete: Completion,
    *,
    packet_path: str | Path,
    result_path: str | Path,
    attempts: int,
) -> dict[str, Any]:
    """Complete one stage without ever treating a failed candidate as analysed."""
    if attempts < 1:
        raise ValueError("attempts must be positive")
    packet = project.task_packet()
    write_json(Path(packet_path), packet)
    feedback: str | None = None
    failures: list[str] = []
    destination = Path(result_path)

    for attempt in range(1, attempts + 1):
        try:
            candidate = complete(packet, feedback)
            write_json(_attempt_path(destination, attempt), candidate)
            accepted = project.submit_agent_result(candidate)
        except (CompletionError, ValueError) as error:
            feedback = str(error)
            failures.append(f"attempt {attempt}: {feedback}")
            if attempt < attempts:
                time.sleep(min(attempt, 3))
            continue
        write_json(destination, candidate)
        return {
            "attempt": attempt,
            "accepted": accepted,
            "action": packet["action"],
            "failures_before_success": failures,
        }
    raise CompletionError("; ".join(failures))


def _attempt_path(destination: Path, attempt: int) -> Path:
    return destination.with_name(
        f"{destination.stem}.candidate-attempt{attempt}{destination.suffix}"
    )
