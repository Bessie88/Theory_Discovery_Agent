"""Local bridge from Prime's OpenAI requests to SGLang thinking settings.

Prime 0.9.1 has reasoning levels but no numeric thinking-budget request field.
The model alias encodes a budget chosen by the runner; this loopback-only bridge
restores the actual served-model name and places a positive budget in SGLang's
``custom_params.thinking_budget`` transport.  SGLang's OpenAI *chat* endpoint
does not expose ``max_thinking_tokens`` even though its lower-level generation
input does.  A separate disabled alias selects Qwen's non-thinking template.
"""

from __future__ import annotations

import argparse
import json
import logging
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


MODEL_ALIAS_PREFIX = "__gt_thinking_budget_"
DISABLED_MODEL_ALIAS_PREFIX = "__gt_thinking_disabled__"
MAX_REQUEST_BYTES = 32 * 1024 * 1024


class AdapterRequestError(ValueError):
    """The local caller sent a request the bridge must not reinterpret."""


def thinking_model_alias(model: str, thinking_budget: int | None) -> str:
    """Encode either disabled thinking or a strict positive cap."""
    if not isinstance(model, str) or not model:
        raise ValueError("served model name must be a non-empty string")
    if thinking_budget is None:
        return f"{DISABLED_MODEL_ALIAS_PREFIX}{model}"
    if isinstance(thinking_budget, bool) or not isinstance(thinking_budget, int):
        raise ValueError("thinking budget must be an integer")
    if thinking_budget <= 0:
        raise ValueError("thinking budget must be a positive integer or None to disable thinking")
    return f"{MODEL_ALIAS_PREFIX}{thinking_budget}__{model}"


def decode_thinking_model_alias(model: Any) -> tuple[str, int | None]:
    """Return the served-model name and explicit thinking mode from its alias."""
    if not isinstance(model, str):
        raise AdapterRequestError("request model is not a Grounded Theory thinking alias")
    if model.startswith(DISABLED_MODEL_ALIAS_PREFIX):
        served_model = model[len(DISABLED_MODEL_ALIAS_PREFIX) :]
        if not served_model:
            raise AdapterRequestError("thinking-disabled model alias does not contain a served model")
        return served_model, None
    if not model.startswith(MODEL_ALIAS_PREFIX):
        raise AdapterRequestError("request model is not a Grounded Theory thinking alias")
    raw_budget, marker, served_model = model[len(MODEL_ALIAS_PREFIX) :].partition("__")
    if not marker or not served_model:
        raise AdapterRequestError("thinking model alias does not contain a served model")
    try:
        thinking_budget = int(raw_budget)
    except ValueError as error:
        raise AdapterRequestError("thinking model alias has an invalid budget") from error
    if thinking_budget <= 0:
        raise AdapterRequestError("thinking model alias must carry a positive budget")
    return served_model, thinking_budget


def rewrite_chat_completion_payload(payload: Any) -> dict[str, Any]:
    """Inject the alias-selected thinking mode without trusting the client."""
    if not isinstance(payload, dict):
        raise AdapterRequestError("request body must be a JSON object")
    body = dict(payload)
    served_model, thinking_budget = decode_thinking_model_alias(body.get("model"))

    requested_budget = body.get("max_thinking_tokens")
    if requested_budget is not None and requested_budget != thinking_budget:
        raise AdapterRequestError("client max_thinking_tokens conflicts with the selected stage budget")

    custom_params = body.get("custom_params")
    if custom_params is not None and not isinstance(custom_params, dict):
        raise AdapterRequestError("custom_params must be an object when supplied")
    # The alias is authoritative.  Do not let the caller select a cap via
    # SGLang's internal transport, but retain unrelated custom parameters.
    if isinstance(custom_params, dict) and "thinking_budget" in custom_params:
        if custom_params["thinking_budget"] != thinking_budget:
            raise AdapterRequestError("client thinking_budget conflicts with the selected stage budget")
        custom_params = dict(custom_params)
        del custom_params["thinking_budget"]
    elif custom_params is None:
        custom_params = {}
    else:
        custom_params = dict(custom_params)

    template_kwargs = body.get("chat_template_kwargs")
    if template_kwargs is None:
        template_kwargs = {}
    if not isinstance(template_kwargs, dict):
        raise AdapterRequestError("chat_template_kwargs must be an object when supplied")
    template_kwargs = dict(template_kwargs)
    # The disabled alias always uses Qwen's non-thinking template. Positive
    # caps use strict SGLang reasoning with Qwen's medium template effort; the
    # grammar cap, rather than an xhigh prompt instruction, controls its length.
    template_kwargs["enable_thinking"] = thinking_budget is not None
    if thinking_budget is not None:
        template_kwargs["reasoning_effort"] = "medium"
        custom_params["thinking_budget"] = thinking_budget
    else:
        template_kwargs.pop("reasoning_effort", None)

    body["model"] = served_model
    # ``max_thinking_tokens`` is not a supported ChatCompletionRequest field.
    # Remove it to ensure the transport above is the sole effective cap.
    body.pop("max_thinking_tokens", None)
    if custom_params:
        body["custom_params"] = custom_params
    else:
        body.pop("custom_params", None)
    body["chat_template_kwargs"] = template_kwargs
    return body


class ThinkingAdapterServer(ThreadingHTTPServer):
    """Threaded loopback server holding the immutable upstream origin."""

    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], upstream: str):
        super().__init__(address, ThinkingAdapterHandler)
        self.upstream = upstream.rstrip("/")


class ThinkingAdapterHandler(BaseHTTPRequestHandler):
    """Forward one OpenAI chat request after adding the strict budget."""

    server: ThinkingAdapterServer

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path != "/health":
            self._json_error(HTTPStatus.NOT_FOUND, "only /health is available by GET")
            return
        self._json_response(HTTPStatus.OK, {"status": "ok"})

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path != "/v1/chat/completions":
            self._json_error(HTTPStatus.NOT_FOUND, "only /v1/chat/completions is available")
            return
        try:
            raw = self._read_json_body()
            body = rewrite_chat_completion_payload(raw)
            encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
            upstream_request = Request(
                self.server.upstream + self.path,
                data=encoded,
                headers={"Content-Type": "application/json", "Accept": self.headers.get("Accept", "application/json")},
                method="POST",
            )
            with urlopen(upstream_request, timeout=900) as response:
                self._forward_response(response.status, response.headers.get_content_type(), response.read())
        except AdapterRequestError as error:
            self._json_error(HTTPStatus.BAD_REQUEST, str(error))
        except json.JSONDecodeError:
            self._json_error(HTTPStatus.BAD_REQUEST, "request body is not valid JSON")
        except HTTPError as error:
            self._forward_response(
                error.code,
                error.headers.get_content_type() if error.headers else "application/json",
                error.read(),
            )
        except URLError as error:
            self._json_error(HTTPStatus.BAD_GATEWAY, f"SGLang upstream is unavailable: {error.reason}")
        except TimeoutError:
            self._json_error(HTTPStatus.GATEWAY_TIMEOUT, "SGLang upstream timed out")

    def _read_json_body(self) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise AdapterRequestError("Content-Length is required")
        try:
            length = int(raw_length)
        except ValueError as error:
            raise AdapterRequestError("Content-Length must be an integer") from error
        if length < 1 or length > MAX_REQUEST_BYTES:
            raise AdapterRequestError(f"request body must be between 1 and {MAX_REQUEST_BYTES} bytes")
        value = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(value, dict):
            raise AdapterRequestError("request body must be a JSON object")
        return value

    def _forward_response(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json_response(self, status: HTTPStatus, value: dict[str, Any]) -> None:
        self._forward_response(status, "application/json", json.dumps(value).encode("utf-8"))

    def _json_error(self, status: HTTPStatus, message: str) -> None:
        self._json_response(status, {"error": {"message": message, "type": "invalid_request_error"}})

    def log_message(self, format: str, *args: Any) -> None:
        logging.getLogger(__name__).info("%s - %s", self.client_address[0], format % args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Loopback adapter for strict SGLang thinking budgets")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=30039, type=int)
    parser.add_argument("--upstream", required=True, help="SGLang origin, without /v1")
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "::1", "localhost"}:
        raise SystemExit("thinking adapter must bind only to loopback")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    server = ThinkingAdapterServer((args.host, args.port), args.upstream)
    logging.info("strict-thinking adapter listening on http://%s:%d/v1", args.host, args.port)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
