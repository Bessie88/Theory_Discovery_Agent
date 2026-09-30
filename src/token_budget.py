"""Shared conservative token estimates for JSON packets."""

from __future__ import annotations

import json
from typing import Any


def packet_token_estimate(packet: dict[str, Any]) -> int:
    """Conservative deterministic estimate for UTF-8 JSON packet size."""
    encoded = json.dumps(packet, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return (len(encoded) + 2) // 3
