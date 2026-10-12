"""OpenRouter-specific routing and retry helpers."""

from __future__ import annotations

import os
from typing import Any

DEFAULT_PROVIDER_ORDER = ("venice", "deepinfra", "novita", "together")

RETRYABLE_ERROR_MARKERS = (
    "provider returned error",
    "the operation was aborted",
    "bad gateway",
    "gateway timeout",
    "temporarily unavailable",
    "origin web server returned an invalid or incomplete response",
    "api connection error",
    "connection error",
    "server disconnected",
    "remoteprotocolerror",
    "timed out",
    "rate limit",
    "no choices",
    "empty response",
)


def is_openrouter_base_url(base_url: Any) -> bool:
    return "openrouter.ai" in str(base_url or "").lower()


def parse_bool_env(name: str) -> bool | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None


def parse_csv_env(name: str) -> list[str]:
    raw = os.getenv(name, "")
    return [item.strip() for item in raw.split(",") if item.strip()]


def get_openrouter_provider_config(base_url: Any) -> dict[str, Any] | None:
    if not is_openrouter_base_url(base_url):
        return None

    provider: dict[str, Any] = {}

    provider["order"] = list(DEFAULT_PROVIDER_ORDER)
    provider["allow_fallbacks"] = False

    require_parameters = parse_bool_env("OPENROUTER_REQUIRE_PARAMETERS")
    if require_parameters is not None:
        provider["require_parameters"] = require_parameters

    return provider or None


def get_openrouter_extra_body(base_url: Any) -> dict[str, Any] | None:
    provider = get_openrouter_provider_config(base_url)
    if not provider:
        return None
    return {"provider": provider}


def get_client_max_retries(base_url: Any, default: int = 2) -> int:
    env_name = "OPENROUTER_CLIENT_MAX_RETRIES" if is_openrouter_base_url(base_url) else "LLM_CLIENT_MAX_RETRIES"
    raw = os.getenv(env_name)
    if raw:
        try:
            return max(0, int(raw))
        except ValueError:
            pass
    if is_openrouter_base_url(base_url):
        return max(default, 4)
    return default


def is_retryable_provider_error(error: Any) -> bool:
    text = str(error or "").lower()
    return any(marker in text for marker in RETRYABLE_ERROR_MARKERS)


def get_retry_delay(attempt: int, base_delay: float = 2.0, backoff: float = 2.0, max_delay: float = 30.0) -> float:
    delay = base_delay * (backoff ** max(0, attempt - 1))
    return min(delay, max_delay)
