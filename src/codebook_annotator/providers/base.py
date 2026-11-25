"""Provider interface.

A provider sends one (system prompt, user prompt) pair to a model and returns the
raw text plus usage metadata. Errors are returned, not raised, so a failed item
is recorded and the run continues.
"""

from __future__ import annotations

import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass

import httpx

_USERINFO = re.compile(r"(https?://)[^/\s'\"@]+@")
_QUERY = re.compile(r"(https?://[^\s'\"?#]*)[?#][^\s'\"]*")


def redact_urls(text: str) -> str:
    """Remove credentials (user:password@) and query strings from URLs in a message."""
    return _QUERY.sub(r"\1", _USERINFO.sub(r"\1", text))


def join_url(base: str, path: str) -> str:
    """Append a path to a base URL, keeping any query string at the end."""
    url = httpx.URL(base)
    return str(url.copy_with(path=url.path.rstrip("/") + path))


@dataclass
class ProviderResponse:
    text: str
    error: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: float | None = None
    attempts: int = 1

    @property
    def usage(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_ms": None if self.latency_ms is None else round(self.latency_ms, 1),
            "attempts": self.attempts,
        }


class Provider:
    """Base class. Subclasses implement `_request` (one HTTP attempt) or override `query`."""

    name = "base"
    needs_api_key = False

    def __init__(self, *, model_tag: str, temperature: float = 0.0, seed: int | None = None,
                 num_ctx: int | None = None, max_tokens: int | None = None, timeout: float = 300.0,
                 max_retries: int = 5, backoff: float = 2.0, api_base: str | None = None,
                 api_key: str | None = None, transport: httpx.BaseTransport | None = None,
                 sleep: Callable[[float], None] = time.sleep, json_mode: bool = True):
        self.model_tag = model_tag
        self.temperature = temperature
        self.seed = seed
        self.num_ctx = num_ctx
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff
        self.api_base = api_base
        self.api_key = api_key
        self.json_mode = json_mode
        self._sleep = sleep
        self.client = httpx.Client(timeout=timeout, transport=transport)

    # Overridable hooks -------------------------------------------------------------------------
    def build_payload(self, system_prompt: str, user_prompt: str) -> dict:
        raise NotImplementedError

    def url(self) -> str:
        raise NotImplementedError

    def headers(self) -> dict:
        return {}

    def extract(self, data: dict) -> ProviderResponse:
        raise NotImplementedError

    def observed_model_digest(self) -> str | None:
        """Content digest of the model actually served, if the provider exposes it."""
        return None

    def close(self) -> None:
        self.client.close()

    # Retry loop shared by API providers: max_retries attempts with backoff*2^n waits.
    # 429 and other errors wait and retry, and the last error is returned.
    def query(self, system_prompt: str, user_prompt: str, *, item_id: str | None = None) -> ProviderResponse:
        payload = self.build_payload(system_prompt, user_prompt)
        start = time.perf_counter()
        for attempt in range(self.max_retries):
            try:
                response = self.client.post(self.url(), json=payload, headers=self.headers())
                if response.status_code == 429:
                    wait_time = self.backoff * (2 ** attempt)
                    print(f"  {self.name} rate limit (429). Retrying in {wait_time:.1f}s... "
                          f"(attempt {attempt + 1}/{self.max_retries})", file=sys.stderr)
                    self._sleep(wait_time)
                    continue
                response.raise_for_status()
                out = self.extract(response.json())
                out.latency_ms = (time.perf_counter() - start) * 1000
                out.attempts = attempt + 1
                return out
            except Exception as e:
                if attempt == self.max_retries - 1:
                    message = redact_urls(str(e))
                    print(f"  {self.name} request failed after {self.max_retries} attempts: {message}",
                          file=sys.stderr)
                    return ProviderResponse("", message, latency_ms=(time.perf_counter() - start) * 1000,
                                            attempts=attempt + 1)
                self._sleep(self.backoff * (2 ** attempt))
        return ProviderResponse("", "Max retries exceeded", latency_ms=(time.perf_counter() - start) * 1000,
                                attempts=self.max_retries)
