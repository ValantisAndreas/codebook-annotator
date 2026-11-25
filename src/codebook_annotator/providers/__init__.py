"""Model providers. Use `make_provider(cfg, categories)`."""

from __future__ import annotations

import os

import httpx

from .adapters import PROVIDERS
from .base import Provider, ProviderResponse

__all__ = ["PROVIDERS", "Provider", "ProviderResponse", "make_provider"]


def _is_openai_host(api_base: str | None) -> bool:
    try:
        return httpx.URL(api_base or "").host == "api.openai.com"
    except Exception:
        return False


def make_provider(cfg, categories: list[str]) -> Provider:
    """Build the provider named in a RunConfig. Raises ValueError if an API key is missing."""
    cls = PROVIDERS[cfg.provider]
    env = cfg.api_key_env or getattr(cls, "api_key_env", None)
    if cfg.provider == "openai_compatible" and not cfg.api_key_env and not _is_openai_host(cfg.api_base):
        env = None  # never send OPENAI_API_KEY to another server unless api_key_env asks for it
    api_key = os.environ.get(env) if env else None
    if cls.needs_api_key and not api_key:
        raise ValueError(f"Error: {env} environment variable is not set. {cfg.provider} runs require a valid key.")
    kw = dict(model_tag=cfg.model_tag, temperature=cfg.temperature, seed=cfg.seed, num_ctx=cfg.num_ctx,
              max_tokens=cfg.max_tokens, timeout=cfg.timeout, max_retries=cfg.max_retries,
              api_base=cfg.api_base, api_key=api_key, json_mode=cfg.json_mode)
    if cfg.provider == "ollama":
        kw["ollama_host"] = cfg.ollama_host
    elif cfg.provider == "stub":
        kw["categories"] = categories
    elif cfg.provider == "replay":
        kw["replay_from"] = str(cfg.replay_from)
    return cls(**kw)
