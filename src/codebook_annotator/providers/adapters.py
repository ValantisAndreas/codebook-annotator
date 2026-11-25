"""Concrete providers.

Each provider builds one request per call and returns a ProviderResponse.
"""

from __future__ import annotations

import json
import os
import sys
import time

from .base import Provider, ProviderResponse, join_url, redact_urls

DEFAULT_OLLAMA_HOST = "http://localhost:11434"


class OllamaProvider(Provider):
    """Local Ollama, /api/generate with format=json. Single attempt, no retries."""

    name = "ollama"

    def __init__(self, *, ollama_host: str | None = None, **kw):
        super().__init__(**kw)
        host = ollama_host or self.api_base or os.environ.get("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST
        if not host.startswith("http"):
            host = "http://" + host
        self.host = host.rstrip("/")

    def url(self) -> str:
        return f"{self.host}/api/generate"

    def build_payload(self, system_prompt, user_prompt):
        options = {"temperature": self.temperature, "num_ctx": self.num_ctx}
        if self.seed is not None:
            options["seed"] = self.seed
        return {
            "model": self.model_tag,
            "prompt": user_prompt,
            "system": system_prompt,
            "stream": False,
            "format": "json",
            "options": options,
        }

    def query(self, system_prompt, user_prompt, *, item_id=None) -> ProviderResponse:
        payload = self.build_payload(system_prompt, user_prompt)
        start = time.perf_counter()
        try:
            response = self.client.post(self.url(), json=payload)
            response.raise_for_status()
            data = response.json()
            return ProviderResponse(
                data.get("response", "").strip(),
                None,
                input_tokens=data.get("prompt_eval_count"),
                output_tokens=data.get("eval_count"),
                latency_ms=(time.perf_counter() - start) * 1000,
            )
        except Exception as e:
            message = redact_urls(str(e))
            print(f"  Ollama request failed: {message}", file=sys.stderr)
            return ProviderResponse("", message, latency_ms=(time.perf_counter() - start) * 1000)

    def observed_model_digest(self) -> str | None:
        """Full sha256 digest of the local model, from /api/tags (None if unavailable)."""
        try:
            r = self.client.get(f"{self.host}/api/tags")
            r.raise_for_status()
            names = {self.model_tag, self.model_tag + ":latest"}
            for m in r.json().get("models", []):
                if m.get("name") in names or m.get("model") in names:
                    d = m.get("digest")
                    return f"sha256:{d}" if d and not d.startswith("sha256:") else d
        except Exception as e:
            print(f"  Could not read model digest from Ollama: {e}", file=sys.stderr)
        return None

    def available_models(self) -> list[str]:
        r = self.client.get(f"{self.host}/api/tags")
        r.raise_for_status()
        return [m.get("name") for m in r.json().get("models", [])]


class MistralProvider(Provider):
    name = "mistral_api"
    needs_api_key = True
    api_key_env = "MISTRAL_API_KEY"

    def url(self):
        return join_url(self.api_base or "https://api.mistral.ai/v1", "/chat/completions")

    def headers(self):
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def build_payload(self, system_prompt, user_prompt):
        payload = {
            "model": self.model_tag,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
        }
        if self.seed is not None:
            payload["random_seed"] = self.seed
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        return payload

    def extract(self, data):
        usage = data.get("usage") or {}
        return ProviderResponse(data["choices"][0]["message"]["content"].strip(), None,
                                usage.get("prompt_tokens"), usage.get("completion_tokens"))


class AnthropicProvider(Provider):
    name = "claude_api"
    needs_api_key = True
    api_key_env = "ANTHROPIC_API_KEY"
    DEFAULT_MAX_TOKENS = 1024

    def url(self):
        return join_url(self.api_base or "https://api.anthropic.com/v1", "/messages")

    def headers(self):
        return {"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}

    def build_payload(self, system_prompt, user_prompt):
        return {
            "model": self.model_tag,
            "max_tokens": self.max_tokens or self.DEFAULT_MAX_TOKENS,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
            "temperature": self.temperature,
        }

    def extract(self, data):
        usage = data.get("usage") or {}
        return ProviderResponse(data["content"][0]["text"].strip(), None,
                                usage.get("input_tokens"), usage.get("output_tokens"))


class OpenAICompatibleProvider(Provider):
    """Any server implementing the OpenAI /chat/completions API (OpenAI, vLLM, LM Studio,
    llama.cpp server, OpenRouter, DeepSeek, Qwen/DashScope, ...). Set `api_base`."""

    name = "openai_compatible"
    needs_api_key = False  # local servers usually need none
    api_key_env = "OPENAI_API_KEY"

    def url(self):
        if not self.api_base:
            raise ValueError("provider openai_compatible requires api_base (e.g. http://localhost:8000/v1)")
        return join_url(self.api_base, "/chat/completions")

    def headers(self):
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def build_payload(self, system_prompt, user_prompt):
        payload = {
            "model": self.model_tag,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
        }
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        if self.seed is not None:
            payload["seed"] = self.seed
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        return payload

    def extract(self, data):
        usage = data.get("usage") or {}
        content = data["choices"][0]["message"].get("content") or ""
        return ProviderResponse(content.strip(), None, usage.get("prompt_tokens"), usage.get("completion_tokens"))


class StubProvider(Provider):
    """Offline provider: answers every item with all categories absent. For dry runs and tests."""

    name = "stub"

    def __init__(self, *, categories: list[str], **kw):
        super().__init__(**kw)
        self.categories = list(categories)

    def query(self, system_prompt, user_prompt, *, item_id=None):
        return ProviderResponse(json.dumps({
            "predictions": {c: False for c in self.categories},
            "reasoning": "STUB: offline stub provider.",
        }), None, 0, 0, 0.0)


class ReplayProvider(Provider):
    """Offline provider: returns stored raw responses from an earlier run file, by item_id.

    Useful to re-parse or re-evaluate a run under a new tool version without calling a model.
    """

    name = "replay"

    def __init__(self, *, replay_from: str, **kw):
        super().__init__(**kw)
        self.responses: dict[str, str] = {}
        with open(replay_from, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    if "raw_response" in rec:
                        self.responses[rec["item_id"]] = rec["raw_response"]

    def query(self, system_prompt, user_prompt, *, item_id=None):
        if item_id not in self.responses:
            return ProviderResponse("", f"replay: no stored response for item {item_id}")
        return ProviderResponse(self.responses[item_id], None)


PROVIDERS = {
    "ollama": OllamaProvider,
    "mistral_api": MistralProvider,
    "claude_api": AnthropicProvider,
    "openai_compatible": OpenAICompatibleProvider,
    "stub": StubProvider,
    "replay": ReplayProvider,
}
