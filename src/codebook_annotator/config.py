"""Run configuration.

Sources, lowest to highest precedence: built-in defaults < config file(s) (JSON or YAML,
later files override earlier ones) < options given explicitly on the command line.
Unknown keys are an error. Relative paths are resolved against the current directory.

"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

ProviderName = Literal["ollama", "mistral_api", "claude_api", "openai_compatible", "stub", "replay"]


class ConfigError(ValueError):
    pass


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    # What to run
    run_id: str = Field(description="Name of the run. Output file: <output_dir>/<run_id>.jsonl")
    codebook: Path = Field(description="Codebook YAML path")
    codebook_id: str | None = Field(None, description="Recorded codebook id (default: codebook 'id' or file stem)")
    codebook_version: str | None = Field(None, description="Recorded codebook version (default: codebook 'version' or '1.0')")
    dataset: Path = Field(description="Input items (.jsonl/.json/.csv/.tsv/.xlsx)")
    dataset_sheet: str | None = Field(None, description="Sheet name for .xlsx input")
    dataset_name: str | None = Field(None, description="Recorded dataset name (default: dataset file stem)")
    dataset_revision: str | None = Field(None, description="Recorded dataset revision/version (free text)")

    # Fields and sampling
    id_field: str = Field("item_id", description="Field holding the item id in the dataset")
    text_field: str = Field("text", description="Field holding the text to annotate")
    split_field: str | None = Field("split", description="Field used by split_value")
    split_value: str | None = Field(None, description="Keep only records with split_field == split_value")
    sample_size: int | None = Field(None, ge=1, description="Number of items (default: all)")
    sampling: Literal["first_n_by_id", "random"] = Field(
        "first_n_by_id", description="How sample_size items are chosen: the first N in id order, or N at random")
    sampling_seed: int = Field(42, description="Random seed for sampling: random")
    sample_id: str | None = Field(None, description="Recorded sample id (default: derived from the sample contents)")

    # Model
    provider: ProviderName = Field(description="Model provider")
    model_tag: str = Field(description="Model name as the provider knows it (e.g. llama3.1:8b)")
    model_digest: str | None = Field(None, description="Recorded model digest. 'auto' reads it from Ollama")
    api_base: str | None = Field(
        None, description="Base URL of the API. Required for openai_compatible, optional for the other API providers")
    api_key_env: str | None = Field(None, description="Environment variable holding the API key")
    ollama_host: str | None = Field(
        None, description="Ollama server address (default: the OLLAMA_HOST variable, else http://localhost:11434)")
    num_ctx: int = Field(65536, description="Context window in tokens. Sent to Ollama only, recorded for every run")
    temperature: float = Field(0.0, description="Sampling temperature")
    seed: int | None = Field(None, description="Generation seed, sent to providers that support one")
    max_tokens: int | None = Field(
        None, description="Maximum response length in tokens (claude_api uses 1024 when not set, ollama ignores it)")
    json_mode: bool = Field(True, description="openai_compatible only: request response_format json_object")
    timeout: float = Field(300.0, description="Timeout per request, in seconds")
    max_retries: int = Field(
        5, ge=1, description="Attempts per item for API providers, with growing waits in between. Ollama makes one attempt")
    replay_from: Path | None = Field(None, description="provider 'replay': earlier run file with raw_response")

    # Prompt and optional features (all off by default)
    prompt_template: str = Field(
        "default_v1", description="Built-in template name, or the path of a template YAML file")
    few_shot: bool = Field(False, description="Show the codebook's examples in the prompt")
    evidence: bool = Field(False, description="Ask for verbatim quotes that support each present category")
    evidence_policy: Literal["report", "enforce"] = Field(
        "report", description="report: only record quote checks. enforce: set a category absent if no quote is verified")
    verify: bool = Field(False, description="Ask the model to check and correct its first answer in a second call")

    # Execution and output
    output_dir: Path = Field(Path("runs"), description="Folder for the run file")
    concurrency: int = Field(1, ge=1, description="Number of items sent to the model at the same time")
    store_text: bool = Field(True, description="Write the item text into each record")
    store_raw_response: bool = Field(True, description="Write the full model answer into each record")
    strict_digest: bool = Field(True, description="Stop if the observed Ollama digest differs from model_digest")
    force_resume: bool = Field(False, description="Resume even if the existing file came from a different config")

    @field_validator("model_digest", mode="before")
    @classmethod
    def _null_digest(cls, v):
        # the strings "null", "none" and "" mean no digest
        if isinstance(v, str) and v.lower() in ("null", "none", ""):
            return None
        return v

    @model_validator(mode="after")
    def _checks(self):
        if self.provider == "replay" and self.replay_from is None:
            raise ValueError("provider 'replay' needs replay_from")
        if self.provider == "openai_compatible" and not self.api_base:
            raise ValueError("provider 'openai_compatible' needs api_base")
        return self

    @property
    def output_file(self) -> Path:
        return self.output_dir / f"{self.run_id}.jsonl"


def read_config_file(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"Config file not found at: {p}")
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))  # YAML is a superset of JSON
    except yaml.YAMLError as e:
        raise ConfigError(f"{p}: cannot parse config: {e}") from e
    if not isinstance(data, dict):
        raise ConfigError(f"{p}: config must be a mapping")
    return data


def build_config(config_files: list[str | Path] = (), overrides: dict | None = None) -> RunConfig:
    merged: dict = {}
    for f in config_files:
        merged.update(read_config_file(f))
    merged.update({k: v for k, v in (overrides or {}).items() if v is not None})
    try:
        return RunConfig.model_validate(merged)
    except ValidationError as e:
        lines = []
        for err in e.errors():
            loc = ".".join(str(x) for x in err["loc"]) or "(config)"
            lines.append(f"  {loc}: {err['msg']}")
        raise ConfigError("Invalid run configuration:\n" + "\n".join(lines)) from None


# Fields that change what the model sees or how outputs are produced. Their hash is stored in
# every record and checked on resume. (Execution-only settings such as concurrency, timeouts,
# output_dir and storage flags are excluded.)
_HASHED = ["codebook_id", "codebook_version", "dataset_name", "dataset_revision", "id_field", "text_field",
           "split_field", "split_value", "sample_size", "sampling", "sampling_seed", "provider", "model_tag",
           "model_digest", "api_base", "num_ctx", "temperature", "seed", "max_tokens", "json_mode",
           "prompt_template", "few_shot", "evidence", "evidence_policy", "verify"]


def config_fingerprint(cfg: RunConfig, *, codebook_sha256: str, system_prompt_sha256: str,
                       sample_sha256: str, model_digest: str | None = None) -> str:
    """Hash of everything that shapes the outputs. `model_digest` is the resolved digest
    (e.g. the one read from Ollama for model_digest 'auto'), so a changed model changes the hash."""
    d = {k: getattr(cfg, k) for k in _HASHED}
    d["model_digest"] = model_digest
    d.update(codebook_sha256=codebook_sha256, system_prompt_sha256=system_prompt_sha256, sample_sha256=sample_sha256)
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode("utf-8")).hexdigest()
