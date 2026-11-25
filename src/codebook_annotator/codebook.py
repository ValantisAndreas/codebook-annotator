"""Codebook loading and validation.

A codebook is a YAML file. Only `categories` is required, everything else is optional.

    schema_version: 1            # optional
    id: my_codebook              # optional (default: file stem)
    version: "1.0"               # optional
    prompt:                      # optional: domain wording for the prompt template
      role: ...                  #   any of: role, task, rules_heading, reasoning_example
    abstention_instruction: ...  # optional: inserted after the task line
    render:                      # optional: which category fields to print, in order
      - {field: definition, label: Definition}
    categories:                  # required: key -> {field: value, ...}
      Care: {name: Care, definition: ...}
    output:                      # optional (default {type: binary})
      type: ordinal
      scale: {min: 1, max: 10}
      scale_description: ...     # optional, printed in the output instructions
    examples:                    # optional few-shot examples (used only when few_shot is on)
      - text: ...
        labels: [Care]           # binary: categories present
        scores: {Care: 7}        # ordinal: category -> score (omitted/null = absent)
        reasoning: ...           # optional
        evidence: {Care: ["..."]}  # optional
    provenance: {...}            # optional, free-form
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

KNOWN_TOP_LEVEL_KEYS = {
    "schema_version", "id", "version", "prompt", "abstention_instruction", "render",
    "categories", "output", "examples", "provenance", "codebook",
}
PROMPT_KEYS = {"role", "task", "rules_heading", "reasoning_example"}

# Field layout for codebooks without a `render` block.
DEFAULT_RENDER = [
    {"field": "name", "label": "Name"},
    {"field": "definition", "label": "Definition"},
    {"field": "example_item", "label": "Example Item"},
    {"field": "example_words", "label": "Illustrative Words"},
]


class CodebookError(ValueError):
    """Raised when a codebook cannot be used."""


@dataclass
class Issue:
    level: str  # "error" | "warning"
    message: str

    def __str__(self) -> str:
        return f"[{self.level.upper()}] {self.message}"


@dataclass
class Codebook:
    data: dict[str, Any]
    path: Path | None = None
    sha256: str | None = None
    issues: list[Issue] = field(default_factory=list)

    @property
    def id(self) -> str | None:
        if self.data.get("id"):
            return str(self.data["id"])
        return self.path.stem if self.path else None

    @property
    def version(self) -> str | None:
        v = self.data.get("version")
        return None if v is None else str(v)

    @property
    def categories(self) -> dict[str, dict]:
        return self.data.get("categories") or {}

    @property
    def category_keys(self) -> list[str]:
        """Category keys in prompt and output order (alphabetical)."""
        return sorted(self.categories.keys())

    @property
    def render_spec(self) -> list[dict]:
        return self.data.get("render", DEFAULT_RENDER)

    @property
    def prompt_overrides(self) -> dict[str, str]:
        return dict(self.data.get("prompt") or {})

    @property
    def _output(self) -> dict:
        out = self.data.get("output")
        return out if isinstance(out, dict) else {}

    @property
    def output_type(self) -> str:
        return self._output.get("type", "binary")

    @property
    def scale(self) -> tuple[int, int]:
        s = self._output.get("scale")
        s = s if isinstance(s, dict) else {}
        try:
            return int(s.get("min", 1)), int(s.get("max", 10))
        except (TypeError, ValueError):
            return 1, 10

    @property
    def scale_description(self) -> str | None:
        return self._output.get("scale_description")

    @property
    def examples(self) -> list[dict]:
        exs = self.data.get("examples")
        return [e for e in exs if isinstance(e, dict)] if isinstance(exs, list) else []

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "warning"]


def load_codebook(path: str | Path, *, strict: bool = True) -> Codebook:
    """Load and validate a codebook. With strict=True, validation errors raise CodebookError."""
    path = Path(path)
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        raise CodebookError(f"{path}: not UTF-8 text ({e.reason} at byte {e.start}). Save the file as UTF-8") from None
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise CodebookError(f"{path}: top level must be a mapping, got {type(data).__name__}")
    cb = Codebook(data=data, path=path, sha256=hashlib.sha256(raw).hexdigest())
    cb.issues = validate_codebook(cb)
    if strict and cb.errors:
        raise CodebookError(f"{path}: invalid codebook:\n" + "\n".join(f"  {e}" for e in cb.errors))
    return cb


def codebook_from_dict(data: dict, *, strict: bool = True) -> Codebook:
    cb = Codebook(data=data)
    cb.issues = validate_codebook(cb)
    if strict and cb.errors:
        raise CodebookError("invalid codebook:\n" + "\n".join(f"  {e}" for e in cb.errors))
    return cb


def validate_codebook(cb: Codebook) -> list[Issue]:
    d = cb.data
    issues: list[Issue] = []

    def err(msg: str) -> None:
        issues.append(Issue("error", msg))

    def warn(msg: str) -> None:
        issues.append(Issue("warning", msg))

    for k in d:
        if k not in KNOWN_TOP_LEVEL_KEYS:
            warn(f"unknown top-level key '{k}' (ignored)")

    cats = d.get("categories")
    if not isinstance(cats, dict) or not cats:
        err("'categories' must be a non-empty mapping of category key -> fields")
        return issues
    for key, cat in cats.items():
        if not isinstance(key, str):
            err(f"category key {key!r} must be a string")
        if not isinstance(cat, dict):
            err(f"category '{key}' must be a mapping of field -> value")

    render = d.get("render", DEFAULT_RENDER)
    if not isinstance(render, list) or not render:
        err("'render' must be a non-empty list of {field, label} entries")
    else:
        for i, entry in enumerate(render):
            if not isinstance(entry, dict) or "field" not in entry or "label" not in entry:
                err(f"render[{i}] must have 'field' and 'label'")
                continue
            f = entry["field"]
            present = [k for k, c in cats.items() if isinstance(c, dict) and c.get(f) not in (None, "")]
            if not present:
                warn(f"render field '{f}' is empty or missing in every category (it will never be printed)")
            elif len(present) < len(cats):
                missing = sorted(set(cats) - set(present))
                warn(f"render field '{f}' is empty or missing in: {', '.join(missing)} (skipped for those)")

    prompt = d.get("prompt")
    if prompt is not None:
        if not isinstance(prompt, dict):
            err("'prompt' must be a mapping")
        else:
            for k, v in prompt.items():
                if k not in PROMPT_KEYS:
                    err(f"unknown prompt key '{k}' (allowed: {', '.join(sorted(PROMPT_KEYS))})")
                elif not isinstance(v, str):
                    err(f"prompt.{k} must be a string")

    ai = d.get("abstention_instruction")
    if ai is not None and not isinstance(ai, str):
        err("'abstention_instruction' must be a string")

    out = d.get("output")
    if out is not None:
        if not isinstance(out, dict):
            err("'output' must be a mapping")
        else:
            t = out.get("type", "binary")
            if t not in ("binary", "ordinal"):
                err(f"output.type must be 'binary' or 'ordinal', got {t!r}")
            if t == "ordinal":
                s = out.get("scale")
                if s is not None and not isinstance(s, dict):
                    err("output.scale must be a mapping with 'min' and 'max'")
                else:
                    s = s or {}
                    lo, hi = s.get("min", 1), s.get("max", 10)
                    if not all(isinstance(v, int) and not isinstance(v, bool) for v in (lo, hi)):
                        err("output.scale.min and output.scale.max must be whole numbers")
                    elif lo >= hi:
                        err("output.scale.min must be smaller than output.scale.max")

    exs = d.get("examples")
    if exs is not None:
        if not isinstance(exs, list):
            err("'examples' must be a list")
        else:
            for i, ex in enumerate(exs):
                if not isinstance(ex, dict) or not isinstance(ex.get("text"), str):
                    err(f"examples[{i}] must be a mapping with a 'text' string")
                    continue
                labels, scores, ev = ex.get("labels") or [], ex.get("scores") or {}, ex.get("evidence") or {}
                bad_types = [n for n, v, t in (("labels", labels, list), ("scores", scores, dict),
                                               ("evidence", ev, dict)) if not isinstance(v, t)]
                for n in bad_types:
                    err(f"examples[{i}].{n} must be a {'list' if n == 'labels' else 'mapping'}")
                if bad_types:
                    continue
                for c in list(labels) + list(scores) + list(ev):
                    if c not in cats:
                        err(f"examples[{i}] refers to unknown category '{c}'")
                for c, v in scores.items():
                    if v is not None and (isinstance(v, bool) or not isinstance(v, int)):
                        err(f"examples[{i}].scores.{c} must be a whole number or null, got {v!r}")
    return issues
