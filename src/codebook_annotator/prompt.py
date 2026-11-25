"""Compile a codebook into the system and user prompts.

Prompt structure:

    <role>
    <task>
    [<abstention_instruction>]
    <blank>
    <rules_heading>
    <one block per category, alphabetical, fields in `render` order>
    [### Examples:  ...]                       (few_shot only)
    <blank>
    ### Output Format Constraints:
    ...

The domain wording (role, task, rules_heading, reasoning_example) comes from a
named template (templates/*.yaml), overridable per codebook via `prompt:`.
The structure is frozen: with all optional features off, a codebook and template always
produce the same prompt, byte for byte.

Optional features (all off by default) change the prompt and are therefore
recorded in `run.prompt_condition`:
  - few_shot  -> base condition "few_shot" (otherwise "zero_shot")
  - evidence  -> "+evidence": ask for verbatim supporting quotes per category
  - ordinal   -> "+ordinal": codebook `output.type: ordinal`, integer scores or null
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from importlib import resources
from pathlib import Path

import yaml

from .codebook import Codebook, codebook_from_dict

OUTPUT_MARKER = "### Output Format Constraints:"
EXAMPLES_MARKER = "### Examples:"
TEMPLATE_KEYS = ("role", "task", "rules_heading", "reasoning_example")


class PromptError(ValueError):
    """The prompt cannot be built (for example few_shot without examples)."""


class TemplateError(PromptError):
    """Unknown or incomplete prompt template."""


@cache
def _builtin_templates() -> dict[str, dict]:
    out = {}
    for entry in resources.files("codebook_annotator.templates").iterdir():
        if entry.name.endswith(".yaml"):
            t = yaml.safe_load(entry.read_text(encoding="utf-8"))
            out[t["name"]] = t
    return out


def available_templates() -> list[str]:
    return sorted(_builtin_templates())


def load_template(template: str | Path) -> dict:
    """Return a template by built-in name, or load it from a YAML file path."""
    builtins = _builtin_templates()
    if isinstance(template, str) and template in builtins:
        return dict(builtins[template])
    p = Path(template)
    if p.suffix in (".yaml", ".yml") and p.exists():
        try:
            t = yaml.safe_load(p.read_text(encoding="utf-8"))
        except (yaml.YAMLError, UnicodeDecodeError) as e:
            raise TemplateError(f"template {p} cannot be read: {e}") from None
        if not isinstance(t, dict):
            raise TemplateError(f"template {p} must be a mapping with the keys {', '.join(TEMPLATE_KEYS)}")
        missing = [k for k in TEMPLATE_KEYS if k not in t]
        if missing:
            raise TemplateError(f"template {p} lacks keys: {', '.join(missing)}")
        t.setdefault("name", p.stem)
        return t
    raise TemplateError(f"unknown prompt template {template!r}. Built-in: {', '.join(builtins)}")


@dataclass(frozen=True)
class RenderedPrompt:
    system: str
    prompt_condition: str
    template: str
    strings: dict  # the resolved role/task/rules_heading/reasoning_example


def prompt_condition(*, few_shot: bool = False, evidence: bool = False, ordinal: bool = False,
                     verify: bool = False) -> str:
    cond = "few_shot" if few_shot else "zero_shot"
    if evidence:
        cond += "+evidence"
    if ordinal:
        cond += "+ordinal"
    if verify:
        cond += "+verify"
    return cond


def _format_value(value) -> str:
    if isinstance(value, list):
        return ", ".join(f"'{w}'" for w in value)
    if isinstance(value, dict) and "words" in value:
        return ", ".join(f"'{w}'" for w in value["words"])
    return str(value)


def _example_output(cb: Codebook, ex: dict, *, ordinal: bool, evidence: bool) -> str:
    keys = cb.category_keys
    if ordinal:
        scores = ex.get("scores") or {}
        preds = {k: scores.get(k) for k in keys}
    else:
        labels = set(ex.get("labels") or [])
        preds = {k: k in labels for k in keys}
    obj: dict = {"predictions": preds, "reasoning": ex.get("reasoning", "")}
    if evidence:
        obj["evidence"] = {k: v for k, v in (ex.get("evidence") or {}).items()}
    return json.dumps(obj, indent=2, ensure_ascii=False)


def render_prompts(codebook: Codebook, template: str | Path = "default_v1", *, few_shot: bool = False,
                   evidence: bool = False) -> RenderedPrompt:
    """Render the system prompt for a codebook. Ordinal output is taken from the codebook."""
    tpl = load_template(template)
    strings = {k: tpl[k] for k in TEMPLATE_KEYS}
    strings.update(codebook.prompt_overrides)
    ordinal = codebook.output_type == "ordinal"
    categories = codebook.categories
    render_spec = codebook.render_spec

    prompt = [strings["role"], strings["task"], "", strings["rules_heading"]]

    # Optional shared abstention instruction, inserted after the task line.
    # Codebooks without the key are unaffected.
    if codebook.data.get("abstention_instruction"):
        prompt.insert(2, codebook.data["abstention_instruction"])

    sorted_keys = sorted(categories.keys())
    for key in sorted_keys:
        cat = categories[key]
        prompt.append(f"\n- **Category**: {key}")
        for render_entry in render_spec:
            value = cat.get(render_entry["field"])
            if value is None or value == "":
                continue
            prompt.append(f"  * **{render_entry['label']}**: {_format_value(value)}")

    if few_shot:
        examples = codebook.examples
        if not examples:
            raise PromptError("few_shot is on but the codebook has no 'examples'")
        prompt.extend(["", EXAMPLES_MARKER])
        for i, ex in enumerate(examples, 1):
            prompt.append(f"\nExample {i} text:\n{ex['text']}")
            prompt.append(f"Example {i} output:\n{_example_output(codebook, ex, ordinal=ordinal, evidence=evidence)}")

    if ordinal:
        lo, hi = codebook.scale
        example_keys = [f'"{k}": {lo}-{hi} or null' for k in sorted_keys]
        pred_line = (f"1. \"predictions\": A dictionary mapping each category key to an integer score from {lo} to {hi} "
                     f"indicating how strongly the category is expressed, or null if the category is absent.")
        if codebook.scale_description:
            pred_line += f" {codebook.scale_description}"
    else:
        example_keys = [f'"{k}": true/false' for k in sorted_keys]
        pred_line = ("1. \"predictions\": A dictionary mapping each category key to a boolean (true/false) "
                     "indicating its presence/absence.")
    example_predictions = ",\n    ".join(example_keys)

    n_keys = "three" if evidence else "two"
    prompt.extend([
        "",
        OUTPUT_MARKER,
        f"Evaluate the input text against each of the categories. You must output exactly a JSON object containing {n_keys} top-level keys:",
        pred_line,
        "2. \"reasoning\": A brief single-sentence explanation of your evaluation.",
    ])
    if evidence:
        prompt.append(
            "3. \"evidence\": A dictionary mapping each category you judged present to a list of short quotes "
            "copied exactly, character for character, from the input text that support it. Use an empty "
            "dictionary if no category is present.")
    prompt.extend([
        "",
        "Example Output Format (you must use these exact keys):",
        "{",
        f"  \"predictions\": {{\n    {example_predictions}\n  }},",
    ])
    if evidence:
        prompt.append(f"  \"reasoning\": \"{strings['reasoning_example']}\",")
        prompt.append("  \"evidence\": {\n    \"<category key>\": [\"<exact quote from the text>\"]\n  }")
    else:
        prompt.append(f"  \"reasoning\": \"{strings['reasoning_example']}\"")
    prompt.extend([
        "}",
        "",
        "IMPORTANT: Do not return any other text, conversational comments, or markdown wrappers (like ```json). Output ONLY the raw JSON object.",
    ])

    return RenderedPrompt(
        system="\n".join(prompt),
        prompt_condition=prompt_condition(few_shot=few_shot, evidence=evidence, ordinal=ordinal),
        template=str(tpl.get("name", template)),
        strings=strings,
    )


def get_user_prompt(text: str) -> str:
    """Return the user prompt containing the text to evaluate."""
    return f"Evaluate the following text and return the JSON object:\n\n{text}"


def get_verify_prompt(text: str, first_response: str, *, evidence: bool = False) -> str:
    """User prompt for the optional second (verification) pass."""
    extra = " Keep only quotes that appear exactly in the text." if evidence else ""
    return (
        f"{get_user_prompt(text)}\n\n"
        "### Your previous annotation of this text:\n"
        f"{first_response}\n\n"
        "### Verification instructions:\n"
        "Re-check your previous annotation against the coding rules. Keep a category only if the text "
        "supports it under that category's description, and add a category only if the text clearly expresses "
        f"it and it was missed.{extra} Return the corrected JSON object in exactly the same format."
    )


# ── Dict-codebook API (moral_foundations_v1 wording) ─────────────────────────────────────────────────

def load_codebook(filepath: Path | str) -> dict:
    """Load a YAML codebook file as a plain dict (no validation)."""
    with open(filepath, encoding="utf-8") as f:
        return yaml.safe_load(f)


def compile_system_prompt(codebook: dict) -> str:
    """Render a dict codebook with the frozen 'moral_foundations_v1' template."""
    return render_prompts(codebook_from_dict(codebook, strict=False), template="moral_foundations_v1").system
