"""Run a codebook against a model over a dataset sample and write one JSONL record per item.

Behaviour:
  - items are processed in sample order, and each record is appended and flushed immediately.
  - an existing output file is resumed. Records with parse_status 'ok' are kept and the rest are
    re-annotated (they are moved to <run_id>.attempts.jsonl rather than deleted).
  - query errors and unparseable responses are recorded with parse_status 'failed'. The run
    exits with status 1 if any item is not 'ok' (fail loudly).

Provenance per record: tool_version, prompt_template, codebook/system-prompt/config/sample
sha256 digests, observed model digest (Ollama), token usage and latency.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import tempfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from . import __version__
from .codebook import CodebookError, load_codebook
from .config import RunConfig, config_fingerprint
from .data import DataError, load_records, sample_fingerprint, select_sample
from .parsing import NOTE_SEPARATOR, parse_response
from .prompt import get_user_prompt, get_verify_prompt, render_prompts
from .providers import make_provider
from .providers.adapters import OllamaProvider
from .providers.base import Provider, ProviderResponse

SCHEMA_VERSION = 2
NO_HASH_RESUME_KEYS = ("codebook_id", "model_tag", "provider", "temperature")


class RunError(RuntimeError):
    pass


@dataclass
class RunSummary:
    run_id: str
    output_file: Path
    sample_size: int
    completed: int
    resumed: int
    errors: int
    fallbacks: int
    format_warnings: int

    @property
    def exit_code(self) -> int:
        return 1 if self.errors else 0


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _digests_match(a: str, b: str) -> bool:
    a = a.lower().removeprefix("sha256:")
    b = b.lower().removeprefix("sha256:")
    n = min(len(a), len(b))
    return n >= 12 and a[:n] == b[:n]


def _timestamp() -> str:
    return datetime.datetime.now(datetime.UTC).replace(tzinfo=None).isoformat() + "Z"


_NOT_JSON = object()


def _loads(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return _NOT_JSON


def _is_record(value) -> bool:
    return isinstance(value, dict) and isinstance(value.get("item_id"), str) and isinstance(value.get("run"), dict)


def _read_existing(path: Path) -> tuple[list[dict], list[str]]:
    """Read the records of an existing run file. Returns (records, unreadable chunks).

    Each line is read as one record. Lines are only joined when a record really spans several
    lines (pretty-printed JSON). A damaged line never hides the records after it: as soon as a
    line is a complete record on its own, the text collected before it is set aside as unreadable.
    Values that are not run records (no item_id or run) are set aside as well.
    """
    records, junk = [], []
    buffer = ""

    def take(value, raw: str) -> None:
        (records.append(value) if _is_record(value) else junk.append(raw))

    with open(path, encoding="utf-8") as f:
        for line in f:
            if not buffer:
                if not line.strip():
                    continue
                value = _loads(line)
                if value is _NOT_JSON:
                    buffer = line
                else:
                    take(value, line)
                continue
            single = _loads(line)
            if single is not _NOT_JSON and _is_record(single):
                junk.append(buffer)
                buffer = ""
                records.append(single)
                continue
            buffer += line
            value = _loads(buffer)
            if value is not _NOT_JSON:
                take(value, buffer)
                buffer = ""
    if buffer.strip():
        junk.append(buffer)
    return records, junk


# Recorded run fields that explain why a resume is refused, with readable names.
_EXPLAINED_FIELDS = {
    "codebook_sha256": "codebook file", "system_prompt_sha256": "prompt (codebook, template or features)",
    "sample_sha256": "sample (dataset, split or sample size)", "model_tag": "model_tag",
    "model_digest": "model version (model_digest)", "provider": "provider", "temperature": "temperature",
    "seed": "seed", "prompt_condition": "prompt condition", "context_window": "num_ctx",
    "codebook_id": "codebook_id", "dataset": "dataset name",
}


def _describe_differences(records: list[dict], current: dict) -> str:
    diffs = {}
    for rec in records:
        for key, label in _EXPLAINED_FIELDS.items():
            if key in rec["run"] and rec["run"][key] != current.get(key) and label not in diffs:
                diffs[label] = (rec["run"][key], current.get(key))
    if not diffs:
        return ("The difference is in a setting that is not stored in the records "
                "(for example max_tokens, sampling, evidence_policy or the replay file).")
    parts = []
    for label, (old, new) in diffs.items():
        old_s, new_s = str(old), str(new)
        if len(old_s) > 24:
            old_s, new_s = old_s[:12] + "...", new_s[:12] + "..."
        parts.append(f"{label}: {old_s} in the file, {new_s} now")
    return "Differences: " + ". ".join(parts) + "."


def _write_atomic(path: Path, records: list[dict]) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


@dataclass
class _Context:
    cfg: RunConfig
    provider: Provider
    system_prompt: str
    categories: list[str]
    output_type: str
    scale: tuple[int, int]
    run_meta: dict


def _process_item(ctx: _Context, item: dict) -> dict:
    cfg = ctx.cfg
    iid, text = item["item_id"], item["text"]
    parse_kw = dict(output_type=ctx.output_type, scale=ctx.scale, evidence=cfg.evidence, text=text,
                    include_raw_excerpt=cfg.store_raw_response)
    passes: list[ProviderResponse] = []
    first_pass = None
    try:
        resp = ctx.provider.query(ctx.system_prompt, get_user_prompt(text), item_id=iid)
        passes.append(resp)
        if resp.error:
            res = parse_response("", ctx.categories, **parse_kw)
            res.parse_status, res.parse_notes, res.reasoning = "failed", f"Harness query error: {resp.error}", None
            raw = ""
        else:
            res = parse_response(resp.text, ctx.categories, **parse_kw)
            raw = resp.text

        if cfg.verify and not resp.error:
            first_pass = {"raw_response": raw, "parse_status": res.parse_status, "parse_notes": res.parse_notes,
                          "predictions": res.predictions, "reasoning": res.reasoning}
            if res.scores is not None:
                first_pass["scores"] = res.scores
            if res.evidence is not None:
                first_pass["evidence"] = res.evidence
            if res.parse_status == "failed":
                res.parse_notes = NOTE_SEPARATOR.join([res.parse_notes, "verify pass skipped (first pass failed)"])
            else:
                vresp = ctx.provider.query(ctx.system_prompt, get_verify_prompt(text, raw.strip(), evidence=cfg.evidence),
                                           item_id=iid)
                passes.append(vresp)
                if vresp.error:
                    res = parse_response("", ctx.categories, **parse_kw)
                    res.parse_status = "failed"
                    res.parse_notes = f"Harness query error (verify pass): {vresp.error}"
                    res.reasoning = None
                    raw = ""
                else:
                    res = parse_response(vresp.text, ctx.categories, **parse_kw)
                    raw = vresp.text
    except Exception as e:  # one failing item never stops the run. It is recorded and retried on resume
        res = parse_response("", ctx.categories, **parse_kw)
        res.parse_status, res.parse_notes, res.reasoning = "failed", f"Harness internal error: {e!r}", None
        raw = ""

    enforced, before = [], None
    if cfg.evidence and cfg.evidence_policy == "enforce" and res.parse_status != "failed":
        before = dict(res.predictions)
        for cat in ctx.categories:
            if res.predictions[cat] and not any(q["verified"] for q in (res.evidence or {}).get(cat, [])):
                res.predictions[cat] = False
                if res.scores is not None:
                    res.scores[cat] = None
                enforced.append(cat)

    run = dict(ctx.run_meta)
    run["timestamp"] = _timestamp()
    rec = {"item_id": iid}
    if cfg.store_text:
        rec["text"] = text
    rec["run"] = run
    rec["parse_status"] = res.parse_status
    rec["parse_notes"] = res.parse_notes
    rec["predictions"] = res.predictions
    if cfg.store_raw_response:
        rec["raw_response"] = raw
    rec["reasoning"] = res.reasoning
    if res.scores is not None:
        rec["scores"] = res.scores
    if cfg.evidence:
        rec["evidence"] = res.evidence or {}
        if cfg.evidence_policy == "enforce":
            rec["evidence_enforced"] = enforced
            rec["predictions_before_evidence"] = before
    if first_pass is not None:
        if not cfg.store_raw_response:
            first_pass.pop("raw_response", None)
        rec["first_pass"] = first_pass

    def total(attr):
        vals = [getattr(p, attr) for p in passes if getattr(p, attr) is not None]
        return sum(vals) if vals else None

    rec["usage"] = {"input_tokens": total("input_tokens"), "output_tokens": total("output_tokens"),
                    "latency_ms": None if total("latency_ms") is None else round(total("latency_ms"), 1),
                    "passes": len(passes)}
    rec["_fallback"] = res.used_fallback
    rec["_format_warnings"] = res.format_warnings
    rec["schema_version"] = SCHEMA_VERSION
    return rec


def prepare(cfg: RunConfig, *, provider: Provider | None = None, log: Callable[[str], None] = print,
            dry_run: bool = False):
    """Load and validate everything needed for a run (no model calls except the digest lookup)."""
    try:
        codebook = load_codebook(cfg.codebook)
    except (CodebookError, OSError) as e:
        raise RunError(f"Error loading codebook YAML: {e}") from e
    for w in codebook.warnings:
        log(f"  codebook {w}")
    cfg.codebook_id = cfg.codebook_id or codebook.id
    cfg.codebook_version = cfg.codebook_version or codebook.version or "1.0"
    categories = codebook.category_keys

    if cfg.few_shot and not codebook.examples:
        raise RunError("few_shot is on but the codebook has no 'examples'")
    try:
        rendered = render_prompts(codebook, cfg.prompt_template, few_shot=cfg.few_shot, evidence=cfg.evidence)
    except ValueError as e:
        raise RunError(str(e)) from e
    condition = rendered.prompt_condition + ("+verify" if cfg.verify else "")

    try:
        records = load_records(cfg.dataset, sheet=cfg.dataset_sheet)
        sample = select_sample(records, id_field=cfg.id_field, text_field=cfg.text_field,
                               split_field=cfg.split_field, split_value=cfg.split_value,
                               sample_size=cfg.sample_size, sampling=cfg.sampling,
                               sampling_seed=cfg.sampling_seed, log=log)
    except DataError as e:
        raise RunError(str(e)) from e
    sample_sha = sample_fingerprint(sample)
    dataset_name = cfg.dataset_name or Path(cfg.dataset).stem
    sample_id = cfg.sample_id or f"{dataset_name}_n{len(sample)}_{sample_sha[:8]}"

    if provider is None:
        try:
            provider = make_provider(cfg, categories)
        except ValueError as e:
            raise RunError(str(e)) from e

    if isinstance(provider, OllamaProvider):
        try:
            available = provider.available_models()
        except Exception as e:
            raise RunError(f"Cannot reach Ollama at {provider.host} ({e}). Is `ollama serve` running?") from e
        if cfg.model_tag not in available and f"{cfg.model_tag}:latest" not in available:
            raise RunError(f"Ollama model '{cfg.model_tag}' is not installed. Installed: "
                           f"{', '.join(available) or '(none)'}. Pull it with `ollama pull {cfg.model_tag}` "
                           f"or choose another model_tag.")
    observed = provider.observed_model_digest() if cfg.provider == "ollama" else None
    digest = cfg.model_digest
    if digest == "auto" and dry_run:
        digest = None  # not resolved in a dry run (no server contact)
    elif digest == "auto" and cfg.provider != "ollama":
        log(f"Note: model_digest 'auto' only works with the ollama provider. "
            f"No digest is recorded for provider '{cfg.provider}'.")
        digest = None
    elif digest == "auto":
        if observed is None:
            raise RunError("model_digest 'auto' needs a running Ollama server that has the model "
                           f"'{cfg.model_tag}' (check `ollama list`)")
        digest = observed
    elif digest and observed and not _digests_match(digest, observed):
        msg = (f"Configured model_digest {digest} does not match the digest Ollama reports for "
               f"'{cfg.model_tag}': {observed}")
        if cfg.strict_digest:
            raise RunError(msg + " (set strict_digest: false to run anyway)")
        log("Warning: " + msg)

    system_sha = _sha256(rendered.system)
    fingerprint = config_fingerprint(cfg, codebook_sha256=codebook.sha256, system_prompt_sha256=system_sha,
                                     sample_sha256=sample_sha, model_digest=digest)
    run_meta = {
        "dataset": dataset_name,
        "dataset_revision": cfg.dataset_revision,
        "codebook_id": cfg.codebook_id,
        "codebook_version": cfg.codebook_version,
        "model_tag": cfg.model_tag,
        "model_digest": digest,
        "provider": cfg.provider,
        "prompt_condition": condition,
        "sample_id": sample_id,
        "context_window": cfg.num_ctx,
        "temperature": cfg.temperature,
        "seed": cfg.seed,
        "timestamp": None,
        "tool_version": __version__,
        "prompt_template": rendered.template,
        "codebook_sha256": codebook.sha256,
        "system_prompt_sha256": system_sha,
        "config_sha256": fingerprint,
        "sample_sha256": sample_sha,
    }
    if observed:
        run_meta["model_digest_observed"] = observed
    ctx = _Context(cfg, provider, rendered.system, categories, codebook.output_type, codebook.scale, run_meta)
    return ctx, sample, rendered


def run(cfg: RunConfig, *, provider: Provider | None = None, log: Callable[[str], None] = print) -> RunSummary:
    ctx, sample, _ = prepare(cfg, provider=provider, log=log)
    log(f"Executing run '{cfg.run_id}' against model '{cfg.model_tag}' ({cfg.provider})")
    log(f"Sample size: {len(sample)} items")
    log(f"Prompt: template={ctx.run_meta['prompt_template']} condition={ctx.run_meta['prompt_condition']} "
        f"system_prompt_sha256={ctx.run_meta['system_prompt_sha256'][:12]}")

    out_file = cfg.output_file
    out_file.parent.mkdir(parents=True, exist_ok=True)
    attempts_file = out_file.with_suffix(".attempts.jsonl")

    existing_ids: set[str] = set()
    fallbacks = format_warnings = canonical = 0
    if out_file.exists():
        records, junk = _read_existing(out_file)
        mismatched = []
        for rec in records:
            r = rec["run"]
            if "config_sha256" in r:
                if r["config_sha256"] != ctx.run_meta["config_sha256"]:
                    mismatched.append(rec)
            elif any(r.get(k) != ctx.run_meta[k] for k in NO_HASH_RESUME_KEYS):
                mismatched.append(rec)
        if mismatched and not cfg.force_resume:
            raise RunError(
                f"{out_file} contains {len(mismatched)} record(s) produced with a different configuration. "
                f"{_describe_differences(mismatched, ctx.run_meta)} Use a new run_id, delete the file, or set "
                f"force_resume: true to mix them deliberately.")
        kept: dict[str, dict] = {}
        drop, duplicates = [], 0
        for rec in records:
            if rec.get("parse_status") != "ok":
                drop.append(rec)
                continue
            if rec["item_id"] in kept:
                duplicates += 1
                drop.append(kept[rec["item_id"]])
            kept[rec["item_id"]] = rec
        keep = list(kept.values())
        for rec in keep:
            existing_ids.add(rec["item_id"])
            if "Fallback key used" in (rec.get("parse_notes") or ""):
                fallbacks += 1
            else:
                canonical += 1
        if drop or junk:
            with open(attempts_file, "a", encoding="utf-8") as f:
                for rec in drop:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                for chunk in junk:
                    f.write(json.dumps({"unparseable_output_chunk": chunk}, ensure_ascii=False) + "\n")
            parts = [f"{len(drop) - duplicates} non-ok record(s)"]
            if duplicates:
                parts.append(f"{duplicates} duplicate record(s)")
            if junk:
                parts.append(f"{len(junk)} unreadable chunk(s)")
            log(f"  Moved {', '.join(parts)} to {attempts_file}")
        _write_atomic(out_file, keep)
        if existing_ids:
            log(f"Resuming run '{cfg.run_id}': {len(existing_ids)}/{len(sample)} items already annotated. "
                f"Annotating remaining {len(sample) - len(existing_ids & {s['item_id'] for s in sample})} items...")

    todo = [it for it in sample if it["item_id"] not in existing_ids]
    completed = len(sample) - len(todo)
    resumed = completed
    errors = 0

    def write(out, rec):
        nonlocal completed, errors, fallbacks, format_warnings, canonical
        fb, fw = rec.pop("_fallback"), rec.pop("_format_warnings")
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out.flush()
        completed += 1
        if rec["parse_status"] != "ok":
            errors += 1
        elif not fb:
            canonical += 1
        fallbacks += fb
        format_warnings += fw
        if completed % 50 == 0 or completed == len(sample):
            log(f"  Progress: {completed}/{len(sample)} items annotated")

    try:
        with open(out_file, "a", encoding="utf-8") as out:
            if cfg.concurrency == 1:
                for item in todo:
                    write(out, _process_item(ctx, item))
            else:
                with ThreadPoolExecutor(max_workers=cfg.concurrency) as pool:
                    for rec in pool.map(lambda it: _process_item(ctx, it), todo):
                        write(out, rec)
    finally:
        ctx.provider.close()

    log(f"✓ Run finished. Output written to: {out_file}")
    log(f"  Total items: {completed}/{len(sample)}")
    log(f"  - Canonical parses ('predictions' key): {canonical}/{completed}")
    if fallbacks > 0:
        log(f"  ⚠ Format drift / Fallback key activations: {fallbacks}/{completed} "
            f"(used alternative keys instead of 'predictions')")
    if format_warnings > 0:
        log(f"  ⚠ Non-boolean/score format warnings: {format_warnings} (see parse_notes)")
    if errors > 0:
        log(f"  ✖ Parse / Query errors: {errors}/{completed}")
    return RunSummary(cfg.run_id, out_file, len(sample), completed, resumed, errors, fallbacks, format_warnings)
