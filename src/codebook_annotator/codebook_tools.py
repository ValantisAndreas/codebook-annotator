"""Authoring helpers: spreadsheet -> codebook, ablation variants, prompt-invariance check."""

from __future__ import annotations

import copy
import datetime
import difflib
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .codebook import DEFAULT_RENDER, Codebook, CodebookError
from .data import load_records
from .prompt import OUTPUT_MARKER, render_prompts


def estimate_tokens(text: str) -> int:
    """Rough English token estimate (~1.3 tokens per word)."""
    return int(len(text.split()) * 1.3) if text else 0


def default_label(fld: str) -> str:
    s = fld.replace("_", " ").strip()
    return s[:1].upper() + s[1:]


def dump_codebook(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, sort_keys=False, allow_unicode=True, width=120)


def _canonical(name: str, expected: list[str] | None) -> str:
    if not expected:
        return name
    for c in expected:
        if c.lower() == name.lower():
            return c
    raise CodebookError(f"unexpected category '{name}'. Expected one of: {', '.join(expected)}")


def table_to_codebook(path: str | Path, *, category_column: str | None = None, sheet: str | None = None,
                      columns: list[str] | None = None, labels: dict[str, str] | None = None,
                      list_columns: set[str] = frozenset(), list_sep: str = ",",
                      expected_categories: list[str] | None = None, allow_empty: bool = False,
                      add_name: bool = False, codebook_id: str | None = None, version: str | None = None,
                      author: str | None = None) -> dict:
    """Build a codebook dict from a table with one row per category.

    Every column other than the category column (or only `columns`, in that order) becomes a
    field, rendered in column order with `labels[col]` (default: the column name, humanised).
    Empty cells are an error unless allow_empty is set.
    """
    rows = load_records(path, sheet=sheet)
    if not rows:
        raise CodebookError(f"{path}: no rows")
    header = list(rows[0].keys())
    cat_col = category_column or header[0]
    if cat_col not in header:
        raise CodebookError(f"category column '{cat_col}' not found. Columns: {', '.join(header)}")
    fields = columns or [h for h in header if h and h != cat_col]
    missing = [c for c in fields if c not in header]
    if missing:
        raise CodebookError(f"columns not found: {', '.join(missing)}. Available: {', '.join(header)}")
    labels = labels or {}

    categories: dict[str, dict] = {}
    errors = []
    for i, row in enumerate(rows, 2):
        raw_name = row.get(cat_col)
        if raw_name is None or str(raw_name).strip() == "":
            errors.append(f"row {i}: empty category name")
            continue
        try:
            name = _canonical(str(raw_name).strip(), expected_categories)
        except CodebookError as e:
            errors.append(f"row {i}: {e}")
            continue
        if name in categories:
            errors.append(f"row {i}: duplicate category '{name}'")
            continue
        cat: dict = {"name": name} if add_name else {}
        for col in fields:
            val = row.get(col)
            if val is None or str(val).strip() == "":
                if not allow_empty:
                    errors.append(f"category '{name}': field '{col}' is empty")
                cat[col] = None
                continue
            if col in list_columns:
                cat[col] = [w.strip() for w in str(val).split(list_sep) if w.strip()]
            else:
                cat[col] = str(val).strip()
        categories[name] = cat
    if expected_categories:
        absent = sorted(set(expected_categories) - set(categories))
        if absent:
            errors.append(f"missing categories: {', '.join(absent)}")
    if errors:
        raise CodebookError("table conversion failed:\n" + "\n".join(f"  {e}" for e in errors))

    render = ([{"field": "name", "label": "Name"}] if add_name else []) + [
        {"field": c, "label": labels.get(c, default_label(c))} for c in fields]
    out: dict = {}
    if codebook_id:
        out["id"] = codebook_id
    if version:
        out["version"] = str(version)
    prov = {"source": Path(path).name, "converted_on": datetime.date.today().isoformat()}
    if author:
        prov["author"] = author
    out["provenance"] = prov
    out["render"] = render
    out["categories"] = categories
    return out


def token_report(codebook: dict) -> dict[str, int]:
    """Estimated tokens per category over the rendered fields."""
    render = codebook.get("render", DEFAULT_RENDER)
    out = {}
    for key, cat in codebook["categories"].items():
        parts = []
        for r in render:
            v = cat.get(r["field"])
            if v is None or v == "":
                continue
            parts.append(" ".join(map(str, v)) if isinstance(v, list) else str(v))
        out[key] = estimate_tokens(" ".join(parts))
    return out


CARRIED_KEYS = ("prompt", "abstention_instruction", "output", "examples")


def build_ablations(base: dict, extension: dict[str, dict], fields: list[tuple[str, str]], *,
                    names: dict[str, str] | None = None, prefix: str = "variant", full_name: str = "full",
                    single: bool = True, full: bool = True, base_name: str = "base",
                    author: str | None = None, date: str | None = None) -> dict[str, dict]:
    """Create codebook variants = base codebook + added field(s) from `extension`.

    Variants: one per field (`<prefix>_<names[field] or field>`) and/or one with all fields
    (`<prefix>_<full_name>`). Base category fields are copied verbatim (checked), render order is
    the base render followed by the added fields, and prompt-level keys of the base codebook
    (prompt, abstention_instruction, output, examples) are carried over unchanged.
    """
    names = names or {}
    base_cats = base["categories"]
    unknown = sorted(set(extension) - set(base_cats))
    if unknown:
        raise CodebookError(f"extension has categories not in the base codebook: {', '.join(unknown)}")
    existing = sorted({f for f, _ in fields if any(f in c for c in base_cats.values())})
    if existing:
        raise CodebookError(f"added field(s) already exist in the base codebook: {', '.join(existing)}. "
                            "Added fields must be new.")
    base_render = list(base.get("render", DEFAULT_RENDER))
    date = date or datetime.date.today().isoformat()

    specs = []
    if single:
        specs += [([f], f"{prefix}_{names.get(f[0], f[0])}", f[0]) for f in fields]
    if full and (len(fields) > 1 or not single):  # with one field and single on, "full" would duplicate it
        specs.append((fields, f"{prefix}_{full_name}", f"all {len(fields)} added fields"))

    variants = {}
    for active, name, desc in specs:
        cats = {}
        for key, cat in base_cats.items():
            c = copy.deepcopy(cat)
            ext = extension.get(key, {})
            for fkey, _ in active:
                val = ext.get(fkey)
                c[fkey] = None if val is None else str(val).strip()
            cats[key] = c
        out: dict = {"provenance": {
            "source": f"Ablation variant of {base_name}: base category fields verbatim + {desc}",
            "base_codebook": base_name,
            "added_fields": [k for k, _ in active],
            "date": date,
        }}
        if author:
            out["provenance"]["author"] = author
        for k in CARRIED_KEYS:
            if k in base:
                out[k] = copy.deepcopy(base[k])
        out["render"] = base_render + [{"field": k, "label": lab} for k, lab in active]
        out["categories"] = cats
        # verbatim check of base fields
        for key, cat in base_cats.items():
            for fk, fv in cat.items():
                if out["categories"][key].get(fk) != fv:
                    raise CodebookError(f"internal error: base field {key}.{fk} changed in {name}")
        variants[name] = out
    return variants


def load_extension_table(path: str | Path, *, category_column: str | None = None,
                         sheet: str | None = None) -> dict[str, dict]:
    rows = load_records(path, sheet=sheet)
    if not rows:
        return {}
    cat_col = category_column or list(rows[0].keys())[0]
    out = {}
    for row in rows:
        name = row.get(cat_col)
        if name is None or str(name).strip() == "":
            continue
        out[str(name).strip()] = row
    return out


@dataclass
class InvarianceReport:
    reference: str
    results: list[dict] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(r["ok"] for r in self.results)


def check_invariance(codebooks: list[Codebook], *, template: str = "default_v1", few_shot: bool = False,
                     evidence: bool = False, names: list[str] | None = None) -> InvarianceReport:
    """Check that prompts differ ONLY in the codebook block.

    For the first codebook (reference) against each other one: the preamble (up to and including
    the rules heading) and everything from the output-format section on must be identical, and
    the category block must differ. The user prompt never depends on the codebook.
    """
    if len(codebooks) < 2:
        raise CodebookError("check-invariance needs at least two codebooks (the first one is the reference)")
    names = names or [str(cb.path or i) for i, cb in enumerate(codebooks)]
    rendered = [render_prompts(cb, template, few_shot=few_shot, evidence=evidence) for cb in codebooks]

    def split(rp):
        lines = rp.system.splitlines()
        heading = rp.strings["rules_heading"].splitlines() or [""]
        k = len(heading)
        h = next((i for i in range(len(lines) - k + 1) if lines[i:i + k] == heading), None)
        o = next((i for i, ln in enumerate(lines) if ln == OUTPUT_MARKER), None)
        if h is None or o is None or o < h:
            raise CodebookError("cannot locate the rules heading and the output section in the rendered prompt")
        return lines[:h + k], lines[h + k:o], lines[o:]

    ref_pre, ref_block, ref_post = split(rendered[0])
    report = InvarianceReport(reference=names[0])
    for name, rp in zip(names[1:], rendered[1:], strict=True):
        pre, block, post = split(rp)
        res = {"codebook": name, "preamble_identical": pre == ref_pre, "output_section_identical": post == ref_post,
               "codebook_block_differs": block != ref_block, "diff": ""}
        res["ok"] = res["preamble_identical"] and res["output_section_identical"] and res["codebook_block_differs"]
        if not res["preamble_identical"]:
            res["diff"] += "\n".join(difflib.unified_diff(ref_pre, pre, names[0], name, lineterm=""))
        if not res["output_section_identical"]:
            res["diff"] += "\n".join(difflib.unified_diff(ref_post, post, names[0], name, lineterm=""))
        report.results.append(res)
    return report
