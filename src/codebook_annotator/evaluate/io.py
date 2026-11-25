"""Loading gold data and run records, plus category and field helpers."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# Default gold-data field names.
DEFAULT_ID_FIELD = "item_id"
DEFAULT_MAJORITY_FIELD = "label_majority"
DEFAULT_ANNOTATORS_FIELD = "labels_per_annotator"
MISSING_VALUE = "(missing)"


@dataclass(frozen=True)
class GoldFields:
    """Names of the gold-data fields the evaluators read."""

    id_field: str = DEFAULT_ID_FIELD
    majority_field: str = DEFAULT_MAJORITY_FIELD
    annotators_field: str = DEFAULT_ANNOTATORS_FIELD


DEFAULT_FIELDS = GoldFields()


class EvalDataError(ValueError):
    """Unusable evaluation input (missing file, no readable records, missing field)."""


_INT_LIKE = re.compile(r"^-?\d+$")


def id_sort_key(ids) -> Callable[[str], Any]:
    """Sort key for item ids: numeric when every id looks like an integer, text otherwise."""
    ids = [str(i) for i in ids]
    if ids and all(_INT_LIKE.match(i) for i in ids):
        return lambda i: (int(i), str(i))
    return str


def sort_ids(ids) -> list[str]:
    ids = list(ids)
    return sorted(ids, key=id_sort_key(ids))


def read_jsonl_records(filepath: Path | str, id_field: str = DEFAULT_ID_FIELD) -> list[dict]:
    """Read a JSONL file into a list of records (file order), robustly.

    Each line is parsed on its own. A line that does not parse starts a buffer for a record
    spread over several lines. If a later line parses on its own as an object with ``id_field``,
    the buffer was an unreadable fragment: it is dropped and the line is kept. Unreadable
    fragments and records without ``id_field`` are skipped with a printed note, so one bad line
    never hides the valid records after it.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise EvalDataError(f"file not found: {filepath}")
    records: list[dict] = []
    unreadable = 0
    no_id = 0
    buffer = ""

    def keep(obj) -> None:
        nonlocal no_id
        if isinstance(obj, dict) and id_field in obj:
            records.append(obj)
        else:
            no_id += 1

    with open(filepath, encoding="utf-8-sig") as f:
        for line in f:
            if not buffer:
                if not line.strip():
                    continue
                try:
                    keep(json.loads(line))
                except json.JSONDecodeError:
                    buffer = line
                continue
            if line.strip():
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    obj = None
                if isinstance(obj, dict) and id_field in obj:
                    unreadable += 1
                    buffer = ""
                    records.append(obj)
                    continue
            buffer += line
            try:
                keep(json.loads(buffer))
                buffer = ""
            except json.JSONDecodeError:
                continue
    if buffer.strip():
        unreadable += 1
    if unreadable:
        print(f"Note: {filepath.name}: {unreadable} unreadable line(s) or fragment(s) skipped.")
    if no_id:
        print(f"Note: {filepath.name}: {no_id} record(s) without the field '{id_field}' skipped.")
    return records


def load_jsonl_indexed(filepath: Path | str, id_field: str = DEFAULT_ID_FIELD) -> dict[str, dict]:
    """Load a JSONL file indexed by ``str(record[id_field])`` (see ``read_jsonl_records``).

    Ids are compared as text, so a numeric id in the gold file matches the same id in a run.
    Later records with the same id replace earlier ones, with a printed note.
    """
    out: dict[str, dict] = {}
    duplicates = 0
    for rec in read_jsonl_records(filepath, id_field):
        key = str(rec[id_field])
        duplicates += key in out
        out[key] = rec
    if duplicates:
        print(f"Note: {Path(filepath).name}: {duplicates} duplicate id(s), the last record of each was kept.")
    return out


def load_dataset(filepath: Path | str, fields: GoldFields = DEFAULT_FIELDS) -> dict[str, dict]:
    """Load the gold dataset (JSONL) indexed by the configured id field (as text)."""
    dataset = load_jsonl_indexed(filepath, fields.id_field)
    check_gold(dataset, filepath, fields)
    return dataset


def check_gold(dataset: dict[str, dict], filepath: Path | str, fields: GoldFields) -> None:
    """Raise EvalDataError when the gold data cannot be evaluated against."""
    if not dataset:
        raise EvalDataError(f"no readable records with the id field '{fields.id_field}' in {filepath}. "
                            "Check the file and --id-field.")
    if not any(fields.majority_field in g for g in dataset.values()):
        raise EvalDataError(f"no record in {filepath} has the majority field '{fields.majority_field}'. "
                            "Check --majority-field.")


def load_run_predictions(filepath: Path | str) -> dict[str, dict]:
    """Load a run file indexed by ``item_id`` (run records always use ``item_id``)."""
    return load_jsonl_indexed(filepath, "item_id")


def majority(gold: dict, fields: GoldFields) -> list:
    return gold.get(fields.majority_field) or []


def annotators(gold: dict, fields: GoldFields) -> dict:
    return gold.get(fields.annotators_field) or {}


def has_annotators(dataset: dict[str, dict], fields: GoldFields) -> bool:
    """True if at least one gold item carries the per-annotator labels field."""
    return any(fields.annotators_field in g for g in dataset.values())


def categories_from_codebook(path: Path | str) -> list[str]:
    """Category keys in the order they appear in a codebook YAML file."""
    with open(path, encoding="utf-8") as f:
        cb = yaml.safe_load(f)
    return list((cb or {}).get("categories", {}).keys())


def categories_from_runs(runs: list[dict[str, dict]]) -> list[str]:
    """Sorted union of the ``predictions`` keys across all run records."""
    cats: set[str] = set()
    for preds in runs:
        for rec in preds.values():
            cats.update((rec.get("predictions") or {}).keys())
    return sorted(cats)


def resolve_categories(
    explicit: list[str] | None,
    codebook: str | None,
    runs: list[dict[str, dict]],
) -> list[str]:
    """Category order: --categories, else --codebook order, else sorted union of run predictions."""
    if explicit:
        return list(explicit)
    if codebook:
        return categories_from_codebook(codebook)
    return categories_from_runs(runs)


def parse_bool_labels(specs: list[str] | None) -> dict[str, tuple[str, str]]:
    """Parse ``FIELD=TRUE_LABEL,FALSE_LABEL`` specs."""
    out: dict[str, tuple[str, str]] = {}
    for spec in specs or []:
        field, _, labels = spec.partition("=")
        true_label, sep, false_label = labels.partition(",")
        if not field or not sep:
            raise ValueError(f"--bool-labels expects FIELD=TRUE_LABEL,FALSE_LABEL, got: {spec!r}")
        out[field] = (true_label, false_label)
    return out


def slice_value(gold: dict, field: str, bool_labels: dict[str, tuple[str, str]]) -> str:
    """Slice label of a gold item for ``field``.

    With bool labels configured for the field, the value's truthiness selects the
    label (a missing value counts as false). Otherwise the value is rendered as a
    string, and a missing/null value becomes ``(missing)``.
    """
    if field in bool_labels:
        true_label, false_label = bool_labels[field]
        return true_label if gold.get(field, False) else false_label
    val = gold.get(field)
    if val is None:
        return MISSING_VALUE
    return str(val)


def slice_sheet_name(field: str) -> str:
    """Excel sheet name for a slice field, e.g. ``lexical_overlap`` -> ``Lexical Overlap Slices``."""
    return (field.replace("_", " ").title() + " Slices")[:31]


def prompt_condition(meta: dict[str, Any]) -> str:
    return str(meta.get("prompt_condition") or "")


def is_ordinal_run(records: dict[str, dict]) -> bool:
    first = next(iter(records.values()), {})
    return "+ordinal" in prompt_condition(first.get("run", {})) or any("scores" in r for r in records.values())


def is_evidence_run(records: dict[str, dict]) -> bool:
    first = next(iter(records.values()), {})
    return "+evidence" in prompt_condition(first.get("run", {})) or any("evidence" in r for r in records.values())
