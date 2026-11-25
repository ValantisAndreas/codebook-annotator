"""Loading input items and drawing the run sample.

Supported input formats: .jsonl, .json (a list of objects), .csv, .tsv, .xlsx (first sheet
unless `sheet` is given). Each record needs an id field and a text field (names configurable).

Sampling ('first_n_by_id'): optionally keep records whose `split_field` equals `split_value`
(if none match, all records are used, with a warning), sort by id (numerically when every id is a
whole number), then take the first
`sample_size` (all if None). 'random' draws `sample_size` records
with `sampling_seed` and keeps them in id order.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from collections.abc import Callable
from pathlib import Path


class DataError(ValueError):
    pass


_INT_ID = re.compile(r"-?\d+")


def load_records(path: str | Path, *, sheet: str | None = None) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise DataError(f"Dataset file not found at: {path}")
    try:
        return _load_records(path, sheet)
    except UnicodeDecodeError as e:
        raise DataError(f"{path}: not UTF-8 text ({e.reason} at byte {e.start}). Save the file as UTF-8") from None


def _load_records(path: Path, sheet: str | None) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        records = []
        with open(path, encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        raise DataError(f"{path}:{n}: invalid JSON: {e}") from e
        return records
    if suffix == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise DataError(f"{path}: invalid JSON: {e}") from None
        if not isinstance(data, list):
            raise DataError(f"{path}: expected a JSON list of objects")
        return data
    if suffix in (".csv", ".tsv"):
        with open(path, encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f, delimiter="\t" if suffix == ".tsv" else ","))
    if suffix == ".xlsx":
        try:
            import openpyxl
        except ImportError as e:  # pragma: no cover
            raise DataError("reading .xlsx needs openpyxl: pip install 'codebook-annotator[excel]'") from e
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        if sheet and sheet not in wb.sheetnames:
            raise DataError(f"{path}: sheet '{sheet}' not found. Sheets: {', '.join(wb.sheetnames)}")
        ws = wb[sheet] if sheet else wb.worksheets[0]
        rows = ws.iter_rows(values_only=True)
        first = next(rows, None)
        if first is None:
            raise DataError(f"{path}: the sheet '{ws.title}' is empty")
        header = [str(h) if h is not None else "" for h in first]
        return [dict(zip(header, r, strict=False)) for r in rows if any(v is not None for v in r)]
    raise DataError(f"unsupported dataset format '{suffix}' (use .jsonl, .json, .csv, .tsv or .xlsx)")


def select_sample(records: list[dict], *, id_field: str = "item_id", text_field: str = "text",
                  split_field: str | None = "split", split_value: str | None = None,
                  sample_size: int | None = None, sampling: str = "first_n_by_id", sampling_seed: int = 42,
                  log: Callable[[str], None] = print) -> list[dict]:
    """Return the sample as a list of dicts with 'item_id' and 'text' (plus the original fields)."""
    if not records:
        raise DataError("dataset is empty")
    for fld in (id_field, text_field):
        if fld not in records[0]:
            raise DataError(f"field '{fld}' not found in dataset records. Available: {', '.join(map(str, records[0]))}")

    items = []
    seen = set()
    for i, r in enumerate(records):
        if r.get(id_field) is None or r.get(text_field) is None:
            raise DataError(f"record {i} lacks '{id_field}' or '{text_field}'")
        iid = str(r[id_field])
        if iid in seen:
            raise DataError(f"duplicate id '{iid}' in dataset")
        seen.add(iid)
        item = dict(r)
        item["item_id"] = iid
        item["text"] = str(r[text_field])
        items.append(item)

    pool = items
    if split_value is not None and split_field:
        pool = [r for r in items if str(r.get(split_field)) == str(split_value)]
        if not pool:
            log(f"Warning: no records with {split_field} == {split_value!r}. Using all {len(items)} records.")
            pool = items

    pool = sorted(pool, key=_id_sort_key(pool))
    if sample_size is None:
        return pool
    if len(pool) < sample_size:
        log(f"Warning: Requested sample size {sample_size} is larger than test split size {len(pool)}. "
            f"Using all available test items.")
        return pool
    if sampling == "first_n_by_id":
        return pool[:sample_size]
    if sampling == "random":
        chosen = random.Random(sampling_seed).sample(pool, sample_size)
        return sorted(chosen, key=_id_sort_key(chosen))
    raise DataError(f"unknown sampling '{sampling}' (use first_n_by_id or random)")


def _id_sort_key(items: list[dict]) -> Callable[[dict], object]:
    """Ids sort as numbers when every id is a whole number (so 2 comes before 10), else as text."""
    if items and all(_INT_ID.fullmatch(it["item_id"]) for it in items):
        return lambda x: int(x["item_id"])
    return lambda x: x["item_id"]


def sample_fingerprint(items: list[dict]) -> str:
    """sha256 over the (id, text) pairs of the sample, in sample order."""
    h = hashlib.sha256()
    for it in items:
        h.update(it["item_id"].encode("utf-8") + b"\x1f" + it["text"].encode("utf-8") + b"\x1e")
    return h.hexdigest()
