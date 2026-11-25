"""Dataset preparation helpers.

- aggregate_annotations: one-row-per-annotation tables -> one record per item with
  `labels_per_annotator` and `label_majority` (a label is in the majority if at least
  `threshold` of the item's annotators assigned it. An empty majority is a valid outcome).
- add_lexical_overlap: flag items whose text contains any word from a codebook field
  (case-insensitive, word boundaries).
- strip_run_text: remove `text` and `raw_response` from run records and blank verbatim evidence
  quotes (for redistribution).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from .data import DataError
from .parsing import NOTE_SEPARATOR


def stable_item_id(text: str, prefix: str = "item_") -> str:
    """Deterministic id from the text: prefix + 12 digits derived from its sha256."""
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"{prefix}{int(h, 16) % (10**12):012d}"


def parse_labels(value, *, sep: str = ",", label_map: dict[str, str] | None = None,
                 drop: set[str] = frozenset()) -> list[str]:
    if value is None:
        return []
    raw_labels = value if isinstance(value, list) else [s.strip() for s in str(value).split(sep)]
    out = []
    for raw in raw_labels:
        if raw == "":
            continue
        if label_map is not None:
            if raw in label_map:
                norm = label_map[raw]
            else:
                print(f"  WARNING: Unknown annotation label: '{raw}'", file=sys.stderr)
                norm = f"UNKNOWN:{raw}"
        else:
            norm = raw
        if norm not in drop:
            out.append(norm)
    return sorted(set(out))


def aggregate_annotations(rows: list[dict], *, text_field: str = "text", annotator_field: str = "annotator",
                          labels_field: str = "annotation", group_field: str | None = None,
                          label_sep: str = ",", label_map: dict[str, str] | None = None,
                          drop_labels: set[str] = frozenset(), threshold: float = 0.5,
                          id_prefix: str = "item_", keep_fields: list[str] = ()) -> list[dict]:
    """Group per-annotator rows into items (sorted by item_id)."""
    if rows:
        needed = [f for f in (group_field, text_field, annotator_field) if f]
        missing = [f for f in needed if f not in rows[0]]
        if missing:
            raise DataError(f"column(s) not found: {', '.join(missing)}. Available: {', '.join(map(str, rows[0]))}")
    groups: dict[str, list[dict]] = defaultdict(list)
    for i, row in enumerate(rows):
        try:
            groups[str(row[group_field]) if group_field else row[text_field]].append(row)
        except KeyError as e:
            raise DataError(f"row {i + 1} has no '{e.args[0]}' value") from None

    records = []
    for key, grp in groups.items():
        text = grp[0][text_field]
        per_annotator = {}
        for row in grp:
            per_annotator[str(row[annotator_field])] = parse_labels(
                row.get(labels_field), sep=label_sep, label_map=label_map, drop=drop_labels)
        n = len(per_annotator)
        votes = Counter(lab for labs in per_annotator.values() for lab in labs)
        majority = sorted(c for c, k in votes.items() if k >= n * threshold)
        rec = {
            "item_id": stable_item_id(key if group_field else text, id_prefix),
            "text": text,
            "labels_per_annotator": per_annotator,
            "label_majority": majority,
        }
        for f in keep_fields:
            rec[f] = grp[0].get(f)
        records.append(rec)
    ids = [r["item_id"] for r in records]
    if len(set(ids)) != len(ids):
        raise DataError("item id collision (two different texts got the same id). Use --group-field with unique ids")
    return sorted(records, key=lambda r: r["item_id"])


def codebook_words(codebook_data: dict, fld: str = "example_words") -> list[str]:
    words = []
    for cat in codebook_data.get("categories", {}).values():
        v = cat.get(fld)
        if isinstance(v, dict) and "words" in v:
            v = v["words"]
        if isinstance(v, list):
            words.extend(str(w) for w in v)
    return words


def add_lexical_overlap(records: list[dict], words: list[str], *, text_field: str = "text",
                        out_field: str = "lexical_overlap") -> int:
    """Set records[i][out_field] = True if the text contains any of `words`. Returns the count."""
    if not words:
        raise DataError("no words found for lexical overlap. Check that the codebook field holds word lists")
    if records and text_field not in records[0]:
        raise DataError(f"field '{text_field}' not found. Available: {', '.join(map(str, records[0]))}")
    pattern = re.compile(r"\b(" + "|".join(re.escape(t) for t in words) + r")\b", re.IGNORECASE)
    n = 0
    for i, r in enumerate(records):
        if text_field not in r:
            raise DataError(f"record {i + 1} has no '{text_field}' field")
        r[out_field] = bool(pattern.search(str(r[text_field])))
        n += r[out_field]
    return n


def write_jsonl(records: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _blank_quotes(evidence) -> None:
    """Evidence quotes are copied from the text. Keep the verified flags, drop the quote strings."""
    if isinstance(evidence, dict):
        for quotes in evidence.values():
            for q in quotes if isinstance(quotes, list) else []:
                if isinstance(q, dict):
                    q["quote"] = None


RAW_EXCERPT_MARKER = "Raw response: "
VERIFY_SKIPPED_SUFFIX = "verify pass skipped (first pass failed)"


def redact_raw_excerpt(notes):
    """Remove the 'Raw response: <excerpt>' part that parse_notes carry for unreadable answers."""
    if not isinstance(notes, str) or RAW_EXCERPT_MARKER not in notes:
        return notes
    head, _, tail = notes.partition(RAW_EXCERPT_MARKER)
    suffix = NOTE_SEPARATOR + VERIFY_SKIPPED_SUFFIX
    suffix = suffix if tail.endswith(suffix) else ""
    return f"{head}{RAW_EXCERPT_MARKER}[removed]{suffix}"


def _strip_record(r: dict) -> dict:
    if "unparseable_output_chunk" in r:
        r["unparseable_output_chunk"] = None
    r.pop("text", None)
    r.pop("raw_response", None)
    if "parse_notes" in r:
        r["parse_notes"] = redact_raw_excerpt(r["parse_notes"])
    _blank_quotes(r.get("evidence"))
    fp = r.get("first_pass")
    if isinstance(fp, dict):
        fp.pop("raw_response", None)
        if "parse_notes" in fp:
            fp["parse_notes"] = redact_raw_excerpt(fp["parse_notes"])
        _blank_quotes(fp.get("evidence"))
    return r


def strip_run_text(in_path: str | Path, out_path: str | Path) -> int:
    """Write a copy of a run (or attempts) file without text, raw answers and evidence quotes.

    The output must be a different file. It is written to a temporary file first and then moved
    into place, so an interrupted call never leaves a half-written file.
    """
    in_path, out_path = Path(in_path), Path(out_path)
    if out_path.resolve() == in_path.resolve():
        raise DataError(f"{out_path} is the input file itself. Choose an --out-dir different from the run's folder")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    fd, tmp = tempfile.mkstemp(dir=out_path.parent, prefix=f".{out_path.name}.", suffix=".tmp")
    try:
        with open(in_path, encoding="utf-8") as fin, os.fdopen(fd, "w", encoding="utf-8") as fout:
            for lineno, line in enumerate(fin, 1):
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError as e:
                    raise DataError(f"{in_path}:{lineno}: not a JSON record ({e.msg})") from None
                if not isinstance(r, dict):
                    raise DataError(f"{in_path}:{lineno}: not a JSON object")
                fout.write(json.dumps(_strip_record(r), ensure_ascii=False) + "\n")
                n += 1
        os.replace(tmp, out_path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return n
