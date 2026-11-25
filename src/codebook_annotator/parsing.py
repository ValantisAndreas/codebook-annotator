"""Parse an LLM response into per-category predictions.

Parsing rules:
  - strip a leading ```json and a trailing ``` fence, then parse JSON (simplejson).
  - expect an object with "predictions" (and "reasoning").
  - fall back to the keys "predicting", "prediction", "categories", "results", "result",
    or to the root object if it contains category keys. This is recorded as a FORMAT WARNING.
  - parse_status is "ok" (all categories present), "partial" (some missing, counted as absent)
    or "failed" (no usable predictions).
  - non-boolean values (e.g. the string "false") are read by meaning and recorded as a
    FORMAT WARNING. `truthiness_coercion=True` uses Python truthiness instead, without a note.
  - ordinal output (scores) and evidence quotes are parsed when requested (optional features).

Notes in parse_notes are joined with NOTE_SEPARATOR.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

import simplejson

FALLBACK_KEYS = ["predicting", "prediction", "categories", "results", "result"]
NOTE_SEPARATOR = "\x3b "  # separator between notes in parse_notes (a semicolon and a space)
_TRUE_STRINGS = {"true", "yes", "1", "present"}
_FALSE_STRINGS = {"false", "no", "0", "", "null", "none", "absent", "n/a", "na"}
_NULL_STRINGS = {"", "null", "none", "n/a", "na", "absent", "false"}


@dataclass
class ParseResult:
    predictions: dict[str, bool]
    parse_status: str
    parse_notes: str | None
    reasoning: Any
    used_fallback: bool = False
    fallback_key: str | None = None
    scores: dict[str, int | None] | None = None
    evidence: dict[str, list[dict]] | None = None
    format_warnings: int = 0  # non-fallback format warnings (coercions)

    def as_legacy_tuple(self):
        return (self.predictions, self.parse_status, self.parse_notes, self.reasoning,
                self.used_fallback, self.fallback_key)


def _coerce_bool(cat: str, val: Any, notes: list[str], truthiness: bool) -> bool:
    if truthiness or isinstance(val, bool):
        return bool(val)
    if isinstance(val, str):
        s = val.strip().lower()
        out = True if s in _TRUE_STRINGS else False if s in _FALSE_STRINGS else bool(val)
    else:
        out = bool(val)
    notes.append(f"FORMAT WARNING: non-boolean value for '{cat}' ({val!r}) read as {str(out).lower()}")
    return out


def _coerce_score(cat: str, val: Any, lo: int, hi: int, notes: list[str]) -> tuple[int | None, bool]:
    """Return (score or None, valid)."""
    if val is None or val is False:
        return None, True
    if isinstance(val, str):
        s = val.strip().lower()
        if s in _NULL_STRINGS:
            return None, True
        try:
            num = float(s)
        except ValueError:
            notes.append(f"Invalid score for '{cat}': {val!r}")
            return None, False
        notes.append(f"FORMAT WARNING: string score for '{cat}' ({val!r})")
        val = num
    if isinstance(val, bool) or not isinstance(val, int | float):
        notes.append(f"Invalid score for '{cat}': {val!r}")
        return None, False
    if isinstance(val, float) and not math.isfinite(val):
        notes.append(f"Invalid score for '{cat}': {val!r} (not a finite number)")
        return None, False
    if isinstance(val, float) and not val.is_integer():
        notes.append(f"Invalid score for '{cat}': {val!r} (not an integer)")
        return None, False
    score = int(val)
    if not lo <= score <= hi:
        shown = str(score) if len(str(score)) <= 20 else str(score)[:17] + "..."
        notes.append(f"Invalid score for '{cat}': {shown} (outside {lo}-{hi})")
        return None, False
    return score, True


_QUOTE_STRIP = " \t\r\n\"'“”‘’«»…"


def normalize_for_match(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.casefold()


def verify_quote(quote: str, text: str | None) -> bool:
    """True if the quote occurs in the text (whitespace-collapsed, case-insensitive, Unicode NFC).

    Surrounding quotation marks and ellipses are ignored. Anything else must match exactly.
    """
    if text is None or not isinstance(quote, str):
        return False
    q = quote.strip(_QUOTE_STRIP)
    q = re.sub(r"^(\.\.\.|…)+|(\.\.\.|…)+$", "", q).strip(_QUOTE_STRIP)
    if not q:
        return False
    return normalize_for_match(q) in normalize_for_match(text)


def _parse_evidence(data: dict, expected: list[str], text: str | None, notes: list[str]) -> dict[str, list[dict]]:
    ev = data.get("evidence")
    if ev is None:
        notes.append("FORMAT WARNING: missing 'evidence' key")
        return {}
    if not isinstance(ev, dict):
        notes.append(f"FORMAT WARNING: 'evidence' is not an object ({type(ev).__name__})")
        return {}
    out: dict[str, list[dict]] = {}
    unknown = [k for k in ev if k not in expected]
    if unknown:
        notes.append(f"FORMAT WARNING: evidence for unknown categories ignored: {', '.join(map(str, unknown))}")
    for cat in expected:
        quotes = ev.get(cat)
        if quotes is None:
            continue
        if isinstance(quotes, str):
            quotes = [quotes]
        if not isinstance(quotes, list):
            notes.append(f"FORMAT WARNING: evidence for '{cat}' is not a list")
            continue
        items = [{"quote": q, "verified": verify_quote(q, text)} for q in quotes if isinstance(q, str) and q.strip()]
        if items:
            out[cat] = items
    return out


def parse_response(raw_response: str, expected_categories: list[str], *, output_type: str = "binary",
                   scale: tuple[int, int] = (1, 10), evidence: bool = False, text: str | None = None,
                   truthiness_coercion: bool = False, include_raw_excerpt: bool = True) -> ParseResult:
    """Parse a model response. See module docstring.

    With include_raw_excerpt=False, the note for an unreadable answer leaves out the excerpt of the
    answer (use this when raw answers must not be stored).
    """
    predictions = {cat: False for cat in expected_categories}
    ordinal = output_type == "ordinal"
    scores = {cat: None for cat in expected_categories} if ordinal else None
    ev_out = {} if evidence else None

    def fail(notes: str | None, reasoning=None, used_fallback=False, fallback_key=None) -> ParseResult:
        return ParseResult(predictions, "failed", notes, reasoning, used_fallback, fallback_key, scores, ev_out)

    if not raw_response:
        return fail("Empty response from model")

    try:
        cleaned = raw_response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        data = simplejson.loads(cleaned)
    except Exception as e:
        if include_raw_excerpt:
            return fail(f"JSON parse error: {e}. Raw response: {raw_response[:200]}")
        return fail(f"JSON parse error: {e}. Raw response not stored")

    if not isinstance(data, dict):
        return fail(f"Response is JSON but not an object. Type: {type(data)}")

    preds_data = data.get("predictions")
    reasoning = data.get("reasoning", "")
    used_fallback = False
    fallback_key = None

    if preds_data is None:
        for alt_key in FALLBACK_KEYS:
            if alt_key in data and isinstance(data[alt_key], dict):
                preds_data = data[alt_key]
                used_fallback = True
                fallback_key = alt_key
                break
    if preds_data is None and any(cat in data for cat in expected_categories):
        preds_data = data
        used_fallback = True
        fallback_key = "root_object"

    if preds_data is None:
        return fail("Missing 'predictions' key in JSON object", reasoning)
    if not isinstance(preds_data, dict):
        return fail(f"'predictions' key exists but is not an object. Type: {type(preds_data)}",
                    reasoning, used_fallback, fallback_key)

    missing_keys = [cat for cat in expected_categories if cat not in preds_data]
    value_notes: list[str] = []
    invalid = []
    for cat in expected_categories:
        val = preds_data.get(cat, False if not ordinal else None)
        if ordinal:
            if cat in missing_keys:
                continue
            score, valid = _coerce_score(cat, val, scale[0], scale[1], value_notes)
            scores[cat] = score
            predictions[cat] = score is not None
            if not valid:
                invalid.append(cat)
        else:
            predictions[cat] = _coerce_bool(cat, val, value_notes, truthiness_coercion) if cat not in missing_keys \
                else bool(val)

    note_parts = []
    if missing_keys:
        parse_status = "partial"
        note_parts.append(f"Missing category predictions in JSON: {', '.join(missing_keys)}")
    elif invalid:
        parse_status = "partial"
    else:
        parse_status = "ok"
    note_parts.extend(value_notes)

    if used_fallback:
        note_parts.append(f"FORMAT WARNING: Fallback key used: '{fallback_key}' (expected 'predictions')")

    if evidence:
        ev_out = _parse_evidence(data, expected_categories, text, note_parts)

    format_warnings = sum(1 for n in note_parts if n.startswith("FORMAT WARNING") and "Fallback key" not in n)
    parse_notes = NOTE_SEPARATOR.join(note_parts) if note_parts else None
    return ParseResult(predictions, parse_status, parse_notes, reasoning, used_fallback, fallback_key,
                       scores, ev_out, format_warnings)


def parse_llm_response(raw_response: str, expected_categories: list[str]):
    """Binary parsing with Python-truthiness coercion of non-boolean values.

    Returns (predictions, parse_status, parse_notes, reasoning, used_fallback, fallback_key).
    """
    return parse_response(raw_response, expected_categories, truthiness_coercion=True).as_legacy_tuple()
