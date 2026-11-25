"""Regression tests for evaluation input handling and consistency fixes."""

import csv
import json

import numpy as np
import pandas as pd
import pytest

from codebook_annotator.evaluate.bootstrap import pval, run_bootstrap, save_csv
from codebook_annotator.evaluate.cli import main as eval_main
from codebook_annotator.evaluate.io import EvalDataError, load_jsonl_indexed, read_jsonl_records, sort_ids
from codebook_annotator.evaluate.report import write_report

CATS = ["A", "B"]


def _jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _run(path, preds_by_id, codebook_id="cb", model="m", text_ids=True):
    rows = []
    for iid, labs in preds_by_id.items():
        rows.append({"item_id": str(iid) if text_ids else iid,
                     "run": {"codebook_id": codebook_id, "model_tag": model, "provider": "stub",
                             "prompt_condition": "zero_shot"},
                     "parse_status": "ok", "predictions": {c: c in labs for c in CATS}, "reasoning": ""})
    return _jsonl(path, rows)


def _gold(path, ids, ann=True):
    rows = []
    for i in ids:
        r = {"item_id": i, "label_majority": ["A"] if int(str(i)[-1]) % 2 else []}
        if ann:
            r["labels_per_annotator"] = {"a1": r["label_majority"], "a2": ["A"]}
        rows.append(r)
    return _jsonl(path, rows)


# ── Numeric ids ───────────────────────────────────────────────────────────────

def test_numeric_gold_ids_match_text_run_ids(tmp_path):
    gold = _gold(tmp_path / "g.jsonl", [2, 10, 9, 11, 13, 15])
    run = _run(tmp_path / "r.jsonl", {i: ["A"] for i in [2, 10, 9, 11, 13, 15]})
    out = tmp_path / "w.xlsx"
    assert eval_main(["evaluate", "--dataset", str(gold), "--runs", str(run), "-o", str(out)]) == 0
    s1 = pd.read_excel(out, sheet_name="Sheet1")
    assert s1.loc[s1.category == "ALL", "n_items"].iloc[0] == 6
    res = run_bootstrap(str(gold), [str(run)], resamples=20, categories=CATS)
    assert res.n_items == 6


def test_numeric_ids_sort_numerically_in_disagreement_csv(tmp_path):
    ids = [2, 10, 9]
    gold = _gold(tmp_path / "g.jsonl", ids)
    r1 = _run(tmp_path / "r1.jsonl", {i: ["A"] for i in ids}, codebook_id="x")
    r2 = _run(tmp_path / "r2.jsonl", {i: ["B"] for i in ids}, codebook_id="y")
    out = tmp_path / "d.csv"
    assert eval_main(["disagreements", "--dataset", str(gold), "--runs", str(r1), str(r2), "-o", str(out)]) == 0
    assert [r["item_id"] for r in csv.DictReader(open(out))] == ["2", "9", "10"]
    assert sort_ids(["10", "2", "9"]) == ["2", "9", "10"]
    assert sort_ids(["b", "a10", "a2"]) == ["a10", "a2", "b"]  # text ids keep text order


# ── Robust reading ────────────────────────────────────────────────────────────

def test_truncated_line_does_not_hide_later_records(tmp_path, capsys):
    lines = [json.dumps({"item_id": f"g{i}", "predictions": {"A": True}}) for i in range(12)]
    lines[3] = lines[3][:25]
    p = tmp_path / "r.jsonl"
    p.write_text("\n".join(lines) + "\n")
    recs = load_jsonl_indexed(p)
    assert len(recs) == 11 and "g3" not in recs
    assert "1 unreadable line(s) or fragment(s) skipped" in capsys.readouterr().out


def test_multiline_records_and_junk_and_duplicates(tmp_path, capsys):
    pretty = json.dumps({"item_id": "m1", "predictions": {"A": True}}, indent=2)
    p = tmp_path / "r.jsonl"
    p.write_text(pretty + "\n" + "[1, 2]\n" + json.dumps({"x": 1}) + "\n"
                 + json.dumps({"item_id": "m2"}) + "\n" + json.dumps({"item_id": "m2", "v": 2}) + "\n")
    recs = load_jsonl_indexed(p)
    assert set(recs) == {"m1", "m2"} and recs["m2"]["v"] == 2
    out = capsys.readouterr().out
    assert "2 record(s) without the field 'item_id' skipped" in out
    assert "1 duplicate id(s), the last record of each was kept" in out


def test_read_missing_file_raises_clean_error(tmp_path):
    with pytest.raises(EvalDataError, match="file not found"):
        read_jsonl_records(tmp_path / "nope.jsonl")


# ── Agreement sheets use gold items only ─────────────────────────────────────

def test_agreement_sheets_use_only_gold_items(tmp_path):
    gold_ids = [f"g{i}" for i in range(6)]
    extra = [f"x{i}" for i in range(10)]
    gold = _gold(tmp_path / "g.jsonl", [int(i[1:]) for i in gold_ids])
    gold.write_text(gold.read_text().replace('"item_id": ', '"item_id": "g').replace(', "label_majority"', '", "label_majority"'))
    p1 = {i: ["A"] for i in gold_ids} | {i: ["A"] for i in extra}
    p2 = {i: ["A"] for i in gold_ids} | {i: ["B"] for i in extra}
    r1 = _run(tmp_path / "r1.jsonl", p1, codebook_id="cb", model="m1")
    r2 = _run(tmp_path / "r2.jsonl", p2, codebook_id="cb", model="m2")
    r3 = _run(tmp_path / "r3.jsonl", p2, codebook_id="other", model="m1")
    out = tmp_path / "w.xlsx"
    assert eval_main(["evaluate", "--dataset", str(gold), "--runs", str(r1), str(r2), str(r3), "-o", str(out)]) == 0
    sheets = pd.read_excel(out, sheet_name=None)
    assert sheets["Model Agreement"]["items_compared"].tolist() == [6]
    ca = sheets["Codebook Agreement"]
    assert ca.loc[ca.slice_type == "overall", "items_compared"].tolist() == [6]


# ── Disagreements ────────────────────────────────────────────────────────────

def test_disagreements_warns_on_collapsing_runs_and_counts_gold_items(tmp_path, capsys):
    gold = _gold(tmp_path / "g.jsonl", [1, 2, 3])
    r1 = _run(tmp_path / "r1.jsonl", {1: ["A"], 2: ["A"], 3: [], 99: ["A"]}, codebook_id="same")
    r2 = _run(tmp_path / "r2.jsonl", {1: ["B"], 2: ["A"], 3: [], 99: ["B"]}, codebook_id="same")
    assert eval_main(["disagreements", "--dataset", str(gold), "--runs", str(r1), str(r2),
                      "-o", str(tmp_path / "d.csv")]) == 0
    out = capsys.readouterr().out
    assert "2 runs share the column key 'same'" in out and "--key run_id" in out
    assert "Total items checked (in the gold file and in every run): 3" in out


def test_disagreements_categories_restrict_predictions(tmp_path):
    gold = _gold(tmp_path / "g.jsonl", [1, 2])
    r1 = _run(tmp_path / "r1.jsonl", {1: ["A", "B"], 2: ["B"]}, codebook_id="x")
    r2 = _run(tmp_path / "r2.jsonl", {1: ["A"], 2: []}, codebook_id="y")
    out = tmp_path / "d.csv"
    assert eval_main(["disagreements", "--dataset", str(gold), "--runs", str(r1), str(r2), "-o", str(out),
                      "--categories", "A"]) == 0
    rows = list(csv.DictReader(open(out)))
    assert all("B" not in r["pred_x"] for r in rows)
    assert [r["item_id"] for r in rows if r["pred_x"] != r["pred_y"]] == []  # B differences are out of scope


def test_annotator_csv_quotes_commas(tmp_path):
    gold = _jsonl(tmp_path / "g.jsonl", [{"item_id": "a,b", "label_majority": [], "labels_per_annotator": {"x": ["A"], "y": []}}])
    out = tmp_path / "a.csv"
    assert eval_main(["annotator-disagreements", "--dataset", str(gold), "-o", str(out), "--split", "all"]) == 0
    rows = list(csv.DictReader(line for line in open(out) if not line.startswith("#")))
    assert rows[0]["item_id"] == "a,b" and rows[0]["disagree_A"] == "True"


# ── Bootstrap ─────────────────────────────────────────────────────────────────

def test_bootstrap_ignores_gold_labels_outside_categories(tmp_path):
    rows = [{"item_id": f"i{k}", "label_majority": ["A", "Other"], "labels_per_annotator": {"a": ["A", "Other"]}}
            for k in range(6)]
    gold = _jsonl(tmp_path / "g.jsonl", rows)
    run = _run(tmp_path / "r.jsonl", {f"i{k}": ["A"] for k in range(6)})
    res = run_bootstrap(str(gold), [str(run)], resamples=10, categories=CATS)
    pt = dict(zip(res.names, res.point[res.arms[0]], strict=True))
    assert pt["Jaccard"] == 1.0 and pt["coverage%"] == 100.0  # "Other" is not an evaluated category


def test_bootstrap_leaves_unannotated_items_out_of_oor(tmp_path):
    rows = [{"item_id": "i0", "label_majority": [], "labels_per_annotator": {"a": []}},
            {"item_id": "i1", "label_majority": [], "labels_per_annotator": {}},
            {"item_id": "i2", "label_majority": []}]
    gold = _jsonl(tmp_path / "g.jsonl", rows)
    run = _run(tmp_path / "r.jsonl", {"i0": [], "i1": ["A"], "i2": ["A"]})
    res = run_bootstrap(str(gold), [str(run)], resamples=10, categories=CATS)
    pt = dict(zip(res.names, res.point[res.arms[0]], strict=True))
    assert pt["OOR%"] == 0.0  # only i0 is annotated, and it has no out-of-range label
    assert any("2 item(s) have no annotator labels" in ln for ln in res.lines)


def test_bootstrap_errors(tmp_path):
    gold = _gold(tmp_path / "g.jsonl", [1, 3])
    other = _run(tmp_path / "r.jsonl", {7: ["A"]})
    with pytest.raises(EvalDataError, match="no item is present"):
        run_bootstrap(str(gold), [str(other)], resamples=5, categories=CATS)
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    ra = _run(tmp_path / "a" / "run.jsonl", {1: ["A"], 3: ["A"]})
    rb = _run(tmp_path / "b" / "run.jsonl", {1: [], 3: []})
    with pytest.raises(EvalDataError, match="same name 'run'"):
        run_bootstrap(str(gold), [str(ra), str(rb)], resamples=5, categories=CATS)
    assert eval_main(["bootstrap", "--dataset", str(gold), "--runs", str(other)]) == 2
    assert np.isnan(pval(np.array([np.nan, np.nan]), float("nan")))


def test_bootstrap_csv_applies_p_floor(tmp_path):
    ids = [f"i{k}" for k in range(30)]
    rows = [{"item_id": i, "label_majority": ["A"], "labels_per_annotator": {"a": ["A"]}} for i in ids]
    gold = _jsonl(tmp_path / "g.jsonl", rows)
    good = _run(tmp_path / "good.jsonl", {i: ["A"] for i in ids})
    bad = _run(tmp_path / "bad.jsonl", {i: ["B"] for i in ids})
    res = run_bootstrap(str(gold), [str(good), str(bad)], resamples=50, categories=CATS)
    out = tmp_path / "b.csv"
    save_csv(res, str(out))
    rows = [r for r in csv.DictReader(open(out)) if r["arm"] == "bad"]
    floored = [r for r in rows if r["p_at_floor"] == "True"]
    assert floored and all(float(r["p_value"]) == pytest.approx(1 / 50) for r in floored)


# ── Clean errors and merge ───────────────────────────────────────────────────

def test_evaluate_commands_give_clean_errors(tmp_path, capsys):
    no_majority = _jsonl(tmp_path / "g.jsonl", [{"item_id": "a", "labels": []}])
    run = _run(tmp_path / "r.jsonl", {"a": ["A"]})
    empty_run = tmp_path / "empty.jsonl"
    empty_run.write_text("")
    assert eval_main(["evaluate", "--dataset", str(tmp_path / "missing.jsonl"), "--runs", str(run)]) != 0
    assert eval_main(["evaluate", "--dataset", str(no_majority), "--runs", str(run)]) == 2
    assert eval_main(["bootstrap", "--dataset", str(tmp_path / "missing.jsonl"), "--runs", str(run)]) == 2
    assert eval_main(["disagreements", "--dataset", str(no_majority), "--runs", str(run), "-o", str(tmp_path / "d.csv")]) == 2
    gold = _gold(tmp_path / "g2.jsonl", [1])
    assert eval_main(["evaluate", "--dataset", str(gold), "--runs", str(empty_run), "-o", str(tmp_path / "w.xlsx")]) == 1
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err and "majority field 'label_majority'" in captured.err
    assert "no readable records in" in captured.out


def test_merge_keeps_literal_na_strings(tmp_path):
    out = tmp_path / "wb.xlsx"
    first = {"Sheet1": pd.DataFrame({"run_id": ["r1"], "model_to_model_agreement_alpha": ["NA"], "region": ["NA"]})}
    write_report(first, out, merge=False, evaluated_run_ids={"r1"})
    second = {"Sheet1": pd.DataFrame({"run_id": ["r2"], "model_to_model_agreement_alpha": [0.5], "region": ["EU"]})}
    merged = write_report(second, out, merge=True, evaluated_run_ids={"r2"})
    kept = merged["Sheet1"][merged["Sheet1"]["run_id"] == "r1"].iloc[0]
    assert kept["model_to_model_agreement_alpha"] == "NA" and kept["region"] == "NA"
