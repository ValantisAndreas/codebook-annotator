"""End-to-end evaluation on a tiny synthetic gold set and runs (values hand-computed).

Gold (categories A, B, C), two annotators per item:
  i1: a1=[A], a2=[A, B]   majority [A, B]   union {A, B}
  i2: a1=[],  a2=[C]      majority []       union {C}
  i3: a1=[B], a2=[B]      majority [B]      union {B}
Run predictions: i1 {A}, i2 {A}, i3 {B, C}
"""

import json

import pandas as pd
import pytest

from codebook_annotator.evaluate.bootstrap import run_bootstrap
from codebook_annotator.evaluate.cli import main as eval_main
from codebook_annotator.evaluate.disagreements import annotator_disagreement_csv, run_disagreement_csv

GOLD = [
    {"item_id": "i1", "text": "first text", "labels_per_annotator": {"a1": ["A"], "a2": ["A", "B"]},
     "label_majority": ["A", "B"], "region": "north", "flag": True, "split": "test",
     "gold_scores": {"A": 8, "B": 3, "C": None}},
    {"item_id": "i2", "text": "second text", "labels_per_annotator": {"a1": [], "a2": ["C"]},
     "label_majority": [], "region": "south", "flag": False, "split": "test",
     "gold_scores": {"A": None, "B": None, "C": 5}},
    {"item_id": "i3", "text": "third text", "labels_per_annotator": {"a1": ["B"], "a2": ["B"]},
     "label_majority": ["B"], "region": "north", "split": "train",
     "gold_scores": {"A": None, "B": 6, "C": None}},
]
PREDS = {"i1": {"A"}, "i2": {"A"}, "i3": {"B", "C"}}
CATS = ["A", "B", "C"]


def v1_record(iid, pred_set, codebook_id="cb_x", model="m1"):
    return {
        "item_id": iid, "text": "t",
        "run": {"dataset": "toy", "dataset_revision": None, "codebook_id": codebook_id, "codebook_version": "1.0",
                "model_tag": model, "model_digest": None, "provider": "ollama", "prompt_condition": "zero_shot",
                "sample_id": "s", "context_window": 1024, "temperature": 0.0, "seed": None,
                "timestamp": "2026-01-01T00:00:00Z"},
        "parse_status": "ok", "parse_notes": None,
        "predictions": {c: c in pred_set for c in CATS},
        "raw_response": "{}", "reasoning": f"r-{iid}",
    }


def v2_record(iid, pred_set, **kw):
    rec = v1_record(iid, pred_set, **kw)
    rec.pop("text")
    rec.pop("raw_response")
    rec["schema_version"] = 2
    rec["usage"] = {"input_tokens": 10, "output_tokens": 5, "latency_ms": 12.5}
    rec["run"].update(tool_version="0.1.0", codebook_sha256="a" * 64, system_prompt_sha256="b" * 64,
                      config_sha256="c" * 64, sample_sha256="d" * 64, prompt_template="default_v1")
    return rec


def write_jsonl(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return str(path)


@pytest.fixture
def gold_path(tmp_path):
    return write_jsonl(tmp_path / "gold.jsonl", GOLD)


@pytest.fixture
def run_v1(tmp_path):
    return write_jsonl(tmp_path / "run_v1.jsonl", [v1_record(i, p) for i, p in PREDS.items()])


@pytest.fixture
def run_v2(tmp_path):
    return write_jsonl(tmp_path / "run_v2.jsonl", [v2_record(i, p) for i, p in PREDS.items()])


def evaluate(tmp_path, gold, runs, *extra, name="out.xlsx"):
    out = tmp_path / name
    assert eval_main(["evaluate", "--dataset", gold, "--runs", *runs, "-o", str(out), *extra]) == 0
    return {s: pd.read_excel(out, sheet_name=s) for s in pd.ExcelFile(out).sheet_names}


def test_sheet1_hand_computed(tmp_path, gold_path, run_v1):
    sheets = evaluate(tmp_path, gold_path, [run_v1], "--min-cell-floor", "1")
    s1 = sheets["Sheet1"]
    assert list(s1["category"]) == ["ALL", "A", "B", "C"]
    row = s1[s1.category == "ALL"].iloc[0]
    assert row.n_items == 3
    assert row.micro_f1 == pytest.approx(4 / 7)
    assert row.macro_f1 == pytest.approx(4 / 9)  # A 2/3, B 2/3, C 0
    assert row.jaccard == pytest.approx(1 / 3)
    assert row.hamming_loss == pytest.approx(1 / 3)
    # alpha for A: ratings [1,1,1],[0,0,1],[0,0,0] -> 0.6 (see test_eval_metrics), mean over A, B, C
    a_row = s1[s1.category == "A"].iloc[0]
    assert a_row.model_to_annotator_agreement_alpha == pytest.approx(0.6)
    # only one run per codebook -> the string "NA" is written, which pandas reads back as NaN
    assert pd.isna(a_row.model_to_model_agreement_alpha)


def test_min_cell_floor(tmp_path, gold_path, run_v1):
    s1 = evaluate(tmp_path, gold_path, [run_v1])["Sheet1"]  # default floor 5 > 3 items
    assert (s1.n_items == 3).all()
    assert s1.micro_f1.isna().all() and s1.model_to_annotator_agreement_alpha.isna().all()


def test_v1_and_v2_records_give_identical_metrics(tmp_path, gold_path, run_v1, run_v2):
    a = evaluate(tmp_path, gold_path, [run_v1], "--min-cell-floor", "1", name="a.xlsx")
    b = evaluate(tmp_path, gold_path, [run_v2], "--min-cell-floor", "1", name="b.xlsx")
    cols = ["micro_f1", "macro_f1", "jaccard", "hamming_loss", "model_to_annotator_agreement_alpha"]
    pd.testing.assert_frame_equal(a["Sheet1"][cols], b["Sheet1"][cols])


def test_slices_bool_labels_and_category_order(tmp_path, gold_path, run_v1):
    sheets = evaluate(tmp_path, gold_path, [run_v1], "--min-cell-floor", "1",
                      "--slice-by", "region", "--slice-by", "flag", "--bool-labels", "flag=yes,no",
                      "--categories", "C", "A", "B")
    assert list(sheets) == ["Sheet1", "Region Slices", "Flag Slices", "Category Slices"]
    assert list(sheets["Region Slices"]["region"]) == ["north", "south"]
    assert list(sheets["Flag Slices"]["flag"]) == ["no", "yes"]  # i3 has no 'flag' -> counted as false
    assert list(sheets["Category Slices"]["category"]) == ["C", "A", "B"]
    north = sheets["Region Slices"].iloc[0]  # i1, i3: A tp, B tp+fn, C fp
    assert north.micro_f1 == pytest.approx(2 * 2 / (2 * 2 + 1 + 1))


def test_sheet1_slice_field(tmp_path, gold_path, run_v1):
    s1 = evaluate(tmp_path, gold_path, [run_v1], "--min-cell-floor", "1", "--sheet1-slice-field", "region")["Sheet1"]
    assert list(s1[s1.category == "ALL"]["slice"]) == ["pooled", "north", "south"]
    assert s1[(s1.category == "ALL") & (s1.slice == "south")].iloc[0].n_items == 1


def test_missing_annotators_field(tmp_path, run_v1):
    gold = [{k: v for k, v in g.items() if k != "labels_per_annotator"} for g in GOLD]
    gp = write_jsonl(tmp_path / "gold_noann.jsonl", gold)
    s1 = evaluate(tmp_path, gp, [run_v1], "--min-cell-floor", "1")["Sheet1"]
    assert s1.model_to_annotator_agreement_alpha.isna().all()
    assert s1[s1.category == "ALL"].iloc[0].micro_f1 == pytest.approx(4 / 7)
    res = run_bootstrap(gp, [run_v1], resamples=20)
    assert res.names == ["macro-F1", "micro-F1", "Jaccard"]
    assert list(res.point[res.arms[0]]) == pytest.approx([4 / 9, 4 / 7, 1 / 3])


def test_custom_gold_field_names(tmp_path, run_v1):
    gold = [{"id": g["item_id"], "maj": g["label_majority"], "votes": g["labels_per_annotator"]} for g in GOLD]
    gp = write_jsonl(tmp_path / "gold_custom.jsonl", gold)
    s1 = evaluate(tmp_path, gp, [run_v1], "--min-cell-floor", "1",
                  "--id-field", "id", "--majority-field", "maj", "--annotators-field", "votes")["Sheet1"]
    assert s1[s1.category == "A"].iloc[0].model_to_annotator_agreement_alpha == pytest.approx(0.6)


def test_model_and_codebook_agreement(tmp_path, gold_path):
    r1 = write_jsonl(tmp_path / "r1.jsonl", [v1_record(i, p, codebook_id="cb1", model="m1") for i, p in PREDS.items()])
    r2 = write_jsonl(tmp_path / "r2.jsonl", [v1_record(i, p, codebook_id="cb1", model="m2") for i, p in PREDS.items()])
    r3 = write_jsonl(tmp_path / "r3.jsonl", [v1_record(i, set(), codebook_id="cb2", model="m1") for i in PREDS])
    sheets = evaluate(tmp_path, gold_path, [r1, r2, r3], "--min-cell-floor", "1")
    ma = sheets["Model Agreement"]
    assert len(ma) == 1 and ma.iloc[0].comparison == "r1 vs r2" and ma.iloc[0].mean_model_agreement_alpha == 1.0
    ca = sheets["Codebook Agreement"]
    assert set(ca["comparison"]) == {"cb1 vs cb2"}
    assert list(ca["slice_type"]) == ["overall", "category", "category", "category"]
    s1 = sheets["Sheet1"]
    assert s1[(s1.run_id == "r1") & (s1.category == "ALL")].iloc[0].model_to_model_agreement_alpha == 1.0


def test_merge_replaces_only_reevaluated_runs(tmp_path, gold_path, run_v1, run_v2):
    out = tmp_path / "merged.xlsx"
    eval_main(["evaluate", "--dataset", gold_path, "--runs", run_v1, "-o", str(out)])
    eval_main(["evaluate", "--dataset", gold_path, "--runs", run_v2, "-o", str(out), "--merge"])
    eval_main(["evaluate", "--dataset", gold_path, "--runs", run_v2, "-o", str(out), "--merge"])
    s1 = pd.read_excel(out, sheet_name="Sheet1")
    assert s1.groupby("run_id").size().to_dict() == {"run_v1": 4, "run_v2": 4}


def test_ordinal_sheet(tmp_path, gold_path):
    scores = {"i1": {"A": 7, "B": None, "C": None}, "i2": {"A": 2, "B": None, "C": 9}, "i3": {"A": None, "B": 6, "C": 1}}
    recs = []
    for iid, sc in scores.items():
        rec = v2_record(iid, {c for c, s in sc.items() if s is not None})
        rec["run"]["prompt_condition"] = "zero_shot+ordinal"
        rec["scores"] = sc
        recs.append(rec)
    rp = write_jsonl(tmp_path / "ord.jsonl", recs)
    sheets = evaluate(tmp_path, gold_path, [rp], "--gold-scores-field", "gold_scores")
    o = sheets["Ordinal"].set_index("category")
    # A: gold i1=8 (pred 7) -> 1 pair. B: gold i1=3 (pred null), i3=6 (pred 6). C: gold i2=5 (pred 9)
    assert o.loc["A"].n_pairs == 1 and o.loc["A"].mae == 1 and o.loc["A"].acc_within_1 == 1
    assert o.loc["B"].n_pairs == 2 and o.loc["B"].n_pred_null == 1 and o.loc["B"].mae == 0 and o.loc["B"].acc_within_1 == 0.5
    assert o.loc["C"].mae == 4 and o.loc["C"].acc_within_1 == 0
    assert o.loc["ALL"].n_pairs == 4 and o.loc["ALL"].mae == pytest.approx(5 / 3) and o.loc["ALL"].acc_within_1 == pytest.approx(0.5)
    # without the gold scores field the sheet is skipped
    assert "Ordinal" not in evaluate(tmp_path, gold_path, [rp], name="noord.xlsx")


def test_evidence_sheet(tmp_path, gold_path):
    recs = []
    for iid, p in PREDS.items():
        rec = v2_record(iid, p)
        rec["run"]["prompt_condition"] = "zero_shot+evidence"
        rec["evidence"] = {c: [{"quote": "q", "verified": iid != "i2"}] for c in p}
        recs.append(rec)
    rp = write_jsonl(tmp_path / "ev.jsonl", recs)
    e = evaluate(tmp_path, gold_path, [rp])["Evidence"].set_index("category")
    # present: i1 A, i2 A (unverified), i3 B, i3 C
    assert e.loc["A"].n_present == 2 and e.loc["A"].share_present_supported == 0.5
    assert e.loc["ALL"].n_quotes == 4 and e.loc["ALL"].n_verified == 3 and e.loc["ALL"].share_verified == 0.75
    assert "Evidence" not in evaluate(tmp_path, gold_path, [write_jsonl(tmp_path / "plain.jsonl", [v1_record(i, p) for i, p in PREDS.items()])], name="noev.xlsx")


def test_bootstrap_point_estimates_hand_computed(gold_path, run_v1, run_v2):
    res = run_bootstrap(gold_path, [run_v1, run_v2], resamples=50, seed=1)
    p = res.point["run_v1"]
    # OOR: i2 predicts A (no annotator chose it), i3 predicts C -> 2/3 items, 2/3 labels per item
    # coverage: hits i1 {A}, i3 {B} = 2 of 4 annotator-chosen labels
    assert list(p) == pytest.approx([200 / 3, 2 / 3, 50.0, 4 / 9, 4 / 7, 1 / 3])
    assert res.boot["run_v1"].shape == (50, 6)
    assert any("PAIRED DIFFERENCES vs run_v1" in ln for ln in res.lines)
    again = run_bootstrap(gold_path, [run_v1, run_v2], resamples=50, seed=1)
    assert (again.boot["run_v1"] == res.boot["run_v1"]).all()


def test_annotator_disagreement_csv(tmp_path, gold_path):
    out = tmp_path / "ann.csv"
    rows = annotator_disagreement_csv(gold_path, out, "test")
    assert [r["item_id"] for r in rows] == ["i1", "i2"]  # split filter
    assert rows[0]["disagree_B"] is True and rows[0]["disagree_A"] is False and rows[0]["votes_B"] == "0|1"
    lines = out.read_text().splitlines()
    assert lines[3] == "item_id,n_annotators,disagree_A,disagree_B,disagree_C,any_disagreement,votes_A,votes_B,votes_C"
    assert len(annotator_disagreement_csv(gold_path, out, "all")) == 3
    assert len(annotator_disagreement_csv(gold_path, out, "nonexistent")) == 3  # falls back to all


def test_run_disagreement_csv_text_stripped_runs(tmp_path, gold_path, run_v2):
    r_none = write_jsonl(tmp_path / "none.jsonl", [v2_record(i, set(), codebook_id="cb_none") for i in PREDS])
    out = tmp_path / "dis.csv"
    rows = run_disagreement_csv(gold_path, [run_v2, r_none], out, item_fields=["region"])
    by_id = {r["item_id"]: r for r in rows}
    assert set(by_id) == {"i1", "i2", "i3"}
    assert by_id["i2"]["gold_majority"] == "(none)" and by_id["i2"]["pred_cb_none"] == "(none)"
    assert by_id["i1"]["labels_per_annotator"] == "A: 2/2, B: 1/2 | a1: [A]; a2: [A, B]"
    assert by_id["i2"]["labels_per_annotator"] == "C: 1/2 | a1: [none]; a2: [C]"
    header = out.read_text().splitlines()[0]
    assert header == "item_id,text,region,gold_majority,labels_per_annotator,pred_cb_none,pred_cb_x,reasoning_cb_none,reasoning_cb_x"
    rows_run = run_disagreement_csv(gold_path, [run_v2, r_none], out, key="run_id")
    assert "pred_run_v2" in rows_run[0] and "pred_none" in rows_run[0]


def test_merge_keeps_sheets_not_recomputed(tmp_path):
    """A merge that computes fewer sheets keeps the existing workbook's other sheets unchanged."""
    import pandas as pd

    from codebook_annotator.evaluate.report import write_report

    out = tmp_path / "wb.xlsx"
    first = {"Sheet1": pd.DataFrame({"run_id": ["r1"], "x": [1]}),
             "Region Slices": pd.DataFrame({"run_id": ["r1"], "region": ["north"], "x": [1]})}
    write_report(first, out, merge=False, evaluated_run_ids={"r1"})
    second = {"Sheet1": pd.DataFrame({"run_id": ["r2"], "x": [2]})}  # no slice sheet this time
    merged = write_report(second, out, merge=True, evaluated_run_ids={"r2"})
    assert sorted(merged["Sheet1"]["run_id"]) == ["r1", "r2"]
    assert "Region Slices" in pd.read_excel(out, sheet_name=None)
    assert merged["Region Slices"]["run_id"].tolist() == ["r1"]
