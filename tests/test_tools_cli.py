import json
import sys

import pytest
import yaml

from codebook_annotator.cli import main
from codebook_annotator.codebook import CodebookError, load_codebook
from codebook_annotator.codebook_tools import build_ablations, check_invariance, table_to_codebook
from codebook_annotator.data_tools import add_lexical_overlap, aggregate_annotations, stable_item_id

FIX_CB = __import__("pathlib").Path(__file__).parent / "fixtures" / "codebooks"


def _csv(path, text):
    path.write_text(text, encoding="utf-8")
    return path


def test_convert_table(tmp_path):
    p = _csv(tmp_path / "cb.csv", "category,definition,example_words\naesthetic,Looks,\"nice, view\"\nHistoric,Past,old\n")
    cb = table_to_codebook(p, list_columns={"example_words"}, expected_categories=["Aesthetic", "Historic"],
                           labels={"example_words": "Words"})
    assert list(cb["categories"]) == ["Aesthetic", "Historic"]
    assert cb["categories"]["Aesthetic"]["example_words"] == ["nice", "view"]
    assert cb["render"] == [{"field": "definition", "label": "Definition"}, {"field": "example_words", "label": "Words"}]
    with pytest.raises(CodebookError, match="field 'definition' is empty"):
        table_to_codebook(_csv(tmp_path / "e.csv", "category,definition\nA,\n"))
    with pytest.raises(CodebookError, match="missing categories: C"):
        table_to_codebook(p, expected_categories=["Aesthetic", "Historic", "C"])


def test_ablations_and_invariance(tmp_path):
    base = yaml.safe_load((FIX_CB / "toy_plain.yaml").read_text())
    base["prompt"] = {"role": "R"}
    ext = {"Aesthetic": {"ctx": " a ", "who": "x"}, "Historic": {"ctx": "b", "who": "y"}}
    v = build_ablations(base, ext, [("ctx", "Context"), ("who", "Who")], prefix="v", base_name="toy")
    assert sorted(v) == ["v_ctx", "v_full", "v_who"]
    assert v["v_ctx"]["categories"]["Aesthetic"]["ctx"] == "a" and v["v_ctx"]["categories"]["Social"]["ctx"] is None
    assert v["v_full"]["prompt"] == {"role": "R"}  # prompt-level keys carried over
    for k, c in base["categories"].items():
        for f, val in c.items():
            assert v["v_full"]["categories"][k][f] == val
    paths = []
    for name, cb in v.items():
        p = tmp_path / f"{name}.yaml"
        p.write_text(yaml.safe_dump(cb))
        paths.append(p)
    bp = tmp_path / "base.yaml"
    bp.write_text(yaml.safe_dump(base))
    rep = check_invariance([load_codebook(bp)] + [load_codebook(p) for p in paths])
    assert rep.ok
    rep2 = check_invariance([load_codebook(bp), load_codebook(FIX_CB / "toy_plain.yaml")])
    assert not rep2.ok and not rep2.results[0]["preamble_identical"]  # prompt override differs


def test_aggregate_and_overlap():
    rows = [
        {"text": "t1", "annotator": "a", "annotation": "Care, Purity"},
        {"text": "t1", "annotator": "b", "annotation": "Care"},
        {"text": "t1", "annotator": "c", "annotation": "Non-Moral"},
        {"text": "t2", "annotator": "a", "annotation": "Weird"},
    ]
    recs = aggregate_annotations(rows, label_map={"Care": "Care", "Purity": "Purity", "Non-Moral": "Non-Moral"},
                                 drop_labels={"Non-Moral"})
    by_text = {r["text"]: r for r in recs}
    assert by_text["t1"]["label_majority"] == ["Care"]  # 2/3 >= 0.5, Purity 1/3 < 0.5
    assert by_text["t1"]["labels_per_annotator"]["c"] == []
    assert by_text["t2"]["label_majority"] == ["UNKNOWN:Weird"]
    assert by_text["t1"]["item_id"] == stable_item_id("t1")
    recs = [{"text": "A Sacred place"}, {"text": "sacredness"}]
    assert add_lexical_overlap(recs, ["sacred"]) == 1 and recs[0]["lexical_overlap"] is True


def test_cli_end_to_end_stub(tmp_path, base_cfg, capsys):
    cfg = tmp_path / "run.yaml"
    cfg.write_text(yaml.safe_dump(base_cfg))
    assert main(["validate", str(FIX_CB / "toy_plain.yaml")]) == 0
    assert main(["render", str(FIX_CB / "toy_plain.yaml"), "--template", "moral_foundations_v1", "--text", "hi"]) == 0
    assert main(["run", "-c", str(cfg), "--dry-run", "--show-prompt"]) == 0
    assert main(["run", "-c", str(cfg), "--sample-size", "2"]) == 0
    out = tmp_path / "runs" / "t.jsonl"
    assert len(out.read_text().splitlines()) == 2
    assert main(["strip-text", str(out), "--out-dir", str(tmp_path / "shared")]) == 0
    r = json.loads((tmp_path / "shared" / "t.jsonl").read_text().splitlines()[0])
    assert "text" not in r and "raw_response" not in r
    with pytest.raises(SystemExit):
        main(["run", "-c", str(cfg), "--tempreature", "1"])
    with pytest.raises(SystemExit):
        main(["run", "-c", str(cfg), "--provider", "nope"])


def test_dry_run_with_auto_digest_needs_no_server(tmp_path, base_cfg):
    cfg = tmp_path / "run.yaml"
    cfg.write_text(yaml.safe_dump({**base_cfg, "provider": "ollama", "model_tag": "m", "model_digest": "auto",
                                   "ollama_host": "http://127.0.0.1:9"}))
    assert main(["run", "-c", str(cfg), "--dry-run"]) == 0


def test_init_scaffold_runs_end_to_end(tmp_path, monkeypatch):
    target = tmp_path / "proj"
    assert main(["init", str(target)]) == 0
    assert main(["init", str(target)]) == 1  # refuses to overwrite
    assert main(["init", str(target), "--force"]) == 0
    monkeypatch.chdir(target)
    assert main(["validate", "codebook.yaml"]) == 0
    assert main(["run", "-c", "run.yaml"]) == 0
    assert main(["evaluate", "--dataset", "items.jsonl", "--runs", "runs/first_run.jsonl",
                 "-o", "runs/results.xlsx"]) == 0
    import pandas as pd
    sheets = pd.read_excel("runs/results.xlsx", sheet_name=None)
    assert "Usage" in sheets and sheets["Usage"].loc[0, "n_with_usage"] == 6
    assert sheets["Sheet1"]["micro_f1"].notna().all()  # 6 items >= default floor, so metrics are shown


def test_missing_eval_extra_message(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "pandas", None)  # simulate pandas not installed
    for mod in [m for m in sys.modules if m.startswith("codebook_annotator.evaluate.")]:
        monkeypatch.delitem(sys.modules, mod)
    rc = main(["evaluate", "--dataset", "x.jsonl", "--runs", "y.jsonl"])
    assert rc == 2 and 'pip install -e ".[eval]"' in capsys.readouterr().err


def test_ablate_rejects_existing_field():
    base = yaml.safe_load((FIX_CB / "toy_plain.yaml").read_text())
    with pytest.raises(CodebookError, match="already exist in the base codebook: definition"):
        build_ablations(base, {"Aesthetic": {"definition": "x"}}, [("definition", "Definition")])


def test_clean_errors_instead_of_tracebacks(tmp_path, capsys):
    assert main(["render", str(tmp_path / "missing.yaml")]) == 2
    assert main(["render", str(FIX_CB / "toy_plain.yaml"), "--template", "nosuch"]) == 2
    bad_tpl = tmp_path / "t.yaml"
    bad_tpl.write_text("role: R\n")
    assert main(["render", str(FIX_CB / "toy_plain.yaml"), "--template", str(bad_tpl)]) == 2
    assert main(["convert", str(tmp_path / "missing.csv"), "-o", str(tmp_path / "o.yaml")]) == 2
    err = capsys.readouterr().err
    assert "Traceback" not in err and "unknown prompt template" in err and "lacks keys" in err


def test_strip_text_blanks_evidence_quotes(tmp_path):
    rec = {"item_id": "a", "text": "secret text", "raw_response": "{}", "predictions": {"A": True},
           "evidence": {"A": [{"quote": "secret", "verified": True}]},
           "first_pass": {"raw_response": "{}", "evidence": {"A": [{"quote": "secret", "verified": True}]}}}
    src = tmp_path / "r.jsonl"
    src.write_text(json.dumps(rec) + "\n")
    assert main(["strip-text", str(src), "--out-dir", str(tmp_path / "out")]) == 0
    out = (tmp_path / "out" / "r.jsonl").read_text()
    assert "secret" not in out
    r = json.loads(out)
    assert r["evidence"]["A"] == [{"quote": None, "verified": True}]


def test_evaluate_skips_attempts_files(tmp_path, base_cfg, capsys):
    from codebook_annotator.config import build_config
    from codebook_annotator.runner import run
    cfg = build_config([], base_cfg)
    run(cfg, log=lambda m: None)
    attempts = cfg.output_file.with_suffix(".attempts.jsonl")
    attempts.write_text(cfg.output_file.read_text())
    gold = tmp_path / "gold.jsonl"
    gold.write_text("".join(json.dumps({"item_id": i, "label_majority": []}) + "\n" for i in ["a1", "b2", "c3"]))
    out = tmp_path / "res.xlsx"
    assert main(["evaluate", "--dataset", str(gold), "--runs", str(cfg.output_file), str(attempts), "-o", str(out)]) == 0
    import pandas as pd
    assert pd.read_excel(out, sheet_name="Sheet1")["run_id"].unique().tolist() == ["t"]
    assert "skipped 1 .attempts.jsonl" in capsys.readouterr().out
