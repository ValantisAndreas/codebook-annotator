"""Regression tests for the run-side fixes (data safety, privacy, robustness, clean errors)."""

import json

import pytest
import yaml
from conftest import FIX, ScriptedProvider, answer

from codebook_annotator.cli import main
from codebook_annotator.config import build_config
from codebook_annotator.data import DataError, load_records, select_sample
from codebook_annotator.data_tools import redact_raw_excerpt, strip_run_text
from codebook_annotator.parsing import parse_response
from codebook_annotator.providers.base import ProviderResponse, join_url, redact_urls
from codebook_annotator.runner import RunError, run

ALL_FALSE = {"Aesthetic": False, "Historic": False, "Social": False}
CB = str(FIX / "codebooks" / "toy_plain.yaml")


def read(p):
    return [json.loads(line) for line in open(p, encoding="utf-8")]


def quiet(_m):
    return None


# 1. strip-text never overwrites its input ─────────────────────────────────────────────────────

def test_strip_text_refuses_to_overwrite_input(tmp_path, capsys):
    runs = tmp_path / "runs"
    runs.mkdir()
    f = runs / "r.jsonl"
    content = "".join(json.dumps({"item_id": f"i{k}", "text": "t", "predictions": {}}) + "\n" for k in range(3))
    f.write_text(content)
    assert main(["strip-text", str(f), "--out-dir", str(runs)]) == 2
    assert f.read_text() == content
    assert "would overwrite" in capsys.readouterr().err
    with pytest.raises(DataError, match="input file itself"):
        strip_run_text(f, f)


# 2. API keys and URLs ───────────────────────────────────────────────────────────────────────

def test_openai_key_only_sent_to_openai(monkeypatch):
    from codebook_annotator.providers import make_provider
    monkeypatch.setenv("OPENAI_API_KEY", "sk-SECRET")
    monkeypatch.setenv("OTHER_KEY", "k2")
    base = {"run_id": "x", "codebook": "c", "dataset": "d", "provider": "openai_compatible", "model_tag": "m"}

    def auth(**extra):
        return make_provider(build_config([], {**base, **extra}), []).headers().get("Authorization")

    assert auth(api_base="https://other.example/v1") is None
    assert auth(api_base="http://localhost:8000/v1") is None
    assert auth(api_base="https://api.openai.com/v1") == "Bearer sk-SECRET"
    assert auth(api_base="https://other.example/v1", api_key_env="OTHER_KEY") == "Bearer k2"


def test_urls_keep_query_and_errors_hide_secrets():
    assert join_url("https://h/v1?api-version=2", "/chat/completions") == "https://h/v1/chat/completions?api-version=2"
    assert join_url("https://api.mistral.ai/v1/", "/chat/completions") == "https://api.mistral.ai/v1/chat/completions"
    msg = redact_urls("Client error for url 'https://user:pw@h/v1/chat/completions?key=SECRET'")
    assert "pw" not in msg and "SECRET" not in msg and "https://h/v1/chat/completions" in msg


# 3. no text in notes when raw answers are not stored ────────────────────────────────────────

def test_no_raw_excerpt_when_raw_response_not_stored(base_cfg):
    cfg = build_config([], {**base_cfg, "store_raw_response": False, "verify": True, "sample_size": 1})
    run(cfg, provider=ScriptedProvider(lambda *a: "SECRET TEXT not json"), log=quiet)
    r = read(cfg.output_file)[0]
    assert "SECRET" not in json.dumps(r)
    assert "Raw response not stored" in r["parse_notes"] and "verify pass skipped" in r["parse_notes"]
    assert "raw_response" not in r and "raw_response" not in r["first_pass"]


def test_strip_text_removes_excerpts_and_unreadable_chunks(tmp_path):
    notes = parse_response("SECRET answer", ["A"]).parse_notes + "\x3b verify pass skipped (first pass failed)"
    assert redact_raw_excerpt(notes).endswith("Raw response: [removed]\x3b verify pass skipped (first pass failed)")
    src = tmp_path / "a.attempts.jsonl"
    src.write_text(json.dumps({"item_id": "a", "parse_notes": notes, "first_pass": {"parse_notes": notes}}) + "\n"
                   + json.dumps({"unparseable_output_chunk": '{"item_id": "b", "text": "SECRET'}) + "\n")
    strip_run_text(src, tmp_path / "out.jsonl")
    out = (tmp_path / "out.jsonl").read_text()
    assert "SECRET" not in out and out.count("[removed]") == 2


# 4. numeric ids sort as numbers ──────────────────────────────────────────────────────────────

def test_numeric_ids_sort_numerically():
    recs = [{"id": i, "text": "t"} for i in (2, 10, 9)]
    assert [s["item_id"] for s in select_sample(recs, id_field="id")] == ["2", "9", "10"]
    assert [s["item_id"] for s in select_sample(recs, id_field="id", sample_size=2)] == ["2", "9"]
    mixed = [{"id": i, "text": "t"} for i in ("b2", "a10", "a9")]
    assert [s["item_id"] for s in select_sample(mixed, id_field="id")] == ["a10", "a9", "b2"]


# 5. resume reader ─────────────────────────────────────────────────────────────────────────────

def _finished_run(base_cfg):
    cfg = build_config([], base_cfg)
    run(cfg, provider=ScriptedProvider(lambda *a: answer(ALL_FALSE)), log=quiet)
    return cfg


def test_truncated_line_does_not_hide_later_records(base_cfg):
    cfg = _finished_run(base_cfg)
    lines = cfg.output_file.read_text().splitlines()
    lines[0] = lines[0][:40]  # damaged first record
    cfg.output_file.write_text("\n".join(lines) + "\n")
    p = ScriptedProvider(lambda *a: answer(ALL_FALSE))
    logs = []
    run(cfg, provider=p, log=logs.append)
    assert [c["item_id"] for c in p.calls] == ["a1"]  # only the damaged item is annotated again
    assert any("1 unreadable chunk" in m for m in logs)
    assert sorted(r["item_id"] for r in read(cfg.output_file)) == ["a1", "b2", "c3"]


def test_odd_lines_and_duplicates_are_set_aside(base_cfg):
    cfg = _finished_run(base_cfg)
    recs = read(cfg.output_file)
    extra = ["[]", json.dumps({"foo": 1}), json.dumps({**recs[1], "run": None}), json.dumps(recs[2])]
    cfg.output_file.write_text(cfg.output_file.read_text() + "\n".join(extra) + "\n")
    logs = []
    p = ScriptedProvider(lambda *a: answer(ALL_FALSE))
    run(cfg, provider=p, log=logs.append)
    assert p.calls == []
    out = read(cfg.output_file)
    assert sorted(r["item_id"] for r in out) == ["a1", "b2", "c3"]
    assert any("1 duplicate record" in m and "3 unreadable chunk" in m for m in logs)


def test_resume_refusal_names_the_difference(base_cfg):
    cfg = _finished_run(base_cfg)
    with pytest.raises(RunError, match="temperature: 0.0 in the file, 0.5 now"):
        run(build_config([], {**base_cfg, "temperature": 0.5}), log=quiet)
    with pytest.raises(RunError, match="not stored in the records"):
        run(build_config([], {**base_cfg, "max_tokens": 50}), log=quiet)
    assert cfg.output_file.exists()


# 9. non-finite ordinal scores ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("value", ['"inf"', "1e400", '"nan"', "9" * 400])
def test_non_finite_or_huge_scores_are_invalid_not_crashes(value):
    r = parse_response('{"predictions": {"A": ' + value + ', "B": 3}}', ["A", "B"], output_type="ordinal")
    assert r.parse_status == "partial" and r.scores == {"A": None, "B": 3}
    assert "Invalid score for 'A'" in r.parse_notes and len(r.parse_notes) < 200


# 10. failed verify pass ──────────────────────────────────────────────────────────────────────

def test_failed_verify_pass_gives_empty_failed_result(base_cfg):
    def script(iid, system, user):
        if "### Your previous annotation" in user:
            return ProviderResponse("", "server down", input_tokens=None, output_tokens=None, latency_ms=2.0)
        return ProviderResponse(answer({"Aesthetic": True, "Historic": False, "Social": False}), None, 10, 5, 1.0)
    cfg = build_config([], {**base_cfg, "verify": True, "sample_size": 1})
    run(cfg, provider=ScriptedProvider(script), log=quiet)
    r = read(cfg.output_file)[0]
    assert r["parse_status"] == "failed" and "verify pass" in r["parse_notes"]
    assert r["predictions"] == ALL_FALSE and r["reasoning"] is None
    assert r["first_pass"]["predictions"]["Aesthetic"] is True
    assert r["usage"] == {"input_tokens": 10, "output_tokens": 5, "latency_ms": 3.0, "passes": 2}


# 11. clean errors instead of tracebacks ──────────────────────────────────────────────────────

def test_bad_inputs_give_clean_errors(tmp_path, capsys):
    bad_json = tmp_path / "d.json"
    bad_json.write_text("[{")
    with pytest.raises(DataError, match="invalid JSON"):
        load_records(bad_json)
    import openpyxl
    wb = openpyxl.Workbook()
    wb.save(tmp_path / "empty.xlsx")
    with pytest.raises(DataError, match="is empty"):
        load_records(tmp_path / "empty.xlsx")
    with pytest.raises(DataError, match="sheet 'Nope' not found"):
        load_records(tmp_path / "empty.xlsx", sheet="Nope")

    latin = tmp_path / "latin.yaml"
    latin.write_bytes("categories:\n  A: {definition: caf\xe9}\n".encode("latin-1"))
    typed = tmp_path / "typed.yaml"
    typed.write_text(yaml.safe_dump({"categories": {"A": {"definition": "x"}}, "output": [1],
                                     "examples": [{"text": "t", "scores": ["A"]}]}))
    scale = tmp_path / "scale.yaml"
    scale.write_text(yaml.safe_dump({"categories": {"A": {"definition": "x"}},
                                     "output": {"type": "ordinal", "scale": [1, 5]}}))
    empty_tpl = tmp_path / "empty_tpl.yaml"
    empty_tpl.write_text("")
    no_cats = tmp_path / "base.yaml"
    no_cats.write_text("render: []\n")
    ext = tmp_path / "ext.csv"
    ext.write_text("category,ctx\nA,x\n")
    items = tmp_path / "items.jsonl"
    items.write_text(json.dumps({"item_id": "a", "text": "t"}) + "\n")
    rows = tmp_path / "rows.csv"
    rows.write_text("text,annotation\nhi,A\n")
    run_file = tmp_path / "r.jsonl"
    run_file.write_text("not json\n")

    calls = [
        ["validate", str(latin)],
        ["validate", str(typed)],
        ["validate", str(scale)],
        ["render", CB, "--few-shot"],
        ["render", CB, "--template", str(empty_tpl)],
        ["overlap", str(items), "-o", str(tmp_path / "o.jsonl"), "--codebook", CB, "--field", "nothing"],
        ["overlap", str(items), "-o", str(tmp_path / "o.jsonl"), "--codebook", CB, "--text-field", "body"],
        ["aggregate", str(rows), "-o", str(tmp_path / "a.jsonl")],
        ["ablate", "--base", str(no_cats), "--extension", str(ext), "--fields", "ctx", "--out-dir", str(tmp_path)],
        ["check-invariance", CB],
        ["strip-text", str(run_file), "--out-dir", str(tmp_path / "shared")],
    ]
    for argv in calls:
        assert main(argv) in (1, 2), argv
    captured = capsys.readouterr()
    err = captured.err + captured.out  # validate reports on stdout, the other commands on stderr
    assert "Traceback" not in err
    for expected in ("not UTF-8", "output' must be a mapping", "scores must be a mapping",
                     "output.scale must be a mapping", "no 'examples'", "must be a mapping with the keys",
                     "no words found", "field 'body' not found", "column(s) not found: annotator",
                     "needs a 'categories' mapping", "at least two codebooks", "not a JSON record"):
        assert expected in err, expected


def test_multi_line_rules_heading_is_found(tmp_path):
    from codebook_annotator.codebook import codebook_from_dict
    from codebook_annotator.codebook_tools import check_invariance
    a = codebook_from_dict({"prompt": {"rules_heading": "### Rules\nRead carefully."},
                            "categories": {"A": {"definition": "x"}}, "render": [{"field": "definition", "label": "D"}]})
    b = codebook_from_dict({"prompt": {"rules_heading": "### Rules\nRead carefully."},
                            "categories": {"A": {"definition": "y"}}, "render": [{"field": "definition", "label": "D"}]})
    assert check_invariance([a, b]).ok


def test_keyboard_interrupt_exits_130(monkeypatch, tmp_path, base_cfg, capsys):
    import codebook_annotator.runner as runner

    def interrupted(*a, **k):
        raise KeyboardInterrupt
    monkeypatch.setattr(runner, "run", interrupted)
    cfg = tmp_path / "run.yaml"
    cfg.write_text(yaml.safe_dump(base_cfg))
    assert main(["run", "-c", str(cfg)]) == 130
    assert "Run the same command again to resume" in capsys.readouterr().err


# 12. run summary counts ──────────────────────────────────────────────────────────────────────

def test_summary_never_negative(base_cfg):
    logs = []
    cfg = build_config([], base_cfg)
    run(cfg, provider=ScriptedProvider(lambda *a: json.dumps({"results": {"Aesthetic": True}})), log=logs.append)
    assert "  - Canonical parses ('predictions' key): 0/3" in logs
