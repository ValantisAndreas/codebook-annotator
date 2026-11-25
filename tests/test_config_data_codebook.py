import json

import pytest
import yaml

from codebook_annotator.codebook import CodebookError, codebook_from_dict, load_codebook
from codebook_annotator.config import ConfigError, build_config
from codebook_annotator.data import DataError, load_records, select_sample

# ── config ─────────────────────────────────────────────────────────────────────────────────────


def test_precedence_defaults_file_cli(tmp_path, base_cfg):
    f1 = tmp_path / "a.yaml"
    f1.write_text(yaml.safe_dump({**base_cfg, "temperature": 0.7, "seed": 3}))
    f2 = tmp_path / "b.json"
    f2.write_text(json.dumps({"seed": 4}))
    cfg = build_config([f1, f2], {"temperature": 0.0})  # explicit CLI value equal to the default still wins
    assert cfg.temperature == 0.0 and cfg.seed == 4 and cfg.num_ctx == 65536


def test_unknown_key_rejected(base_cfg):
    with pytest.raises(ConfigError, match="tempreature"):
        build_config([], {**base_cfg, "tempreature": 1})


def test_missing_required(base_cfg):
    cfg = dict(base_cfg)
    del cfg["dataset"]
    with pytest.raises(ConfigError, match="dataset"):
        build_config([], cfg)


@pytest.mark.parametrize("v", ["null", "None", ""])
def test_null_digest_strings(base_cfg, v):
    assert build_config([], {**base_cfg, "model_digest": v}).model_digest is None


def test_provider_requirements(base_cfg):
    with pytest.raises(ConfigError, match="replay_from"):
        build_config([], {**base_cfg, "provider": "replay"})
    with pytest.raises(ConfigError, match="api_base"):
        build_config([], {**base_cfg, "provider": "openai_compatible"})


# ── data ───────────────────────────────────────────────────────────────────────────────────────


def test_formats(tmp_path):
    (tmp_path / "x.csv").write_text("id,body\n2,two\n1,one\n", encoding="utf-8")
    (tmp_path / "x.tsv").write_text("id\tbody\n1\tone\n", encoding="utf-8")
    (tmp_path / "x.json").write_text(json.dumps([{"id": 1, "body": "one"}]), encoding="utf-8")
    import openpyxl
    wb = openpyxl.Workbook()
    wb.active.append(["id", "body"])
    wb.active.append([7, "seven"])
    wb.save(tmp_path / "x.xlsx")
    for ext in ("csv", "tsv", "json", "xlsx"):
        recs = load_records(tmp_path / f"x.{ext}")
        s = select_sample(recs, id_field="id", text_field="body")
        assert all(isinstance(i["item_id"], str) and isinstance(i["text"], str) for i in s)
    with pytest.raises(DataError, match="unsupported"):
        (tmp_path / "x.txt").write_text("a")
        load_records(tmp_path / "x.txt")


def test_sampling_reference_rule(items_file):
    recs = load_records(items_file)
    logs = []
    assert [i["item_id"] for i in select_sample(recs, split_value="test")] == ["a1", "b2"]
    assert [i["item_id"] for i in select_sample(recs, split_value="test", sample_size=1)] == ["a1"]
    # no record matches -> all records, with a warning
    s = select_sample(recs, split_value="dev", log=logs.append)
    assert len(s) == 3 and "no records" in logs[0]
    # more requested than available -> all, with a warning
    s = select_sample(recs, sample_size=10, log=logs.append)
    assert len(s) == 3 and "larger than" in logs[-1]
    r1 = select_sample(recs, sample_size=2, sampling="random", sampling_seed=1)
    assert r1 == select_sample(recs, sample_size=2, sampling="random", sampling_seed=1)
    assert [i["item_id"] for i in r1] == sorted(i["item_id"] for i in r1)


def test_data_errors(items_file):
    recs = load_records(items_file)
    with pytest.raises(DataError, match="field 'body' not found"):
        select_sample(recs, text_field="body")
    with pytest.raises(DataError, match="duplicate"):
        select_sample(recs + [recs[0]])


# ── codebook ───────────────────────────────────────────────────────────────────────────────────


def test_codebook_validation():
    with pytest.raises(CodebookError, match="categories"):
        codebook_from_dict({"render": []})
    with pytest.raises(CodebookError, match="unknown prompt key"):
        codebook_from_dict({"categories": {"A": {"definition": "x"}}, "prompt": {"rol": "x"}})
    with pytest.raises(CodebookError, match="output.type"):
        codebook_from_dict({"categories": {"A": {"definition": "x"}}, "output": {"type": "scale"}})
    with pytest.raises(CodebookError, match="unknown category 'B'"):
        codebook_from_dict({"categories": {"A": {"definition": "x"}}, "examples": [{"text": "t", "labels": ["B"]}]})
    cb = codebook_from_dict({"categories": {"A": {"definition": "x"}}, "foo": 1,
                             "render": [{"field": "definition", "label": "D"}, {"field": "nope", "label": "N"}]})
    msgs = " ".join(str(w) for w in cb.warnings)
    assert "unknown top-level key 'foo'" in msgs and "'nope'" in msgs


def test_codebook_properties(plain_codebook):
    cb = load_codebook(plain_codebook)
    assert cb.id == "toy_plain" and cb.category_keys == ["Aesthetic", "Historic", "Social"]
    assert cb.output_type == "binary" and len(cb.sha256) == 64
