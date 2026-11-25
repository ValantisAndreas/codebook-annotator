import json

import httpx
import pytest
from conftest import ScriptedProvider, answer

from codebook_annotator.config import build_config
from codebook_annotator.providers.adapters import OllamaProvider
from codebook_annotator.providers.base import ProviderResponse
from codebook_annotator.runner import RunError, run

ALL_FALSE = {"Aesthetic": False, "Historic": False, "Social": False}
BASE_RECORD_KEYS = ["item_id", "text", "run", "parse_status", "parse_notes", "predictions", "raw_response", "reasoning"]
BASE_RUN_KEYS = ["dataset", "dataset_revision", "codebook_id", "codebook_version", "model_tag", "model_digest",
                  "provider", "prompt_condition", "sample_id", "context_window", "temperature", "seed", "timestamp"]


def read(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def test_record_schema_and_order(base_cfg):
    cfg = build_config([], base_cfg)
    s = run(cfg, log=lambda m: None)
    recs = read(cfg.output_file)
    assert s.exit_code == 0 and [r["item_id"] for r in recs] == ["a1", "b2", "c3"]
    r = recs[0]
    assert list(r)[:8] == BASE_RECORD_KEYS  # base fields first, in fixed order
    assert list(r["run"])[:13] == BASE_RUN_KEYS
    assert r["run"]["dataset"] == "items" and r["run"]["codebook_id"] == "toy_plain"
    assert r["run"]["codebook_version"] == "1.0" and r["run"]["prompt_condition"] == "zero_shot"
    assert r["run"]["timestamp"].endswith("Z") and r["schema_version"] == 2
    for k in ("codebook_sha256", "system_prompt_sha256", "config_sha256", "sample_sha256"):
        assert len(r["run"][k]) == 64
    assert set(r["usage"]) == {"input_tokens", "output_tokens", "latency_ms", "passes"}
    assert "_fallback" not in r and "scores" not in r and "evidence" not in r and "first_pass" not in r


def test_failures_recorded_and_exit_code(base_cfg):
    def script(iid, s, u):
        if iid == "a1":
            return ProviderResponse("", "connection refused")
        if iid == "b2":
            return "not json"
        return answer(ALL_FALSE)
    cfg = build_config([], base_cfg)
    s = run(cfg, provider=ScriptedProvider(script), log=lambda m: None)
    recs = {r["item_id"]: r for r in read(cfg.output_file)}
    assert s.exit_code == 1 and s.errors == 2
    assert recs["a1"]["parse_status"] == "failed" and recs["a1"]["parse_notes"] == "Harness query error: connection refused"
    assert recs["a1"]["raw_response"] == "" and recs["a1"]["reasoning"] is None
    assert recs["b2"]["parse_status"] == "failed" and recs["b2"]["parse_notes"].startswith("JSON parse error")


def test_resume_retries_non_ok_and_keeps_attempts(base_cfg):
    calls = {"n": 0}

    def flaky(iid, s, u):
        calls["n"] += 1
        return ProviderResponse("", "boom") if (iid == "b2" and calls["n"] <= 3) else answer(ALL_FALSE)
    cfg = build_config([], base_cfg)
    p = ScriptedProvider(flaky)
    assert run(cfg, provider=p, log=lambda m: None).exit_code == 1
    p2 = ScriptedProvider(flaky)
    s = run(cfg, provider=p2, log=lambda m: None)
    assert s.exit_code == 0 and [c["item_id"] for c in p2.calls] == ["b2"]
    recs = read(cfg.output_file)
    assert sorted(r["item_id"] for r in recs) == ["a1", "b2", "c3"] and all(r["parse_status"] == "ok" for r in recs)
    attempts = read(cfg.output_file.with_suffix(".attempts.jsonl"))
    assert [a["item_id"] for a in attempts] == ["b2"] and attempts[0]["parse_status"] == "failed"


def test_resume_refuses_changed_config(base_cfg):
    cfg = build_config([], base_cfg)
    run(cfg, log=lambda m: None)
    with pytest.raises(RunError, match="different configuration"):
        run(build_config([], {**base_cfg, "temperature": 0.5}), log=lambda m: None)
    # force_resume allows it
    run(build_config([], {**base_cfg, "temperature": 0.5, "force_resume": True}), log=lambda m: None)


def test_resume_no_hash_records(base_cfg, tmp_path):
    """Records without config_sha256 are checked on key fields."""
    cfg = build_config([], base_cfg)
    cfg.output_dir.mkdir(parents=True)
    no_hash = {"item_id": "a1", "run": {"codebook_id": "toy_plain", "model_tag": "test-model", "provider": "stub",
                                       "temperature": 0.0}, "parse_status": "ok", "parse_notes": None,
              "predictions": ALL_FALSE}
    cfg.output_file.write_text(json.dumps(no_hash) + "\n")
    p = ScriptedProvider(lambda *a: answer(ALL_FALSE))
    run(cfg, provider=p, log=lambda m: None)
    assert [c["item_id"] for c in p.calls] == ["b2", "c3"]
    no_hash["run"]["model_tag"] = "other"
    cfg.output_file.write_text(json.dumps(no_hash) + "\n")
    with pytest.raises(RunError, match="different configuration"):
        run(cfg, provider=p, log=lambda m: None)


def test_concurrency_preserves_order_and_records(base_cfg):
    import random
    import time

    def slow(iid, s, u):
        time.sleep(random.random() / 50)
        return answer({**ALL_FALSE, "Social": iid == "c3"})
    out = {}
    for c in (1, 3):
        cfg = build_config([], {**base_cfg, "concurrency": c, "run_id": f"c{c}"})
        run(cfg, provider=ScriptedProvider(slow), log=lambda m: None)
        out[c] = [(r["item_id"], r["predictions"]) for r in read(cfg.output_file)]
    assert out[1] == out[3]


def test_storage_flags(base_cfg):
    cfg = build_config([], {**base_cfg, "store_text": False, "store_raw_response": False})
    run(cfg, log=lambda m: None)
    r = read(cfg.output_file)[0]
    assert "text" not in r and "raw_response" not in r


def test_internal_error_is_recorded(base_cfg):
    def boom(*a):
        raise RuntimeError("bad")
    cfg = build_config([], base_cfg)
    s = run(cfg, provider=ScriptedProvider(boom), log=lambda m: None)
    assert s.errors == 3 and "Harness internal error" in read(cfg.output_file)[0]["parse_notes"]


def test_string_booleans_read_by_meaning(base_cfg):
    cfg = build_config([], base_cfg)
    run(cfg, provider=ScriptedProvider(lambda *a: answer({"Aesthetic": "false", "Historic": "true", "Social": False})),
        log=lambda m: None)
    r = read(cfg.output_file)[0]
    assert r["predictions"] == {"Aesthetic": False, "Historic": True, "Social": False}
    assert r["parse_status"] == "ok" and "non-boolean value for 'Aesthetic'" in r["parse_notes"]


# ── Ollama-specific checks (mocked server) ────────────────────────────────────────────────────

def _ollama(models, digest="46e0c10c039eabc"):
    def handler(req):
        if req.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": m, "model": m, "digest": digest} for m in models]})
        return httpx.Response(200, json={"response": answer(ALL_FALSE), "prompt_eval_count": 3, "eval_count": 2})
    return OllamaProvider(model_tag="llama3.1:8b", num_ctx=2048, transport=httpx.MockTransport(handler))


def test_ollama_model_missing(base_cfg):
    cfg = build_config([], {**base_cfg, "provider": "ollama", "model_tag": "llama3.1:8b"})
    with pytest.raises(RunError, match="not installed. Installed: qwen3:30b"):
        run(cfg, provider=_ollama(["qwen3:30b"]), log=lambda m: None)


def test_ollama_digest_auto_and_mismatch(base_cfg):
    cfg = build_config([], {**base_cfg, "provider": "ollama", "model_tag": "llama3.1:8b", "model_digest": "auto"})
    run(cfg, provider=_ollama(["llama3.1:8b"]), log=lambda m: None)
    r = read(cfg.output_file)[0]
    assert r["run"]["model_digest"] == "sha256:46e0c10c039eabc" == r["run"]["model_digest_observed"]
    assert r["usage"]["input_tokens"] == 3
    bad = build_config([], {**base_cfg, "provider": "ollama", "model_tag": "llama3.1:8b", "run_id": "x",
                            "model_digest": "sha256:19e422b02313"})
    with pytest.raises(RunError, match="does not match"):
        run(bad, provider=_ollama(["llama3.1:8b"]), log=lambda m: None)
    ok = build_config([], {**base_cfg, "provider": "ollama", "model_tag": "llama3.1:8b", "run_id": "y",
                           "model_digest": "sha256:46e0c10c039e"})  # short prefix matches
    assert run(ok, provider=_ollama(["llama3.1:8b"]), log=lambda m: None).exit_code == 0


def test_auto_digest_with_non_ollama_provider(base_cfg):
    logs = []
    cfg = build_config([], {**base_cfg, "model_digest": "auto"})
    assert run(cfg, log=logs.append).exit_code == 0
    assert read(cfg.output_file)[0]["run"]["model_digest"] is None
    assert any("only works with the ollama provider" in m for m in logs)


def test_fingerprint_follows_resolved_ollama_digest(base_cfg):
    cfg = build_config([], {**base_cfg, "provider": "ollama", "model_tag": "llama3.1:8b", "model_digest": "auto"})
    run(cfg, provider=_ollama(["llama3.1:8b"], digest="aaaaaaaaaaaaaaaa"), log=lambda m: None)
    # the model was re-pulled: same config file, different model version -> resume refused
    with pytest.raises(RunError, match="different configuration"):
        run(cfg, provider=_ollama(["llama3.1:8b"], digest="bbbbbbbbbbbbbbbb"), log=lambda m: None)
    # same model version -> resume accepted
    run(cfg, provider=_ollama(["llama3.1:8b"], digest="aaaaaaaaaaaaaaaa"), log=lambda m: None)
