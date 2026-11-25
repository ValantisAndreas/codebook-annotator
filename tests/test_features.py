"""Optional features: all off by default, and each one changes run.prompt_condition."""

import json

import httpx
import pytest
import yaml
from conftest import FIX, ScriptedProvider, answer

from codebook_annotator.codebook import codebook_from_dict, load_codebook
from codebook_annotator.config import build_config
from codebook_annotator.parsing import parse_response, verify_quote
from codebook_annotator.prompt import compile_system_prompt, prompt_condition, render_prompts
from codebook_annotator.providers.adapters import OpenAICompatibleProvider
from codebook_annotator.runner import run

CATS = ["Aesthetic", "Historic", "Social"]
TEXT = "A beautiful view over the bay, founded in 1600."


def read(p):
    return [json.loads(line) for line in open(p, encoding="utf-8")]


def test_defaults_unchanged_and_condition_strings():
    cb = load_codebook(FIX / "codebooks" / "toy_plain.yaml")
    golden = (FIX / "golden_prompts" / "toy_plain.moral_foundations_v1.system.txt").read_text(encoding="utf-8")
    assert render_prompts(cb, "moral_foundations_v1").system == golden
    assert compile_system_prompt(yaml.safe_load((FIX / "codebooks" / "toy_plain.yaml").read_text())) == golden
    assert render_prompts(cb, "moral_foundations_v1").prompt_condition == "zero_shot"
    assert prompt_condition(few_shot=True, evidence=True, ordinal=True, verify=True) == "few_shot+evidence+ordinal+verify"


def test_default_template_and_codebook_overrides():
    cb = codebook_from_dict({"categories": {"A": {"definition": "d"}}, "prompt": {"role": "ROLE", "rules_heading": "## R"},
                             "render": [{"field": "definition", "label": "Def"}]})
    s = render_prompts(cb).system
    assert s.startswith("ROLE\nYour task is to predict the presence or absence of each of the following categories")
    assert "\n## R\n" in s and "moral" not in s.lower()


# ── evidence ──────────────────────────────────────────────────────────────────────────────────

def test_evidence_prompt_and_parse():
    cb = load_codebook(FIX / "codebooks" / "toy_plain.yaml")
    rp = render_prompts(cb, evidence=True)
    assert rp.prompt_condition == "zero_shot+evidence" and '3. "evidence"' in rp.system and "three top-level keys" in rp.system
    raw = answer({"Aesthetic": True, "Historic": True, "Social": False},
                 evidence={"Aesthetic": ["beautiful view"], "Historic": "founded in 1700", "Bogus": ["x"]})
    r = parse_response(raw, CATS, evidence=True, text=TEXT)
    assert r.parse_status == "ok"
    assert r.evidence == {"Aesthetic": [{"quote": "beautiful view", "verified": True}],
                          "Historic": [{"quote": "founded in 1700", "verified": False}]}
    assert "unknown categories ignored: Bogus" in r.parse_notes


@pytest.mark.parametrize("quote,ok", [
    ("beautiful view", True), ("A BEAUTIFUL  view", True), ('"over the bay"', True), ("...founded in 1600…", True),
    ("beautiful vista", False), ("", False), ("...", False),
])
def test_verify_quote(quote, ok):
    assert verify_quote(quote, TEXT) is ok


def test_evidence_enforce_policy(base_cfg):
    resp = answer({"Aesthetic": True, "Historic": True, "Social": False},
                  evidence={"Aesthetic": ["beautiful view"], "Historic": ["made up"]})
    cfg = build_config([], {**base_cfg, "evidence": True, "evidence_policy": "enforce", "sample_size": 1})
    run(cfg, provider=ScriptedProvider(lambda *a: resp), log=lambda m: None)
    r = read(cfg.output_file)[0]  # item a1: "A beautiful view over the bay."
    assert r["run"]["prompt_condition"] == "zero_shot+evidence"
    assert r["predictions"] == {"Aesthetic": True, "Historic": False, "Social": False}
    assert r["evidence_enforced"] == ["Historic"] and r["predictions_before_evidence"]["Historic"] is True
    cfg2 = build_config([], {**base_cfg, "evidence": True, "sample_size": 1, "run_id": "rep"})
    run(cfg2, provider=ScriptedProvider(lambda *a: resp), log=lambda m: None)
    r2 = read(cfg2.output_file)[0]
    assert r2["predictions"]["Historic"] is True and "evidence_enforced" not in r2  # report policy: unchanged


# ── ordinal ───────────────────────────────────────────────────────────────────────────────────

ORD = {"output": {"type": "ordinal", "scale": {"min": 1, "max": 10}}, "categories": {c: {"definition": c} for c in CATS},
       "render": [{"field": "definition", "label": "Definition"}]}


def test_ordinal_prompt_and_parse():
    rp = render_prompts(codebook_from_dict(ORD))
    assert rp.prompt_condition == "zero_shot+ordinal" and '"Aesthetic": 1-10 or null' in rp.system
    r = parse_response(answer({"Aesthetic": 7, "Historic": None, "Social": "3"}), CATS, output_type="ordinal")
    assert r.scores == {"Aesthetic": 7, "Historic": None, "Social": 3}
    assert r.predictions == {"Aesthetic": True, "Historic": False, "Social": True}
    assert r.parse_status == "ok" and "string score for 'Social'" in r.parse_notes
    bad = parse_response(answer({"Aesthetic": 11, "Historic": 2.5, "Social": True}), CATS, output_type="ordinal")
    assert bad.parse_status == "partial" and bad.scores == {"Aesthetic": None, "Historic": None, "Social": None}
    miss = parse_response(answer({"Aesthetic": 4}), CATS, output_type="ordinal")
    assert miss.parse_status == "partial" and miss.scores["Historic"] is None


def test_ordinal_run(base_cfg, tmp_path):
    p = tmp_path / "ord.yaml"
    p.write_text(yaml.safe_dump(ORD))
    cfg = build_config([], {**base_cfg, "codebook": str(p), "sample_size": 1})
    run(cfg, provider=ScriptedProvider(lambda *a: answer({"Aesthetic": 8, "Historic": None, "Social": None})),
        log=lambda m: None)
    r = read(cfg.output_file)[0]
    assert r["scores"] == {"Aesthetic": 8, "Historic": None, "Social": None} and r["predictions"]["Aesthetic"] is True


# ── few-shot ──────────────────────────────────────────────────────────────────────────────────

def test_few_shot_prompt(base_cfg):
    d = yaml.safe_load((FIX / "codebooks" / "toy_plain.yaml").read_text())
    d["examples"] = [{"text": "Lovely gardens.", "labels": ["Aesthetic"], "reasoning": "praise",
                      "evidence": {"Aesthetic": ["Lovely gardens"]}}]
    cb = codebook_from_dict(d)
    rp = render_prompts(cb, few_shot=True, evidence=True)
    assert rp.prompt_condition == "few_shot+evidence"
    assert rp.system.index("### Examples:") < rp.system.index("### Output Format Constraints:")
    assert '"Aesthetic": true' in rp.system and '"Lovely gardens"' in rp.system
    assert "### Examples:" not in render_prompts(cb).system  # examples ignored unless few_shot
    with pytest.raises(ValueError, match="no 'examples'"):
        render_prompts(load_codebook(FIX / "codebooks" / "toy_plain.yaml"), few_shot=True)


# ── verify pass ───────────────────────────────────────────────────────────────────────────────

def test_verify_pass(base_cfg):
    def script(iid, system, user):
        if "### Your previous annotation" in user:
            assert '"Social": true' in user  # first answer is shown to the model
            return answer({"Aesthetic": True, "Historic": False, "Social": False}, "verified")
        return answer({"Aesthetic": True, "Historic": False, "Social": True}, "first")
    cfg = build_config([], {**base_cfg, "verify": True, "sample_size": 1})
    p = ScriptedProvider(script)
    run(cfg, provider=p, log=lambda m: None)
    r = read(cfg.output_file)[0]
    assert len(p.calls) == 2 and p.calls[0]["system"] == p.calls[1]["system"]
    assert r["run"]["prompt_condition"] == "zero_shot+verify"
    assert r["predictions"]["Social"] is False and r["reasoning"] == "verified"
    assert r["first_pass"]["predictions"]["Social"] is True and r["usage"]["passes"] == 2
    assert r["usage"]["input_tokens"] == 20


def test_verify_skipped_when_first_pass_fails(base_cfg):
    cfg = build_config([], {**base_cfg, "verify": True, "sample_size": 1})
    p = ScriptedProvider(lambda *a: "garbage")
    run(cfg, provider=p, log=lambda m: None)
    r = read(cfg.output_file)[0]
    assert len(p.calls) == 1 and r["parse_status"] == "failed" and "verify pass skipped" in r["parse_notes"]


# ── openai-compatible provider ────────────────────────────────────────────────────────────────

def test_openai_compatible_payload():
    seen = {}

    def handler(req):
        seen["url"], seen["body"], seen["auth"] = str(req.url), json.loads(req.content), req.headers.get("authorization")
        return httpx.Response(200, json={"choices": [{"message": {"content": " {} "}}],
                                         "usage": {"prompt_tokens": 4, "completion_tokens": 2}})
    p = OpenAICompatibleProvider(model_tag="m", api_base="http://h/v1/", api_key="K", seed=3, max_tokens=50,
                                 transport=httpx.MockTransport(handler))
    out = p.query("S", "U")
    assert seen["url"] == "http://h/v1/chat/completions" and seen["auth"] == "Bearer K"
    assert seen["body"] == {"model": "m", "messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}],
                            "temperature": 0.0, "response_format": {"type": "json_object"}, "seed": 3, "max_tokens": 50}
    assert (out.text, out.input_tokens, out.output_tokens) == ("{}", 4, 2)


def test_evaluation_arm_key_separates_prompt_conditions():
    from codebook_annotator.evaluate.report import arm_key
    assert arm_key({"codebook_id": "B", "prompt_condition": "zero_shot"}) == "B"
    assert arm_key({"codebook_id": "B"}) == "B"
    assert arm_key({"codebook_id": "sup", "prompt_condition": "none (supervised fine-tuning)"}) == "sup"
    assert arm_key({"codebook_id": "B", "prompt_condition": "few_shot+evidence"}) == "B [few_shot+evidence]"
