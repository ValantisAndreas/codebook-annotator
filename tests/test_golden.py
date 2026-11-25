"""Golden tests: frozen prompt and parser behaviour.

Fixtures in tests/fixtures were generated from synthetic inputs only. These tests
must never be edited to make a change pass. A failure means the frozen behaviour changed.
"""

import json
from pathlib import Path

import pytest

from codebook_annotator.parsing import parse_llm_response
from codebook_annotator.prompt import compile_system_prompt, get_user_prompt, load_codebook

FIX = Path(__file__).parent / "fixtures"
CODEBOOKS = sorted((FIX / "codebooks").glob("*.yaml"))
PARSER = json.loads((FIX / "golden_parser_cases.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("cb_path", CODEBOOKS, ids=lambda p: p.stem)
def test_system_prompt_moral_foundations_v1_byte_identical(cb_path):
    expected = (FIX / "golden_prompts" / f"{cb_path.stem}.moral_foundations_v1.system.txt").read_text(encoding="utf-8")
    assert compile_system_prompt(load_codebook(cb_path)) == expected


def test_user_prompt_byte_identical():
    text = (FIX / "golden_prompts" / "user_prompt.input.txt").read_text(encoding="utf-8")
    expected = (FIX / "golden_prompts" / "user_prompt.txt").read_text(encoding="utf-8")
    assert get_user_prompt(text) == expected


@pytest.mark.parametrize("name", sorted(PARSER["cases"]))
def test_parser_golden(name):
    case = PARSER["cases"][name]
    preds, status, notes, reasoning, used_fb, fb_key = parse_llm_response(case["raw"], PARSER["categories"])
    assert preds == case["predictions"]
    assert status == case["parse_status"]
    assert notes == case["parse_notes"]
    if case["reasoning"] == "__NaN__":
        assert reasoning != reasoning
    else:
        assert reasoning == case["reasoning"]
    assert used_fb == case["used_fallback"]
    assert fb_key == case["fallback_key"]
