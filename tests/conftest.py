import json
from pathlib import Path

import pytest

from codebook_annotator.providers.base import Provider, ProviderResponse

FIX = Path(__file__).parent / "fixtures"


class ScriptedProvider(Provider):
    """Test provider: `script(item_id, system, user) -> str | ProviderResponse`."""

    name = "scripted"

    def __init__(self, script, **kw):
        kw.setdefault("model_tag", "test-model")
        super().__init__(**kw)
        self.script = script
        self.calls = []

    def query(self, system_prompt, user_prompt, *, item_id=None):
        self.calls.append({"item_id": item_id, "system": system_prompt, "user": user_prompt})
        out = self.script(item_id, system_prompt, user_prompt)
        if isinstance(out, ProviderResponse):
            return out
        return ProviderResponse(out, None, input_tokens=10, output_tokens=5, latency_ms=1.0)


def answer(preds: dict, reasoning="r", **extra) -> str:
    return json.dumps({"predictions": preds, "reasoning": reasoning, **extra})


@pytest.fixture
def items_file(tmp_path):
    rows = [
        {"item_id": "b2", "text": "The fort was founded in 1600.", "split": "test", "group": "x"},
        {"item_id": "a1", "text": "A beautiful view over the bay.", "split": "test", "group": "y"},
        {"item_id": "c3", "text": "Locals gather here every week.", "split": "train", "group": "x"},
    ]
    p = tmp_path / "items.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return p


@pytest.fixture
def plain_codebook():
    return FIX / "codebooks" / "toy_plain.yaml"


@pytest.fixture
def base_cfg(tmp_path, items_file, plain_codebook):
    return {"run_id": "t", "codebook": str(plain_codebook), "dataset": str(items_file), "provider": "stub",
            "model_tag": "test-model", "output_dir": str(tmp_path / "runs")}
