"""The generated reference pages in docs/reference must match the code."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("gen_reference", ROOT / "scripts" / "gen_reference.py")
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)


@pytest.mark.parametrize("rel", sorted(gen.PAGES))
def test_reference_page_is_up_to_date(rel):
    committed = (ROOT / rel).read_text(encoding="utf-8")
    assert committed == gen.PAGES[rel](), f"{rel} is out of date. Run: python scripts/gen_reference.py"


def test_every_config_field_is_documented():
    page = gen.configuration_page()
    for name, field in gen.RunConfig.model_fields.items():
        assert f"`{name}`" in page
        assert field.description, f"RunConfig.{name} has no description"
