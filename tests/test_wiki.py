"""The Wiki build turns every docs page into a self-contained Wiki page with working links."""

import importlib.util
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("build_wiki", ROOT / "scripts" / "build_wiki.py")
build_wiki = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_wiki)


def test_wiki_build(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    out = tmp_path / "wiki"
    written = build_wiki.build_wiki("docs", out, "mkdocs.yml")
    names = {p.name for p in written}
    assert "Home.md" in names and "_Sidebar.md" in names and "_Footer.md" in names
    n_docs = len(list((ROOT / "docs").rglob("*.md")))
    assert len(written) == n_docs + 2
    page_names = {p.stem for p in written}
    link = re.compile(r"(?<!!)\[[^\]]+\]\(([^)\s]+)\)")
    for page in written:
        text = page.read_text(encoding="utf-8")
        assert "--8<--" not in text, page.name
        assert not re.search(r"^(!!!|\?\?\?)", text, re.M), page.name
        for target in link.findall(text):
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            assert target.split("#")[0] in page_names, f"{page.name}: broken link {target}"
    # every nav page is in the sidebar, in nav order
    nav = build_wiki._nav_entries(yaml.load((ROOT / "mkdocs.yml").read_text(), Loader=yaml.BaseLoader)["nav"])
    sidebar = (out / "_Sidebar.md").read_text(encoding="utf-8")
    positions = [sidebar.index(f"]({build_wiki.page_slug(build_wiki.PurePosixPath(p))})") for _, _, p in nav]
    assert positions == sorted(positions)


def test_admonition_and_links():
    md = 'Intro\n\n!!! tip "No model?"\n    Use the stub.\n    Second line.\n\nAfter'
    out = build_wiki.convert_admonitions(md)
    assert "> **No model?**" in out and "> Use the stub." in out and out.endswith("After")
    assert build_wiki.page_slug(build_wiki.PurePosixPath("guides/optional-features.md")) == "Guides-Optional-Features"
