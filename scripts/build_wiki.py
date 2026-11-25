"""Build GitHub Wiki pages from the MkDocs documentation in docs/.

The Wiki is a separate git repository and cannot render MkDocs features, so this script:
- flattens docs/ into one page per file (guides/running.md -> Guides-Running.md, index.md -> Home.md),
- expands snippet includes (lines like --8<-- "CONTRIBUTING.md", relative to the repository root),
- turns admonitions (!!! tip "Title" plus an indented body) into block quotes,
- rewrites links between docs pages to Wiki page names, and links to other repository files to
  their GitHub URL,
- writes _Sidebar.md in the order of the nav in mkdocs.yml, and _Footer.md.

Usage, from the repository root:

    python scripts/build_wiki.py                 # writes wiki-build/
    python scripts/build_wiki.py docs out_dir
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path, PurePosixPath

import yaml

REPOSITORY_URL = "https://github.com/ValantisAndreas/codebook-annotator"
BRANCH = "main"
HOME_TITLE = "codebook-annotator documentation"

MARKDOWN_LINK = re.compile(r"(?<!!)\[([^\]]+)\]\(([^)\s]+)\)")
PAGE_HEADING = re.compile(r"^#\s+(.+)$", re.MULTILINE)
SNIPPET = re.compile(r'^--8<--\s+"([^"]+)"\s*$', re.MULTILINE)
ADMONITION = re.compile(r'^(!!!|\?\?\?\+?)\s+(\w+)(?:\s+"([^"]*)")?\s*$')


def page_slug(relative_path: PurePosixPath) -> str:
    """Flat, URL-safe Wiki page name for one documentation file."""
    if relative_path.as_posix() == "index.md":
        return "Home"
    words = "-".join(relative_path.with_suffix("").parts).replace("_", "-").split("-")
    return "-".join(word[:1].upper() + word[1:] for word in words if word)


def discover_pages(docs_root: Path) -> dict[PurePosixPath, str]:
    """All Markdown files under docs/, with their Wiki names. Duplicate names are an error."""
    pages: dict[PurePosixPath, str] = {}
    seen: dict[str, PurePosixPath] = {}
    for source in sorted(docs_root.rglob("*.md")):
        relative = PurePosixPath(source.relative_to(docs_root).as_posix())
        slug = page_slug(relative)
        if slug in seen:
            raise ValueError(f"Wiki page name {slug!r} is shared by {seen[slug]} and {relative}")
        pages[relative] = slug
        seen[slug] = relative
    return pages


def _normalize(path: PurePosixPath) -> PurePosixPath:
    parts: list[str] = []
    for part in path.parts:
        if part in ("", "."):
            continue
        if part == "..":
            if parts:
                parts.pop()
            continue
        parts.append(part)
    return PurePosixPath(*parts)


def rewrite_links(markdown: str, source_in_repo: PurePosixPath, pages: dict[PurePosixPath, str],
                  repo_root: Path) -> str:
    """Rewrite relative links. `source_in_repo` is the file the text comes from, relative to the repo root."""

    def replace(match: re.Match[str]) -> str:
        label, destination = match.groups()
        if destination.startswith(("http://", "https://", "mailto:", "#")):
            return match.group(0)
        target, sep, anchor = destination.partition("#")
        resolved = _normalize(source_in_repo.parent / target)
        if resolved.parts and resolved.parts[0] == "docs":
            in_docs = PurePosixPath(*resolved.parts[1:])
            if in_docs in pages:
                return f"[{label}]({pages[in_docs]}{'#' + anchor if sep else ''})"
        if (repo_root / resolved.as_posix()).exists():
            kind = "tree" if (repo_root / resolved.as_posix()).is_dir() else "blob"
            return f"[{label}]({REPOSITORY_URL}/{kind}/{BRANCH}/{resolved.as_posix()}{'#' + anchor if sep else ''})"
        return match.group(0)

    return MARKDOWN_LINK.sub(replace, markdown)


def convert_admonitions(markdown: str) -> str:
    """!!! kind "Title" + 4-space indented body -> a block quote with a bold title."""
    lines = markdown.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        m = ADMONITION.match(lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue
        title = m.group(3) or m.group(2).capitalize()
        out.append(f"> **{title}**")
        out.append(">")
        i += 1
        while i < len(lines) and (lines[i].startswith("    ") or lines[i].strip() == ""):
            if lines[i].strip() == "" and (i + 1 >= len(lines) or not lines[i + 1].startswith("    ")):
                break
            body = lines[i][4:] if lines[i].startswith("    ") else ""
            out.append(f"> {body}".rstrip())
            i += 1
    return "\n".join(out)


def render_page(relative: PurePosixPath, docs_root: Path, repo_root: Path, pages: dict[PurePosixPath, str]) -> str:
    source_in_repo = PurePosixPath("docs") / relative
    text = (docs_root / relative.as_posix()).read_text(encoding="utf-8")

    def include(match: re.Match[str]) -> str:
        included = PurePosixPath(match.group(1))
        content = (repo_root / included.as_posix()).read_text(encoding="utf-8")
        return rewrite_links(content, included, pages, repo_root)

    text = rewrite_links(text, source_in_repo, pages, repo_root)
    text = SNIPPET.sub(lambda m: include(m), text)
    return convert_admonitions(text).rstrip() + "\n"


def _nav_entries(nav, group: str | None = None) -> list[tuple[str | None, str, str]]:
    """Flatten the mkdocs nav into (group, title, docs-relative path)."""
    out = []
    for entry in nav:
        if isinstance(entry, str):
            out.append((group, entry, entry))
            continue
        for title, value in entry.items():
            if isinstance(value, list):
                out.extend(_nav_entries(value, title))
            else:
                out.append((group, title, value))
    return out


def sidebar(nav_entries, pages: dict[PurePosixPath, str]) -> str:
    lines: list[str] = []
    current = object()
    for group, title, path in nav_entries:
        slug = pages.get(PurePosixPath(path))
        if slug is None:
            continue
        if group != current:
            if group is not None:
                lines.extend(["", f"**{group}**", ""])
            current = group
        lines.append(f"- [{HOME_TITLE if slug == 'Home' else title}]({slug})")
    return "\n".join(lines).strip() + "\n"


def build_wiki(docs_root: str | Path = "docs", output_directory: str | Path = "wiki-build",
               mkdocs_file: str | Path = "mkdocs.yml") -> list[Path]:
    docs = Path(docs_root).resolve()
    repo_root = docs.parent
    output = Path(output_directory).resolve()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    pages = discover_pages(docs)
    written = []
    for relative, slug in pages.items():
        dest = output / f"{slug}.md"
        dest.write_text(render_page(relative, docs, repo_root, pages), encoding="utf-8")
        written.append(dest)
    config = yaml.load(Path(mkdocs_file).read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    (output / "_Sidebar.md").write_text(sidebar(_nav_entries(config.get("nav", [])), pages), encoding="utf-8")
    (output / "_Footer.md").write_text(
        f"Generated from the [`codebook-annotator` repository]({REPOSITORY_URL}). "
        "Edit `docs/` in the repository, not the Wiki.\n", encoding="utf-8")
    return [*written, output / "_Sidebar.md", output / "_Footer.md"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the GitHub Wiki pages from docs/.")
    parser.add_argument("docs_root", nargs="?", default="docs")
    parser.add_argument("output_directory", nargs="?", default="wiki-build")
    args = parser.parse_args(argv)
    written = build_wiki(args.docs_root, args.output_directory)
    print(f"Built {len(written)} Wiki files in {Path(args.output_directory).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
