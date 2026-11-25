"""Command-line interface: `cba <command> ...`."""

from __future__ import annotations

import argparse
import json
import sys
import typing
from pathlib import Path

import yaml

from . import __version__
from .codebook import CodebookError, load_codebook
from .config import ConfigError, RunConfig, build_config
from .data import DataError
from .prompt import PromptError


def _print_issues(cb) -> None:
    for i in cb.issues:
        print(f"  {i}")


# ── validate / render / templates ──────────────────────────────────────────────────────────────

def cmd_validate(args) -> int:
    rc = 0
    for p in args.codebooks:
        try:
            cb = load_codebook(p, strict=False)
        except (CodebookError, OSError, yaml.YAMLError) as e:
            print(f"✖ {p}: {e}")
            rc = 1
            continue
        status = "✖" if cb.errors else "✓"
        print(f"{status} {p}: {len(cb.categories)} categories, output={cb.output_type}, "
              f"{len(cb.examples)} examples, {len(cb.errors)} error(s), {len(cb.warnings)} warning(s)")
        _print_issues(cb)
        if cb.errors:
            rc = 1
        elif not cb.prompt_overrides and args.template == "default_v1":
            print("  [NOTE] no `prompt:` block: the generic default_v1 wording will be used "
                  "(consider setting prompt.role / prompt.task for your domain)")
    return rc


def cmd_render(args) -> int:
    from .prompt import get_user_prompt, render_prompts
    cb = load_codebook(args.codebook)
    rp = render_prompts(cb, args.template, few_shot=args.few_shot, evidence=args.evidence)
    print(f"# template={rp.template} prompt_condition={rp.prompt_condition}")
    print("# ───────── system prompt ─────────")
    print(rp.system)
    if args.text is not None:
        print("# ───────── user prompt ─────────")
        print(get_user_prompt(args.text))
    return 0


def cmd_templates(args) -> int:
    from .prompt import available_templates, load_template
    for name in available_templates():
        t = load_template(name)
        print(f"{name}:")
        for k in ("role", "task", "rules_heading", "reasoning_example"):
            print(f"  {k}: {t[k]}")
    return 0


# ── convert / ablate / check-invariance ────────────────────────────────────────────────────────

def _kv_list(items: list[str] | None) -> dict[str, str]:
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"expected FIELD=VALUE, got {it!r}")
        k, v = it.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def cmd_convert(args) -> int:
    from .codebook_tools import dump_codebook, table_to_codebook, token_report
    cats = [c.strip() for c in args.categories.split(",")] if args.categories else None
    try:
        cb = table_to_codebook(args.input, category_column=args.category_column, sheet=args.sheet,
                               columns=args.columns.split(",") if args.columns else None,
                               labels=_kv_list(args.label), list_columns=set(args.list_column or []),
                               list_sep=args.list_sep, expected_categories=cats, allow_empty=args.allow_empty,
                               add_name=args.add_name, codebook_id=args.id, version=args.version, author=args.author)
    except CodebookError as e:
        print(f"✖ {e}", file=sys.stderr)
        return 1
    dump_codebook(cb, args.output)
    print(f"✓ Codebook YAML written to: {args.output}  ({len(cb['categories'])} categories)")
    rep = token_report(cb)
    for k, v in rep.items():
        print(f"  - {k}: {v} estimated tokens")
    print(f"Total estimated rendered tokens: {sum(rep.values())}")
    return 0


def cmd_ablate(args) -> int:
    from .codebook_tools import build_ablations, dump_codebook, load_extension_table
    base = yaml.safe_load(Path(args.base).read_text(encoding="utf-8"))
    if not isinstance(base, dict) or not isinstance(base.get("categories"), dict) or not base["categories"]:
        raise CodebookError(f"{args.base}: the base codebook needs a 'categories' mapping")
    ext = load_extension_table(args.extension, category_column=args.category_column, sheet=args.sheet)
    labels = _kv_list(args.label)
    from .codebook_tools import default_label
    fields = [(f, labels.get(f, default_label(f))) for f in args.fields.split(",")]
    missing = [f for f, _ in fields if ext and f not in next(iter(ext.values()))]
    if missing:
        print(f"✖ fields not found in {args.extension}: {', '.join(missing)}", file=sys.stderr)
        return 1
    try:
        variants = build_ablations(base, ext, fields, names=_kv_list(args.name), prefix=args.prefix,
                                   full_name=args.full_name, single=not args.no_single, full=not args.no_full,
                                   base_name=Path(args.base).name, author=args.author)
    except CodebookError as e:
        print(f"✖ {e}", file=sys.stderr)
        return 1
    out_dir = Path(args.out_dir)
    for name, cb in variants.items():
        dump_codebook(cb, out_dir / f"{name}.yaml")
        print(f"✓ Created: {out_dir / (name + '.yaml')}")
    print(f"✓ Base fields verified verbatim in all {len(variants)} variants")
    return 0


def cmd_check_invariance(args) -> int:
    from .codebook_tools import check_invariance
    cbs = [load_codebook(p) for p in args.codebooks]
    rep = check_invariance(cbs, template=args.template, few_shot=args.few_shot, evidence=args.evidence,
                           names=args.codebooks)
    print(f"Reference: {rep.reference}")
    for r in rep.results:
        mark = "✓" if r["ok"] else "✖"
        print(f"{mark} {r['codebook']}: preamble identical={r['preamble_identical']}, "
              f"output section identical={r['output_section_identical']}, "
              f"codebook block differs={r['codebook_block_differs']}")
        if r["diff"]:
            print(r["diff"])
    print("✓ Prompts differ only in the codebook block" if rep.ok else "✖ Invariance check FAILED")
    return 0 if rep.ok else 1


# ── run ────────────────────────────────────────────────────────────────────────────────────────

def _add_config_options(p: argparse.ArgumentParser) -> None:
    """One --option per RunConfig field, default None (= not given, so config files decide)."""
    g = p.add_argument_group("run settings (override config files)")
    for name, f in RunConfig.model_fields.items():
        flag = "--" + name.replace("_", "-")
        ann = f.annotation
        if typing.get_origin(ann) is typing.Literal:
            base = ann
        else:  # unwrap Optional[X] / X | None
            args = typing.get_args(ann)
            base = next((a for a in args if a is not type(None)), ann) if args else ann
        help_ = f.description or ""
        if base is bool:
            g.add_argument(flag, dest=name, action=argparse.BooleanOptionalAction, default=None, help=help_)
        elif typing.get_origin(base) is typing.Literal:
            g.add_argument(flag, dest=name, choices=typing.get_args(base), default=None, help=help_)
        elif base is int:
            g.add_argument(flag, dest=name, type=int, default=None, help=help_)
        elif base is float:
            g.add_argument(flag, dest=name, type=float, default=None, help=help_)
        else:
            g.add_argument(flag, dest=name, default=None, help=help_)


def _config_from_args(args) -> RunConfig:
    overrides = {k: getattr(args, k) for k in RunConfig.model_fields if getattr(args, k, None) is not None}
    return build_config(args.config or [], overrides)


def cmd_run(args) -> int:
    from .runner import RunError, prepare, run
    try:
        cfg = _config_from_args(args)
    except ConfigError as e:
        print(f"✖ {e}", file=sys.stderr)
        return 2
    if args.print_config:
        print(yaml.safe_dump(json.loads(cfg.model_dump_json()), sort_keys=False, allow_unicode=True))
        return 0
    try:
        if args.dry_run:
            from .providers.adapters import StubProvider
            ctx, sample, rp = prepare(cfg, provider=StubProvider(categories=[], model_tag=cfg.model_tag),
                                      dry_run=True)
            print(f"Dry run (no model calls). Run '{cfg.run_id}': {len(sample)} items -> {cfg.output_file}")
            print(f"  codebook: {cfg.codebook} ({len(ctx.categories)} categories: {', '.join(ctx.categories)})")
            print(f"  provider: {cfg.provider}  model: {cfg.model_tag}  template: {rp.template}  "
                  f"condition: {ctx.run_meta['prompt_condition']}")
            print(f"  sample_id: {ctx.run_meta['sample_id']}  config_sha256: {ctx.run_meta['config_sha256'][:16]}…")
            if cfg.model_digest == "auto":
                print("  (model_digest 'auto' is not resolved in a dry run, so the real run's config_sha256 differs)")
            print("  Not checked in a dry run: the model server, the model and the API key.")
            if sample:
                print(f"  first item: {sample[0]['item_id']}: {sample[0]['text'][:80]!r}")
            if args.show_prompt:
                print("# ───────── system prompt ─────────")
                print(ctx.system_prompt)
            return 0
        summary = run(cfg)
    except RunError as e:
        print(f"✖ {e}", file=sys.stderr)
        return 2
    return summary.exit_code


# ── data utilities ─────────────────────────────────────────────────────────────────────────────

def cmd_aggregate(args) -> int:
    from .data import load_records
    from .data_tools import aggregate_annotations, write_jsonl
    rows = load_records(args.input, sheet=args.sheet)
    label_map = None
    if args.labels:
        label_map = {lab.strip(): lab.strip() for lab in args.labels.split(",")}
        label_map.update(_kv_list(args.map))
    elif args.map:
        raise SystemExit("--map needs --labels (the full list of accepted labels)")
    recs = aggregate_annotations(rows, text_field=args.text_field, annotator_field=args.annotator_field,
                                 labels_field=args.labels_field, group_field=args.group_field,
                                 label_sep=args.label_sep, label_map=label_map,
                                 drop_labels=set(args.drop or []), threshold=args.threshold,
                                 id_prefix=args.id_prefix, keep_fields=args.keep or [])
    write_jsonl(recs, args.output)
    print(f"✓ {len(rows)} annotation rows -> {len(recs)} items written to {args.output}")
    return 0


def cmd_overlap(args) -> int:
    from .data import load_records
    from .data_tools import add_lexical_overlap, codebook_words, write_jsonl
    cb = load_codebook(args.codebook, strict=False)
    words = codebook_words(cb.data, args.field)
    recs = load_records(args.input)
    n = add_lexical_overlap(recs, words, text_field=args.text_field, out_field=args.out_field)
    write_jsonl(recs, args.output)
    print(f"✓ {n}/{len(recs)} items contain at least one of {len(words)} words from '{args.field}' -> {args.output}")
    return 0


def cmd_strip_text(args) -> int:
    from .data_tools import strip_run_text
    out_dir = Path(args.out_dir)
    clashes = [p for p in args.runs if (out_dir / Path(p).name).resolve() == Path(p).resolve()]
    if clashes:
        raise DataError(f"--out-dir {out_dir} would overwrite the input file(s) {', '.join(clashes)}. "
                        "Choose a different folder")
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in args.runs:
        n = strip_run_text(p, out_dir / Path(p).name)
        print(f"✓ {p} -> {out_dir / Path(p).name} ({n} records, text, raw_response and evidence quotes removed)")
    return 0


# ── parser ─────────────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="cba", description="Annotate text with LLMs from a YAML codebook.")
    ap.add_argument("--version", action="version", version=f"codebook-annotator {__version__}")
    sub = ap.add_subparsers(dest="command", required=True, metavar="<command>")

    p = sub.add_parser("init", help="create a starter codebook, items file and run config",
                       description="Create a starter codebook, items file and run config in a folder.")
    p.add_argument("folder", nargs="?", default=".", help="target folder (default: current folder)")
    p.add_argument("--force", action="store_true", help="overwrite existing files")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("validate", help="check codebook files",
                       description="Check codebook files and list errors and warnings.")
    p.add_argument("codebooks", nargs="+", help="codebook YAML file(s)")
    p.add_argument("--template", default="default_v1",
                   help="template you plan to use (only affects a hint, default: default_v1)")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("render", help="print the prompt a codebook produces",
                       description="Print the exact system prompt a codebook produces.")
    p.add_argument("codebook", help="codebook YAML file")
    p.add_argument("--template", default="default_v1", help="prompt template name or file (default: default_v1)")
    p.add_argument("--text", help="also print the user prompt for this text")
    p.add_argument("--few-shot", action="store_true", help="include the codebook's examples")
    p.add_argument("--evidence", action="store_true", help="ask for evidence quotes")
    p.set_defaults(func=cmd_render)

    p = sub.add_parser("templates", help="list built-in prompt templates",
                       description="List the built-in prompt templates and their wording.")
    p.set_defaults(func=cmd_templates)

    p = sub.add_parser("convert", help="spreadsheet/CSV (one row per category) -> codebook YAML",
                       description="Turn a table with one row per category into a codebook YAML. Every other "
                       "column becomes a field, rendered in column order.")
    p.add_argument("input", help="table file (.xlsx, .csv, .tsv, .jsonl or .json)")
    p.add_argument("-o", "--output", required=True, help="codebook YAML to write")
    p.add_argument("--sheet", help="sheet name for .xlsx input (default: first sheet)")
    p.add_argument("--category-column", help="column holding the category names (default: first column)")
    p.add_argument("--columns", help="comma-separated columns to use, in render order (default: all)")
    p.add_argument("--label", action="append", metavar="FIELD=LABEL", help="display label for a field")
    p.add_argument("--list-column", action="append", help="column holding a list (split on --list-sep)")
    p.add_argument("--list-sep", default=",", help="separator for list columns (default: ,)")
    p.add_argument("--categories", help="comma-separated expected category names (case-insensitive match)")
    p.add_argument("--allow-empty", action="store_true", help="allow empty cells (default: error)")
    p.add_argument("--add-name", action="store_true", help="add a rendered 'name' field = category key")
    p.add_argument("--id", help="codebook id to write into the file")
    p.add_argument("--version", dest="version", help="codebook version to write into the file")
    p.add_argument("--author", help="author recorded under provenance")
    p.set_defaults(func=cmd_convert)

    p = sub.add_parser("ablate", help="base codebook + extra fields -> one variant per field and/or all fields",
                       description="Build codebook variants: the base codebook plus one added field each, and the "
                       "base plus all added fields. Base fields are copied verbatim.")
    p.add_argument("--base", required=True, help="base codebook YAML (fields copied verbatim)")
    p.add_argument("--extension", required=True, help="table with one row per category holding the added fields")
    p.add_argument("--fields", required=True, help="comma-separated extension columns to add, in order")
    p.add_argument("--out-dir", required=True, help="folder for the variant YAML files")
    p.add_argument("--prefix", default="variant", help="file name prefix of the variants (default: variant)")
    p.add_argument("--name", action="append", metavar="FIELD=SHORT", help="short name for a field's variant")
    p.add_argument("--full-name", default="full", help="name of the all-fields variant (default: full)")
    p.add_argument("--label", action="append", metavar="FIELD=LABEL", help="display label for an added field")
    p.add_argument("--sheet", help="sheet name for .xlsx extension tables")
    p.add_argument("--category-column", help="extension column holding the category names (default: first column)")
    p.add_argument("--no-single", action="store_true", help="only build the all-fields variant")
    p.add_argument("--no-full", action="store_true", help="only build single-field variants")
    p.add_argument("--author", help="author recorded under provenance")
    p.set_defaults(func=cmd_ablate)

    p = sub.add_parser("check-invariance", help="check prompts differ only in the codebook block (first = reference)",
                       description="Check that the prompts of several codebooks differ only in the category "
                       "block. The first codebook is the reference the others are compared with.")
    p.add_argument("codebooks", nargs="+", help="two or more codebook YAML files")
    p.add_argument("--template", default="default_v1", help="prompt template (default: default_v1)")
    p.add_argument("--few-shot", action="store_true", help="render with the codebooks' examples")
    p.add_argument("--evidence", action="store_true", help="render with evidence quotes requested")
    p.set_defaults(func=cmd_check_invariance)

    p = sub.add_parser("run", help="annotate a dataset", description="Annotate a dataset. Settings come from "
                       "--config file(s) (JSON/YAML, later files override earlier) and options below (override files).")
    p.add_argument("-c", "--config", action="append", help="run config file (repeatable)")
    p.add_argument("--dry-run", action="store_true", help="validate, sample and render without calling the model")
    p.add_argument("--show-prompt", action="store_true", help="with --dry-run: print the system prompt")
    p.add_argument("--print-config", action="store_true", help="print the merged configuration and exit")
    _add_config_options(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("aggregate", help="one row per annotation -> one item per text with per-annotator labels",
                       description="Group a table with one row per (text, annotator) into one item per text, "
                       "with labels_per_annotator and label_majority.")
    p.add_argument("input", help="annotation table (.jsonl, .json, .csv, .tsv or .xlsx)")
    p.add_argument("-o", "--output", required=True, help="items JSONL to write")
    p.add_argument("--sheet", help="sheet name for .xlsx input")
    p.add_argument("--text-field", default="text", help="text column (default: text)")
    p.add_argument("--annotator-field", default="annotator", help="annotator column (default: annotator)")
    p.add_argument("--labels-field", default="annotation", help="labels column (default: annotation)")
    p.add_argument("--group-field", help="group by this id column instead of by text")
    p.add_argument("--label-sep", default=",", help="separator between labels in one cell (default: ,)")
    p.add_argument("--labels", help="comma-separated accepted labels (others become UNKNOWN:<label>)")
    p.add_argument("--map", action="append", metavar="RAW=LABEL", help="map a raw label to an accepted one")
    p.add_argument("--drop", action="append", help="label to drop, e.g. a 'none' label (repeatable)")
    p.add_argument("--threshold", type=float, default=0.5, help="majority share (default 0.5 = at least half)")
    p.add_argument("--id-prefix", default="item_", help="prefix of the generated item ids (default: item_)")
    p.add_argument("--keep", action="append", help="item-level field to keep, taken from the first row (repeatable)")
    p.set_defaults(func=cmd_aggregate)

    p = sub.add_parser("overlap", help="flag items containing any word from a codebook field",
                       description="Mark each item whose text contains any word listed in a codebook field "
                       "(case-insensitive, whole words).")
    p.add_argument("input", help="items file")
    p.add_argument("-o", "--output", required=True, help="items JSONL to write")
    p.add_argument("--codebook", required=True, help="codebook YAML with the word lists")
    p.add_argument("--field", default="example_words", help="codebook field holding the words (default: example_words)")
    p.add_argument("--text-field", default="text", help="text field of the items (default: text)")
    p.add_argument("--out-field", default="lexical_overlap", help="name of the flag field (default: lexical_overlap)")
    p.set_defaults(func=cmd_overlap)

    p = sub.add_parser("strip-text", help="remove text, raw_response and evidence quotes from run files (for sharing)",
                       description="Copy run files without the item text, the raw model answers and the "
                                   "evidence quotes (their verified flags are kept).")
    p.add_argument("runs", nargs="+", help="run JSONL file(s)")
    p.add_argument("--out-dir", required=True, help="folder for the stripped copies")
    p.set_defaults(func=cmd_strip_text)

    try:
        from .evaluate.cli import register
    except ImportError:
        register = None
    if register is not None:
        register(sub)
    return ap


EVAL_EXTRA_MODULES = {"numpy", "pandas", "openpyxl"}


def cmd_init(args) -> int:
    from .scaffold import init_project
    try:
        written = init_project(args.folder, force=args.force)
    except FileExistsError as e:
        print(f"✖ {e}", file=sys.stderr)
        return 1
    for p in written:
        print(f"✓ Created: {p}")
    print(f"Next steps:\n  cd {args.folder}\n  cba validate codebook.yaml\n  cba run -c run.yaml --dry-run\n"
          "  cba run -c run.yaml\n  cba evaluate --dataset items.jsonl --runs runs/first_run.jsonl -o runs/results.xlsx")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (CodebookError, ConfigError, DataError, PromptError, yaml.YAMLError) as e:
        print(f"✖ {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted. Finished items are saved. Run the same command again to resume.", file=sys.stderr)
        return 130
    except OSError as e:  # e.g. a missing file
        print(f"✖ {e.strerror or e}: {e.filename}" if e.filename else f"✖ {e}", file=sys.stderr)
        return 2
    except ImportError as e:
        missing = (e.name or "").split(".")[0]
        if missing in EVAL_EXTRA_MODULES:
            print(f"✖ This command needs '{missing}', which comes with the eval extra. Install it from the\n"
                  "  codebook-annotator folder with:\n    pip install -e \".[eval]\"", file=sys.stderr)
            return 2
        raise


if __name__ == "__main__":
    sys.exit(main())
