"""CLI subcommands: evaluate, bootstrap, disagreements, annotator-disagreements.

``register(subparsers)`` adds them to an argparse subparsers object (used by the
main ``cba`` CLI). ``python -m codebook_annotator.evaluate.cli`` runs them standalone.
"""

from __future__ import annotations

import argparse
import sys


def _add_gold_field_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("gold data fields")
    g.add_argument("--id-field", default="item_id", help="Item id field in the gold data (default: item_id)")
    g.add_argument("--majority-field", default="label_majority", help="Field with the majority/consensus label list (default: label_majority)")
    g.add_argument("--annotators-field", default="labels_per_annotator",
                   help="Field with {annotator: [labels]} (default: labels_per_annotator). If absent, annotator-based metrics are skipped.")


def _add_category_args(p: argparse.ArgumentParser, default_help: str) -> None:
    p.add_argument("--categories", nargs="+", default=None, help=f"Category list and order. {default_help}")
    p.add_argument("--codebook", default=None, help="Take the category list and order from this codebook YAML (file order)")


def _fields(args):
    from .io import GoldFields

    return GoldFields(args.id_field, args.majority_field, args.annotators_field)


def _clean_errors(func):
    """Turn input problems into an 'Error: ...' line and exit code 2 instead of a traceback."""
    import functools

    @functools.wraps(func)
    def wrapper(args) -> int:
        import yaml

        from .io import EvalDataError
        try:
            return func(args)
        except (EvalDataError, ValueError, yaml.YAMLError) as e:
            print(f"Error: {e}", file=sys.stderr)
            return 2
        except OSError as e:
            print(f"Error: {e.strerror or e}: {e.filename}" if e.filename else f"Error: {e}", file=sys.stderr)
            return 2
    return wrapper


# ── evaluate ─────────────────────────────────────────────────────────────────

def _run_files(paths: list[str]) -> list[str]:
    """Drop <run_id>.attempts.jsonl files (earlier failed attempts, not runs), with a note."""
    kept = [p for p in paths if not str(p).endswith(".attempts.jsonl")]
    skipped = len(paths) - len(kept)
    if skipped:
        print(f"Note: skipped {skipped} .attempts.jsonl file(s). They hold earlier failed attempts, not runs.")
    return kept


@_clean_errors
def cmd_evaluate(args) -> int:
    from pathlib import Path

    from .io import load_dataset, parse_bool_labels, resolve_categories
    from .report import EvalOptions, build_report, load_runs, write_report

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: dataset not found at: {dataset_path}")
        return 1
    fields = _fields(args)
    print("Loading dataset...")
    dataset = load_dataset(dataset_path, fields)
    print(f"Loaded {len(dataset)} dataset items.")
    loaded = load_runs(_run_files(args.runs), dataset)
    if not loaded:
        print("Error: No valid prediction runs to evaluate.")
        return 1
    try:
        bool_labels = parse_bool_labels(args.bool_labels)
    except ValueError as e:
        print(f"Error: {e}")
        return 2
    cats = resolve_categories(args.categories, args.codebook, [r["predictions"] for r in loaded])
    opts = EvalOptions(
        categories=cats,
        fields=fields,
        slice_by=args.slice_by or [],
        agreement_slice_by=args.agreement_slice_by,
        bool_labels=bool_labels,
        sheet1_slice_field=args.sheet1_slice_field,
        min_cell_floor=args.min_cell_floor,
        codebooks=args.codebooks,
        codebooks_dir=args.codebooks_dir,
        gold_scores_field=args.gold_scores_field,
    )
    sheets = build_report(dataset, loaded, opts)
    write_report(sheets, args.output_xlsx, args.merge, {r["run_id"] for r in loaded})
    return 0


def _register_evaluate(sub) -> None:
    p = sub.add_parser("evaluate", help="Evaluation workbook: F1/Jaccard/Hamming, Krippendorff alpha, slices, agreement",
                       description="Compare runs with gold labels and write an Excel workbook.")
    p.add_argument("--dataset", "--gold", dest="dataset", required=True, help="Gold dataset JSONL")
    p.add_argument("--runs", nargs="+", required=True, help="Run JSONL files")
    p.add_argument("--output-xlsx", "-o", dest="output_xlsx", default="results.xlsx", help="Output workbook (default: results.xlsx)")
    _add_category_args(p, "Default: sorted union of the runs' prediction keys.")
    p.add_argument("--slice-by", action="append", default=None, metavar="FIELD",
                   help="Gold field to slice by, one '<Field> Slices' sheet each (repeatable)")
    p.add_argument("--agreement-slice-by", action="append", default=None, metavar="FIELD",
                   help="Gold fields for Codebook Agreement slices (repeatable, default: the --slice-by fields)")
    p.add_argument("--bool-labels", action="append", default=None, metavar="FIELD=TRUE,FALSE",
                   help="Render a field by truthiness with these labels (a missing value counts as false)")
    p.add_argument("--sheet1-slice-field", default=None, metavar="FIELD",
                   help="Also report Sheet1 rows per value of this field (default: pooled only)")
    p.add_argument("--min-cell-floor", type=int, default=5, help="Minimum item count for reporting metrics in a cell (default: 5)")
    p.add_argument("--codebooks", nargs="*", default=[], help="Codebook YAMLs for the Token Counts sheet")
    p.add_argument("--codebooks-dir", default="codebooks",
                   help="Fallback for Token Counts: look for <dir>/<codebook_id>.yaml (default: ./codebooks)")
    p.add_argument("--gold-scores-field", default=None,
                   help="Gold field with {category: score|null}, for runs with ordinal scores (Ordinal sheet)")
    p.add_argument("--merge", action="store_true", help="Merge into an existing workbook, replacing rows of re-evaluated runs")
    _add_gold_field_args(p)
    p.set_defaults(func=cmd_evaluate)


# ── bootstrap ────────────────────────────────────────────────────────────────

@_clean_errors
def cmd_bootstrap(args) -> int:
    from .bootstrap import run_bootstrap, save_csv, save_npz
    from .io import categories_from_codebook

    args.runs = _run_files(args.runs)
    if len(args.runs) < 1:
        print("Error: give at least one run (the first is the baseline).")
        return 2
    cats = args.categories or (categories_from_codebook(args.codebook) if args.codebook else None)
    result = run_bootstrap(args.dataset, args.runs, resamples=args.resamples, seed=args.seed,
                           categories=cats, fields=_fields(args))
    print("\n".join(result.lines))
    if args.save_npz:
        save_npz(result, args.save_npz)
        print(f"\nreplicates saved: {args.save_npz}")
    if args.output_csv:
        save_csv(result, args.output_csv)
        print(f"table saved: {args.output_csv}")
    return 0


def _register_bootstrap(sub) -> None:
    p = sub.add_parser("bootstrap", help="Paired bootstrap CIs and p-values vs a baseline run (incl. out-of-range labels)",
                       description="Paired bootstrap confidence intervals and p-values against the first run.")
    p.add_argument("--dataset", "--gold", dest="dataset", required=True, help="Gold dataset/sample JSONL")
    p.add_argument("--runs", nargs="+", required=True, help="Run JSONL files. The FIRST is the baseline")
    p.add_argument("--resamples", type=int, default=5000, help="Bootstrap resamples (default: 5000)")
    p.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    _add_category_args(p, "Default: sorted union of the runs' prediction keys.")
    p.add_argument("--save-npz", default=None, help="Save replicates and point estimates to this .npz")
    p.add_argument("--output-csv", default=None, help="Save a tidy results table to this CSV")
    _add_gold_field_args(p)
    p.set_defaults(func=cmd_bootstrap)


# ── disagreements ────────────────────────────────────────────────────────────

@_clean_errors
def cmd_disagreements(args) -> int:
    from .disagreements import run_disagreement_csv
    from .io import categories_from_codebook

    cats = args.categories or (categories_from_codebook(args.codebook) if args.codebook else None)
    args.runs = _run_files(args.runs)
    try:
        run_disagreement_csv(args.dataset, args.runs, args.output_csv, key=args.key, categories=cats,
                             item_fields=args.item_field, text_field=args.text_field,
                             empty_label=args.empty_label, fields=_fields(args))
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    return 0


def _register_disagreements(sub) -> None:
    p = sub.add_parser("disagreements", help="Item-level audit CSV of disagreements between runs and with annotators",
                       description="Export the items where runs disagree with each other or with the annotators.")
    p.add_argument("--dataset", "--gold", dest="dataset", required=True, help="Gold dataset JSONL")
    p.add_argument("--runs", nargs="+", required=True, help="Run JSONL files")
    p.add_argument("--output-csv", "-o", dest="output_csv", required=True, help="Output CSV")
    p.add_argument("--key", choices=["codebook_id", "run_id"], default="codebook_id",
                   help="Column key per run (default: codebook_id, runs sharing one collapse to the last)")
    _add_category_args(p, "Default: sorted union of the runs' prediction keys.")
    p.add_argument("--item-field", action="append", default=None, metavar="FIELD",
                   help="Extra gold field to copy into the CSV after 'text' (repeatable)")
    p.add_argument("--text-field", default="text", help="Gold text field (default: text)")
    p.add_argument("--empty-label", default="(none)", help="Display string for an empty label set (default: '(none)')")
    _add_gold_field_args(p)
    p.set_defaults(func=cmd_disagreements)


@_clean_errors
def cmd_annotator_disagreements(args) -> int:
    from .disagreements import annotator_disagreement_csv
    from .io import categories_from_codebook

    cats = args.categories or (categories_from_codebook(args.codebook) if args.codebook else None)
    try:
        annotator_disagreement_csv(args.dataset, args.output_csv, args.split, categories=cats,
                                   split_field=args.split_field, fields=_fields(args))
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    return 0


def _register_annotator_disagreements(sub) -> None:
    p = sub.add_parser("annotator-disagreements", help="Per item x category annotator disagreement matrix CSV",
                       description="Export, per item and category, whether the human annotators disagreed.")
    p.add_argument("--dataset", "--gold", dest="dataset", required=True, help="Gold dataset JSONL")
    p.add_argument("--output-csv", "-o", dest="output_csv", required=True, help="Output CSV")
    p.add_argument("--split", default="test",
                   help="Keep records whose split field equals this. 'all' keeps every record (default: test, "
                        "falls back to all records if nothing matches)")
    p.add_argument("--split-field", default="split", help="Split field name (default: split)")
    _add_category_args(p, "Default: sorted union of annotator labels.")
    _add_gold_field_args(p)
    p.set_defaults(func=cmd_annotator_disagreements)


def register(subparsers) -> None:
    """Add the evaluation subcommands to an argparse subparsers object."""
    _register_evaluate(subparsers)
    _register_bootstrap(subparsers)
    _register_disagreements(subparsers)
    _register_annotator_disagreements(subparsers)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m codebook_annotator.evaluate.cli", description="Evaluation tools")
    sub = parser.add_subparsers(dest="command", required=True)
    register(sub)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
