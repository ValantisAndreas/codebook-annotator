"""Disagreement exports.

- ``annotator_disagreement_csv``: per item and category, whether annotators disagreed
- ``run_disagreement_csv``: items where runs disagree with each other, with the
  majority labels, or fall outside the annotators' range.
"""

from __future__ import annotations

import csv
from datetime import UTC, datetime
from pathlib import Path

from .io import (
    DEFAULT_FIELDS,
    EvalDataError,
    GoldFields,
    check_gold,
    id_sort_key,
    load_jsonl_indexed,
    load_run_predictions,
    read_jsonl_records,
)

# Separator between annotators in the labels_per_annotator column (a semicolon and a space).
ANNOTATOR_SEPARATOR = "\x3b "


def annotator_categories(records: list[dict], fields: GoldFields) -> list[str]:
    cats: set[str] = set()
    for rec in records:
        for labs in (rec.get(fields.annotators_field) or {}).values():
            cats.update(labs)
    return sorted(cats)


def annotator_disagreement_csv(
    dataset_path: Path | str,
    output_csv: Path | str,
    filter_split: str | None = "test",
    *,
    categories: list[str] | None = None,
    split_field: str = "split",
    fields: GoldFields = DEFAULT_FIELDS,
) -> list[dict]:
    """Disagreement = True iff >= 1 annotator assigned the category and >= 1 annotator of the item did not.

    ``filter_split``: keep records whose ``split_field`` equals it. ``"all"`` or None keeps all.
    If the filter leaves nothing (e.g. a sample file without a split field), all records are used.
    Categories default to the sorted union of annotator labels.
    """
    dataset_path, output_csv = Path(dataset_path), Path(output_csv)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

    all_records = read_jsonl_records(dataset_path, fields.id_field)
    if not all_records:
        raise EvalDataError(f"no readable records with the id field '{fields.id_field}' in {dataset_path}. "
                            "Check the file and --id-field.")
    records = [rec for rec in all_records
               if filter_split is None or rec.get(split_field) == filter_split or filter_split == "all"]
    if not records:
        print(f"Note: no records with {split_field} == {filter_split!r}. Using all records.")
        records = list(all_records)

    order = id_sort_key([rec[fields.id_field] for rec in records])
    records.sort(key=lambda x: order(str(x[fields.id_field])))
    cats = list(categories) if categories else annotator_categories(records, fields)

    now_utc = datetime.now(UTC).isoformat()
    header_comments = [
        f"# Annotator disagreement audit for {dataset_path.name} (n={len(records)}).",
        "# Rule: disagreement = True iff at least one annotator assigned the category AND at least one annotator who labelled the item did not.",
        f"# Generated: {now_utc}",
    ]

    cols = ["item_id", "n_annotators"]
    cols += [f"disagree_{c.replace(' ', '_')}" for c in cats]
    cols.append("any_disagreement")
    cols += [f"votes_{c.replace(' ', '_')}" for c in cats]

    rows = []
    for rec in records:
        labels_per_ann = rec.get(fields.annotators_field) or {}
        row = {"item_id": rec[fields.id_field], "n_annotators": len(labels_per_ann)}
        any_disagree = False
        for cat in cats:
            safe_cat = cat.replace(" ", "_")
            votes = [1 if cat in ann_labels else 0 for ann_labels in labels_per_ann.values()]
            disagreed = (1 in votes) and (0 in votes)
            any_disagree = any_disagree or disagreed
            row[f"disagree_{safe_cat}"] = disagreed
            row[f"votes_{safe_cat}"] = "|".join(str(v) for v in votes)
        row["any_disagreement"] = any_disagree
        rows.append(row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        for comment in header_comments:
            f.write(comment + "\n")
        w = csv.writer(f, lineterminator="\n")  # quotes values that contain commas or quotes
        w.writerow(cols)
        for r in rows:
            w.writerow([str(r[c]) for c in cols])

    print(f"✓ Annotator disagreement file written to: {output_csv}")
    print(f"  Total items: {len(rows)}")
    disagree_count = sum(1 for r in rows if r["any_disagreement"])
    print(f"  Items with at least one disagreement: {disagree_count}/{len(rows)} ({100*disagree_count/max(1, len(rows)):.1f}%)")
    return rows


def run_disagreement_csv(
    dataset_path: Path | str,
    run_paths: list[str],
    output_csv: Path | str,
    *,
    key: str = "codebook_id",
    categories: list[str] | None = None,
    item_fields: list[str] | None = None,
    text_field: str = "text",
    empty_label: str = "(none)",
    fields: GoldFields = DEFAULT_FIELDS,
) -> list[dict]:
    """Item-level audit of disagreements between runs and with human annotations.

    One ``pred_<key>``/``reasoning_<key>`` column pair per run, keyed by the runs'
    ``codebook_id`` (default) or by ``run_id`` (file stem). Runs sharing a key collapse to
    the last one given, with a printed warning. An item is exported if any two runs
    differ, any run differs from the majority, or a run predicts a category no
    annotator chose / omits one every annotator chose. With explicit ``categories``,
    predictions and majority labels are restricted to those categories.
    """
    if not Path(dataset_path).exists():
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")
    dataset = load_jsonl_indexed(dataset_path, fields.id_field)
    check_gold(dataset, dataset_path, fields)

    run_file_map: dict[str, Path] = {}
    collisions: dict[str, list[str]] = {}
    for r_path in run_paths:
        p = Path(r_path)
        if not p.exists():
            print(f"Warning: run file not found: {p} — skipping")
            continue
        if key == "run_id":
            col = p.stem
        else:
            first = next(iter(read_jsonl_records(p, "item_id")), {})
            col = (first.get("run") or {}).get("codebook_id") or p.stem
        if col in run_file_map:
            collisions.setdefault(col, [str(run_file_map[col])]).append(str(p))
        run_file_map[col] = p
    if not run_file_map:
        raise ValueError("No valid run files provided.")
    for col, paths in collisions.items():
        print(f"Warning: {len(paths)} runs share the column key '{col}' and collapse to the last one "
              f"({paths[-1]}): {', '.join(paths)}. Use --key run_id to keep them apart.")

    loaded_runs = {k: preds for k, path in run_file_map.items() if (preds := load_run_predictions(path))}
    if not loaded_runs:
        raise ValueError("Failed to load predictions from any provided run file.")

    cb_ids = sorted(loaded_runs)
    restrict = set(categories) if categories else None
    if categories:
        all_categories = list(categories)
    else:
        all_categories = sorted({c for preds in loaded_runs.values() for rec in preds.values() for c in rec.get("predictions", {})})

    evaluated_ids = set(loaded_runs[cb_ids[0]])
    for cb_id in cb_ids[1:]:
        evaluated_ids.intersection_update(loaded_runs[cb_id])

    checked_ids = evaluated_ids & set(dataset)
    annot_empty = empty_label.strip("()") or empty_label
    item_fields = list(item_fields or [])
    disagreements = []
    for item_id in sorted(checked_ids):
        gold_item = dataset[item_id]
        gold_maj = set(gold_item.get(fields.majority_field) or [])
        if restrict is not None:
            gold_maj &= restrict
        labels_per_annot = gold_item.get(fields.annotators_field) or {}

        run_actives = {}
        for cb_id in cb_ids:
            preds_dict = loaded_runs[cb_id].get(item_id, {}).get("predictions", {})
            run_actives[cb_id] = {cat for cat, val in preds_dict.items() if val and (restrict is None or cat in restrict)}
        active_sets = list(run_actives.values())

        has_cb_disagreement = any(
            active_sets[i] != active_sets[j] for i in range(len(active_sets)) for j in range(i + 1, len(active_sets))
        )
        has_consensus_disagreement = any(act != gold_maj for act in active_sets)

        outside_human = False
        category_votes = {}
        total_annotators = len(labels_per_annot)
        for cat in all_categories:
            votes = sum(1 if cat in annot_labels else 0 for annot_labels in labels_per_annot.values())
            category_votes[cat] = votes
            if total_annotators > 0:
                if votes == 0 and any(cat in act for act in active_sets):
                    outside_human = True
                if votes == total_annotators and any(cat not in act for act in active_sets):
                    outside_human = True

        if not (has_cb_disagreement or has_consensus_disagreement or outside_human):
            continue

        votes_str_list = [f"{cat}: {v}/{total_annotators}" for cat, v in sorted(category_votes.items()) if v > 0]
        summary_votes = ", ".join(votes_str_list) if votes_str_list else empty_label
        annot_details = []
        for annot_id, cat_list in sorted(labels_per_annot.items()):
            cats_str = ", ".join(cat_list) if cat_list else annot_empty
            annot_details.append(f"{annot_id}: [{cats_str}]")
        full_labels_annot_str = f"{summary_votes} | {ANNOTATOR_SEPARATOR.join(annot_details)}" if labels_per_annot else ""

        row = {"item_id": item_id, "text": gold_item.get(text_field, "")}
        for fld in item_fields:
            row[fld] = gold_item.get(fld, "")
        row["gold_majority"] = ", ".join(sorted(gold_maj)) if gold_maj else empty_label
        row["labels_per_annotator"] = full_labels_annot_str
        for cb_id in cb_ids:
            act = run_actives[cb_id]
            row[f"pred_{cb_id}"] = ", ".join(sorted(act)) if act else empty_label
            row[f"reasoning_{cb_id}"] = loaded_runs[cb_id].get(item_id, {}).get("reasoning", "")
        disagreements.append(row)

    order = id_sort_key(checked_ids)

    def sort_key(item):
        preds = [item[f"pred_{cb}"] for cb in cb_ids]
        return (not len(set(preds)) > 1, order(item["item_id"]))

    disagreements.sort(key=sort_key)

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["item_id", "text", *item_fields, "gold_majority", "labels_per_annotator"]
    fieldnames += [f"pred_{cb}" for cb in cb_ids] + [f"reasoning_{cb}" for cb in cb_ids]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for d in disagreements:
            writer.writerow(d)

    print(f"✓ Consolidated disagreement CSV exported to: {output_path}")
    print(f"  Runs included ({len(cb_ids)}): {', '.join(cb_ids)}")
    print(f"  Total items checked (in the gold file and in every run): {len(checked_ids)}")
    print(f"  Disagreement cases: {len(disagreements)} ({len(disagreements)/max(1, len(checked_ids)):.1%})")
    return disagreements
