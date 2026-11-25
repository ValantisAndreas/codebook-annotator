"""Evaluation workbook: classification + agreement metrics with configurable slicing.

Options:

- categories (``EvalOptions.categories``), default: the runs' prediction keys
- gold field names (``GoldFields``)
- slice sheets from ``slice_by`` fields
- codebook-agreement slices from ``agreement_slice_by``
- an optional Sheet1 slice field (default: none, i.e. ``pooled`` only)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .io import (
    GoldFields,
    annotators,
    has_annotators,
    is_evidence_run,
    is_ordinal_run,
    load_run_predictions,
    majority,
    slice_sheet_name,
    slice_value,
)
from .metrics import (
    calculate_classification_metrics,
    compute_krippendorff_alpha_multi,
    evidence_summary,
    ordinal_metrics,
    single_category_metrics,
)

DEFAULT_RENDER = [
    {"field": "name", "label": "Name"},
    {"field": "definition", "label": "Definition"},
    {"field": "example_item", "label": "Example Item"},
    {"field": "example_words", "label": "Illustrative Words"},
]



_TOOL_CONDITION = re.compile(r"^(zero_shot|few_shot)(\+[a-z_]+)*$")


def arm_key(meta: dict) -> str:
    """Comparison arm of a run: its codebook_id, qualified by the prompt condition when that is a
    non-default condition produced by this tool (e.g. 'B_full [few_shot+evidence]'), so runs that
    share a codebook but differ in prompt condition are kept apart. Plain 'zero_shot' runs (and
    any other condition label) keep the bare codebook_id."""
    cb = meta["codebook_id"]
    cond = meta.get("prompt_condition")
    if cond and cond != "zero_shot" and _TOOL_CONDITION.match(cond):
        return f"{cb} [{cond}]"
    return cb

@dataclass
class EvalOptions:
    categories: list[str]
    fields: GoldFields = field(default_factory=GoldFields)
    slice_by: list[str] = field(default_factory=list)
    agreement_slice_by: list[str] | None = None  # None -> same as slice_by
    bool_labels: dict[str, tuple[str, str]] = field(default_factory=dict)
    sheet1_slice_field: str | None = None
    min_cell_floor: int = 5
    codebooks: list[str] = field(default_factory=list)
    codebooks_dir: str = "codebooks"
    gold_scores_field: str | None = None

    @property
    def agreement_fields(self) -> list[str]:
        return list(self.slice_by if self.agreement_slice_by is None else self.agreement_slice_by)


def _m2a_alpha(subset_gold: list[dict], y_pred_col: np.ndarray, cat: str, fields: GoldFields) -> float:
    """Model-to-annotator alpha for one category over a list of gold items."""
    items_ratings = []
    for i, gold in enumerate(subset_gold):
        ratings = [1 if cat in labels else 0 for labels in annotators(gold, fields).values()]
        ratings.append(y_pred_col[i])
        items_ratings.append(ratings)
    return compute_krippendorff_alpha_multi(items_ratings)


def _matrices(aligned: list[tuple[dict, dict]], cats: list[str], fields: GoldFields) -> tuple[np.ndarray, np.ndarray]:
    n = len(aligned)
    y_true = np.zeros((n, len(cats)), dtype=int)
    y_pred = np.zeros((n, len(cats)), dtype=int)
    for idx, (gold, pred) in enumerate(aligned):
        gold_maj = majority(gold, fields)
        pred_preds = pred["predictions"]
        for class_idx, cat in enumerate(cats):
            y_true[idx, class_idx] = 1 if cat in gold_maj else 0
            y_pred[idx, class_idx] = 1 if pred_preds.get(cat, False) else 0
    return y_true, y_pred


def process_eval_run(run_predictions: dict[str, dict], dataset: dict[str, dict], opts: EvalOptions, use_annotators: bool):
    """Per-run overall, per-slice-field and per-category metrics."""
    cats = opts.categories
    aligned_items = [(dataset[iid], rec) for iid, rec in run_predictions.items() if iid in dataset]
    if not aligned_items:
        print("Warning: No aligned items found between dataset and predictions.")
        return {}, {}, {}

    y_true, y_pred = _matrices(aligned_items, cats, opts.fields)
    golds = [g for g, _ in aligned_items]

    overall = calculate_classification_metrics(y_true, y_pred)
    alpha_by_cat = {}
    for class_idx, cat in enumerate(cats):
        alpha_by_cat[cat] = _m2a_alpha(golds, y_pred[:, class_idx], cat, opts.fields) if use_annotators else None
    overall["model_to_annotator_agreement_alpha"] = (
        float(np.mean(list(alpha_by_cat.values()))) if use_annotators else None
    )

    slice_metrics: dict[str, dict[str, dict]] = {}
    for fld in opts.slice_by:
        values = [slice_value(g, fld, opts.bool_labels) for g in golds]
        per_value = {}
        for v in sorted(set(values)):
            idx_v = [i for i, val in enumerate(values) if val == v]
            m = calculate_classification_metrics(y_true[idx_v], y_pred[idx_v])
            if use_annotators:
                sub_gold = [golds[i] for i in idx_v]
                alphas = [_m2a_alpha(sub_gold, y_pred[idx_v, class_idx], cat, opts.fields) for class_idx, cat in enumerate(cats)]
                m["model_to_annotator_agreement_alpha"] = float(np.mean(alphas))
            else:
                m["model_to_annotator_agreement_alpha"] = None
            per_value[v] = m
        slice_metrics[fld] = per_value

    cat_metrics = {}
    for class_idx, cat in enumerate(cats):
        y_true_cat = y_true[:, class_idx].reshape(-1, 1)
        y_pred_cat = y_pred[:, class_idx].reshape(-1, 1)
        tp = np.sum((y_true_cat == 1) & (y_pred_cat == 1))
        fp = np.sum((y_true_cat == 0) & (y_pred_cat == 1))
        fn = np.sum((y_true_cat == 1) & (y_pred_cat == 0))
        f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        cat_metrics[cat] = {
            "f1": float(f1),
            "model_to_annotator_agreement_alpha": float(alpha_by_cat[cat]) if use_annotators else None,
        }
    return overall, slice_metrics, cat_metrics


def _pair_alpha(preds1: dict, preds2: dict, items: list[str], cat: str) -> float:
    ratings = []
    for iid in items:
        r1 = 1 if preds1[iid]["predictions"].get(cat, False) else 0
        r2 = 1 if preds2[iid]["predictions"].get(cat, False) else 0
        ratings.append([r1, r2])
    return compute_krippendorff_alpha_multi(ratings)


def _token_counts(loaded_runs: list[dict], opts: EvalOptions) -> pd.DataFrame:
    """Estimated rendered tokens per codebook category (~1.3 tokens per word)."""
    codebook_paths_to_count = list(opts.codebooks)
    if not codebook_paths_to_count:
        seen_cb_ids: set[str] = set()
        for run_item in loaded_runs:
            cb_path_from_meta = run_item["meta"].get("codebook")
            if cb_path_from_meta and Path(cb_path_from_meta).exists():
                if cb_path_from_meta not in seen_cb_ids:
                    seen_cb_ids.add(cb_path_from_meta)
                    codebook_paths_to_count.append(cb_path_from_meta)
            else:
                cb_id = run_item["meta"].get("codebook_id", "")
                if cb_id and cb_id not in seen_cb_ids:
                    seen_cb_ids.add(cb_id)
                    inferred = Path(opts.codebooks_dir) / f"{cb_id}.yaml"
                    if inferred.exists():
                        codebook_paths_to_count.append(str(inferred))

    records = []
    for cb_path_str in codebook_paths_to_count:
        cb_path = Path(cb_path_str)
        if not cb_path.exists():
            print(f"Warning: Codebook not found for token counts: {cb_path}")
            continue
        try:
            with open(cb_path, encoding="utf-8") as f:
                cb = yaml.safe_load(f)
            cb_id = str(cb.get("id") or cb_path.stem)
            render_spec = cb.get("render", DEFAULT_RENDER)
            total_tokens = 0
            for cat_key, cat_data in cb.get("categories", {}).items():
                cat_text = ""
                for entry in render_spec:
                    val = cat_data.get(entry["field"])
                    if val is not None:
                        if isinstance(val, list):
                            cat_text += " " + " ".join(str(w) for w in val)
                        else:
                            cat_text += " " + str(val)
                est_tokens = int(len(cat_text.split()) * 1.3)
                total_tokens += est_tokens
                records.append({"codebook": cb_id, "category": cat_key, "estimated_tokens": est_tokens})
            records.append({"codebook": cb_id, "category": "TOTAL", "estimated_tokens": total_tokens})
        except Exception as e:  # noqa: BLE001 - warn and continue
            print(f"Warning: Failed to compute token counts for {cb_path}: {e}")
    return pd.DataFrame(records)


def load_runs(run_paths: list[str], dataset: dict[str, dict]) -> list[dict]:
    """Load run files, print audit notes, keep runs with >= 1 aligned item."""
    loaded = []
    for run_file_str in run_paths:
        run_file = Path(run_file_str)
        if not run_file.exists():
            print(f"Warning: Run file not found: {run_file} — skipping")
            continue
        predictions = load_run_predictions(run_file)
        if not predictions:
            print(f"Warning: no readable records in {run_file} — skipping")
            continue
        run_id = run_file.stem
        meta = next((r["run"] for r in predictions.values() if isinstance(r.get("run"), dict)), None)
        missing = [k for k in ("codebook_id", "model_tag", "provider") if not (meta or {}).get(k)]
        if missing:
            print(f"Warning: {run_file} has no run metadata ({', '.join(missing)} missing). "
                  "Is it a run file written by `cba run`? Skipping.")
            continue

        fallback_count = sum(
            1
            for p in predictions.values()
            if p.get("parse_notes") and ("Fallback key" in p.get("parse_notes") or "FORMAT WARNING" in p.get("parse_notes"))
        )
        failed_count = sum(1 for p in predictions.values() if p.get("parse_status") != "ok")
        if fallback_count > 0:
            print(f"  [Audit Note] {run_id}: {fallback_count}/{len(predictions)} items used format fallbacks (non-canonical key like 'predicting')")
        if failed_count > 0:
            print(f"  [Audit Warning] {run_id}: {failed_count}/{len(predictions)} items failed clean parsing (status != 'ok')")

        aligned = [(dataset[iid], predictions[iid]) for iid in predictions if iid in dataset]
        if not aligned:
            continue
        loaded.append({"run_id": run_id, "run_file": run_file, "meta": meta, "predictions": predictions, "aligned": aligned})
    return loaded


def build_report(dataset: dict[str, dict], loaded_runs: list[dict], opts: EvalOptions) -> dict[str, pd.DataFrame]:
    """Compute all sheets. Returns an ordered mapping sheet name -> DataFrame."""
    cats = opts.categories
    fields = opts.fields
    floor = opts.min_cell_floor
    use_annotators = has_annotators(dataset, fields)
    if not use_annotators:
        print(
            f"Note: gold data has no '{fields.annotators_field}' field. "
            "Annotator-based metrics (model-to-annotator alpha) are left empty."
        )

    codebook_runs: dict[str, list] = {}
    model_predictions: dict[tuple, dict] = {}
    for run_item in loaded_runs:
        meta = run_item["meta"]
        arm = arm_key(meta)
        codebook_runs.setdefault(arm, []).append((run_item["run_id"], run_item["predictions"]))
        m_key = (meta["model_tag"], meta["provider"])
        model_predictions.setdefault(m_key, {})[arm] = run_item["predictions"]

    # ── Sheet1 slice specs ──
    s1_field = opts.sheet1_slice_field
    s1_values: list[str] = []
    if s1_field:
        s1_values = sorted({slice_value(g, s1_field, opts.bool_labels) for r in loaded_runs for g, _ in r["aligned"]})
    slice_specs = []
    for s_cat in ["ALL", *cats]:
        slice_specs.append((s_cat, "pooled"))
        slice_specs.extend((s_cat, v) for v in s1_values)

    def in_slice(gold: dict, s_sub: str) -> bool:
        return s_sub == "pooled" or slice_value(gold, s1_field, opts.bool_labels) == s_sub

    # ── Model-to-model alpha per (codebook, category, slice) ──
    m2m_alpha_map = {}
    for cb_id, cb_run_list in codebook_runs.items():
        if len(cb_run_list) >= 2:
            for s_cat, s_sub in slice_specs:
                pair_vals = []
                for i in range(len(cb_run_list)):
                    for j in range(i + 1, len(cb_run_list)):
                        _, preds1 = cb_run_list[i]
                        _, preds2 = cb_run_list[j]
                        common = sorted(set(preds1) & set(preds2) & set(dataset))
                        if s_sub != "pooled":
                            common = [iid for iid in common if in_slice(dataset[iid], s_sub)]
                        if len(common) < floor:
                            continue
                        if s_cat == "ALL":
                            pair_vals.append(float(np.mean([_pair_alpha(preds1, preds2, common, c) for c in cats])))
                        else:
                            pair_vals.append(_pair_alpha(preds1, preds2, common, s_cat))
                m2m_alpha_map[(cb_id, s_cat, s_sub)] = float(np.mean(pair_vals)) if pair_vals else None

    # ── Sheet1 ──
    main_records = []
    blanked = 0
    for run_item in loaded_runs:
        run_id, meta, aligned = run_item["run_id"], run_item["meta"], run_item["aligned"]
        cb_id = meta["codebook_id"]
        base = {
            "run_id": run_id,
            "codebook_id": cb_id,
            "model_tag": meta["model_tag"],
            "model_digest": meta.get("model_digest", None),
            "provider": meta["provider"],
        }
        for s_cat, s_sub in slice_specs:
            subset = aligned if s_sub == "pooled" else [it for it in aligned if in_slice(it[0], s_sub)]
            n_items = len(subset)
            m2m_val = m2m_alpha_map.get((arm_key(meta), s_cat, s_sub), None)
            m2m_display = "NA" if m2m_val is None else m2m_val
            row = {**base, "category": s_cat, "slice": s_sub, "n_items": n_items}
            if n_items < floor:
                blanked += 1
                row.update(micro_f1=None, macro_f1=None, jaccard=None, hamming_loss=None,
                           model_to_annotator_agreement_alpha=None, model_to_model_agreement_alpha=m2m_display)
                main_records.append(row)
                continue
            y_true, y_pred = _matrices(subset, cats, fields)
            sub_gold = [g for g, _ in subset]
            if s_cat == "ALL":
                clf = calculate_classification_metrics(y_true, y_pred)
                if use_annotators:
                    m2a = float(np.mean([_m2a_alpha(sub_gold, y_pred[:, k], c, fields) for k, c in enumerate(cats)]))
                else:
                    m2a = None
                row.update(micro_f1=clf["micro_f1"], macro_f1=clf["macro_f1"], jaccard=clf["jaccard"],
                           hamming_loss=clf["hamming_loss"], model_to_annotator_agreement_alpha=m2a,
                           model_to_model_agreement_alpha=m2m_display)
            else:
                class_idx = cats.index(s_cat)
                sc = single_category_metrics(y_true[:, class_idx], y_pred[:, class_idx], n_items)
                m2a = float(_m2a_alpha(sub_gold, y_pred[:, class_idx], s_cat, fields)) if use_annotators else None
                row.update(micro_f1=sc["f1"], macro_f1=sc["f1"], jaccard=sc["jaccard"],
                           hamming_loss=sc["hamming_loss"], model_to_annotator_agreement_alpha=m2a,
                           model_to_model_agreement_alpha=m2m_display)
            main_records.append(row)
    sheets: dict[str, pd.DataFrame] = {"Sheet1": pd.DataFrame(main_records)}
    if blanked:
        print(f"Note: {blanked} Sheet1 row(s) have fewer than {floor} items (--min-cell-floor), "
              "so their metrics are left empty.")

    # ── Slice sheets and Category Slices ──
    slice_records: dict[str, list] = {f: [] for f in opts.slice_by}
    category_records = []
    for run_item in loaded_runs:
        run_id, meta = run_item["run_id"], run_item["meta"]
        overall, slices, categories = process_eval_run(run_item["predictions"], dataset, opts, use_annotators)
        if not overall:
            continue
        ident = {"run_id": run_id, "codebook_id": meta["codebook_id"], "model_tag": meta["model_tag"],
                 "model_digest": meta.get("model_digest", None)}
        for fld, per_value in slices.items():
            for v, m in per_value.items():
                slice_records[fld].append({
                    **ident, fld: v,
                    "micro_f1": m["micro_f1"], "macro_f1": m["macro_f1"], "jaccard": m["jaccard"],
                    "hamming_loss": m["hamming_loss"],
                    "model_to_annotator_agreement_alpha": m["model_to_annotator_agreement_alpha"],
                })
        for cat, m in categories.items():
            category_records.append({**ident, "category": cat, "f1": m["f1"],
                                     "model_to_annotator_agreement_alpha": m["model_to_annotator_agreement_alpha"]})
    for fld in opts.slice_by:
        sheets[slice_sheet_name(fld)] = pd.DataFrame(slice_records[fld])
    sheets["Category Slices"] = pd.DataFrame(category_records)

    # ── Model Agreement ──
    model_agreement_records = []
    for cb_id, cb_run_list in codebook_runs.items():
        if len(cb_run_list) >= 2:
            for i in range(len(cb_run_list)):
                for j in range(i + 1, len(cb_run_list)):
                    run_id1, preds1 = cb_run_list[i]
                    run_id2, preds2 = cb_run_list[j]
                    common = sorted(set(preds1) & set(preds2) & set(dataset))
                    if not common:
                        continue
                    alphas = [_pair_alpha(preds1, preds2, common, c) for c in cats]
                    model_agreement_records.append({
                        "codebook_id": cb_id,
                        "comparison": f"{run_id1} vs {run_id2}",
                        "mean_model_agreement_alpha": float(np.mean(alphas)),
                        "items_compared": len(common),
                    })
    sheets["Model Agreement"] = pd.DataFrame(model_agreement_records)

    # ── Codebook Agreement (every pair of codebooks within the same model) ──
    inter_cb_records = []
    for (model_tag, provider), arms_dict in model_predictions.items():
        model_label = f"{model_tag} ({provider})"
        cb_ids_for_model = sorted(arms_dict)
        for i in range(len(cb_ids_for_model)):
            for j in range(i + 1, len(cb_ids_for_model)):
                arm1, arm2 = cb_ids_for_model[i], cb_ids_for_model[j]
                preds1, preds2 = arms_dict[arm1], arms_dict[arm2]
                common = sorted(set(preds1) & set(preds2) & set(dataset))
                if not common:
                    continue
                comp = f"{arm1} vs {arm2}"
                inter_cb_records.append({
                    "model": model_label, "comparison": comp, "slice_type": "overall", "slice_value": "overall",
                    "agreement_alpha": float(np.mean([_pair_alpha(preds1, preds2, common, c) for c in cats])),
                    "items_compared": len(common),
                })
                for fld in opts.agreement_fields:
                    vals = [slice_value(dataset[iid], fld, opts.bool_labels) for iid in common]
                    for v in sorted(set(vals)):
                        items_v = [iid for iid in common if slice_value(dataset[iid], fld, opts.bool_labels) == v]
                        if len(items_v) >= 2:
                            inter_cb_records.append({
                                "model": model_label, "comparison": comp, "slice_type": fld, "slice_value": v,
                                "agreement_alpha": float(np.mean([_pair_alpha(preds1, preds2, items_v, c) for c in cats])),
                                "items_compared": len(items_v),
                            })
                for cat in cats:
                    inter_cb_records.append({
                        "model": model_label, "comparison": comp, "slice_type": "category", "slice_value": cat,
                        "agreement_alpha": _pair_alpha(preds1, preds2, common, cat),
                        "items_compared": len(common),
                    })
    sheets["Codebook Agreement"] = pd.DataFrame(inter_cb_records)

    sheets["Token Counts"] = _token_counts(loaded_runs, opts)

    # ── Optional: ordinal scores ──
    ordinal_records = []
    ordinal_runs = [r for r in loaded_runs if is_ordinal_run(r["predictions"])]
    if ordinal_runs and not opts.gold_scores_field:
        print("Note: ordinal runs found but no --gold-scores-field given. The Ordinal sheet is skipped.")
    if ordinal_runs and opts.gold_scores_field:
        for run_item in ordinal_runs:
            meta = run_item["meta"]
            ident = {"run_id": run_item["run_id"], "codebook_id": meta["codebook_id"],
                     "model_tag": meta["model_tag"], "provider": meta["provider"]}
            all_pairs = []
            for cat in cats:
                pairs = []
                for gold, rec in run_item["aligned"]:
                    g = (gold.get(opts.gold_scores_field) or {}).get(cat)
                    if g is None:
                        continue
                    pairs.append((g, (rec.get("scores") or {}).get(cat)))
                all_pairs.extend(pairs)
                ordinal_records.append({**ident, "category": cat, **ordinal_metrics(pairs)})
            ordinal_records.append({**ident, "category": "ALL", **ordinal_metrics(all_pairs)})
    sheets["Ordinal"] = pd.DataFrame(ordinal_records)

    # ── Optional: evidence quotes ──
    evidence_records = []
    for run_item in loaded_runs:
        if not is_evidence_run(run_item["predictions"]):
            continue
        meta = run_item["meta"]
        ident = {"run_id": run_item["run_id"], "codebook_id": meta["codebook_id"],
                 "model_tag": meta["model_tag"], "provider": meta["provider"]}
        recs = [rec for _, rec in run_item["aligned"]]
        for cat in cats:
            evidence_records.append({**ident, "category": cat, **evidence_summary(recs, [cat])})
        evidence_records.append({**ident, "category": "ALL", **evidence_summary(recs, cats)})
    sheets["Evidence"] = pd.DataFrame(evidence_records)
    sheets["Usage"] = usage_sheet(loaded_runs)
    return sheets


def usage_sheet(loaded_runs: list[dict]) -> pd.DataFrame:
    """Tokens and latency per run, over all records of the run file that carry a ``usage`` field.

    Runs without usage data get no row. Means and percentiles skip records where a value is missing.
    """
    rows = []
    for run_item in loaded_runs:
        usages = [r["usage"] for r in run_item["predictions"].values() if isinstance(r.get("usage"), dict)]
        if not usages:
            continue

        def values(key: str, usages: list[dict] = usages) -> list[float]:
            return [float(u[key]) for u in usages if u.get(key) is not None]

        inp, out, lat, passes = values("input_tokens"), values("output_tokens"), values("latency_ms"), values("passes")
        meta = run_item["meta"]
        rows.append({
            "run_id": run_item["run_id"], "codebook_id": meta["codebook_id"], "model_tag": meta["model_tag"],
            "provider": meta["provider"], "prompt_condition": meta.get("prompt_condition"),
            "n_records": len(run_item["predictions"]), "n_with_usage": len(usages),
            "input_tokens_total": int(sum(inp)) if inp else None,
            "input_tokens_mean": float(np.mean(inp)) if inp else None,
            "output_tokens_total": int(sum(out)) if out else None,
            "output_tokens_mean": float(np.mean(out)) if out else None,
            "latency_ms_mean": float(np.mean(lat)) if lat else None,
            "latency_ms_median": float(np.median(lat)) if lat else None,
            "latency_ms_p95": float(np.percentile(lat, 95)) if lat else None,
            "latency_s_total": float(sum(lat) / 1000) if lat else None,
            "passes_mean": float(np.mean(passes)) if passes else None,
        })
    return pd.DataFrame(rows)


PER_RUN_SHEETS_ALWAYS = ("Sheet1",)
WHOLE_SHEETS = ("Token Counts", "Model Agreement", "Codebook Agreement")


def write_report(sheets: dict[str, pd.DataFrame], output_path: Path | str, merge: bool, evaluated_run_ids: set[str]) -> dict[str, pd.DataFrame]:
    """Write the workbook. With ``merge``, rows of re-evaluated runs replace those in an existing file.

    Merge semantics: a per-run sheet that was computed again keeps the other runs' rows and
    replaces the rows of the re-evaluated runs. A sheet of the existing file that was not
    computed again is kept unchanged. The whole-workbook sheets (Token Counts, Model Agreement,
    Codebook Agreement) are taken from the existing file only when nothing new was computed for
    them. Empty sheets other than Sheet1 are not written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheets = dict(sheets)

    if merge and output_path.exists():
        try:
            kept = []
            with pd.ExcelFile(output_path) as reader:
                for name in reader.sheet_names:
                    df_new = sheets.get(name)
                    recomputed = df_new is not None and (name in PER_RUN_SHEETS_ALWAYS or not df_new.empty)
                    if name in WHOLE_SHEETS:
                        if df_new is None or df_new.empty:
                            sheets[name] = pd.read_excel(reader, sheet_name=name, keep_default_na=False, na_values=[""])
                    elif not recomputed:
                        sheets[name] = pd.read_excel(reader, sheet_name=name, keep_default_na=False, na_values=[""])
                        kept.append(name)
                    else:
                        df_existing = pd.read_excel(reader, sheet_name=name, keep_default_na=False, na_values=[""])
                        if "run_id" in df_existing.columns:
                            df_existing = df_existing[~df_existing["run_id"].isin(evaluated_run_ids)]
                        sheets[name] = pd.concat([df_existing, df_new], ignore_index=True)
            print(f"✓ Merged new run(s) {evaluated_run_ids} into existing sheets of {output_path}")
            if kept:
                print(f"  Kept unchanged (not computed again): {', '.join(kept)}")
        except Exception as e:  # noqa: BLE001 - warn and write new sheets
            print(f"Warning: Failed to merge into existing Excel file: {e}. Writing new sheets instead.")

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            if name == "Sheet1" or not df.empty:
                df.to_excel(writer, sheet_name=name, index=False)
    print(f"\n✓ Evaluation spreadsheet written to: {output_path}")
    print("Sheets included: " + ", ".join(n for n, d in sheets.items() if n == "Sheet1" or not d.empty))
    return sheets
