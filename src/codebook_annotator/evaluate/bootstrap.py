"""Paired bootstrap CIs, p-values and efficiency ratios for runs over the same items.

The first run is the baseline. Defaults: 5000 resamples, seed 42, alphabetical
category order.

Metrics: OOR% (share of items with at least one predicted label that no
annotator chose), OOR/item (mean number of such labels), coverage% (share of
annotator-chosen labels the model also predicted), macro-F1, micro-F1, Jaccard
(the last three against the majority labels). Only labels of the evaluated
categories are counted, as in the workbook evaluator. OOR and coverage need
per-annotator labels. Without them only the last three metrics are reported.
Items without annotator labels are left out of OOR% and OOR/item.

An undefined F1 (a category with no gold labels and no predictions) counts as 0,
as in the workbook evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .io import DEFAULT_FIELDS, EvalDataError, GoldFields, check_gold, load_jsonl_indexed, read_jsonl_records

ALL_NAMES = ["OOR%", "OOR/item", "coverage%", "macro-F1", "micro-F1", "Jaccard"]
GOLD_ONLY_NAMES = ["macro-F1", "micro-F1", "Jaccard"]


def load_gold(path: str, fields: GoldFields = DEFAULT_FIELDS) -> tuple[dict, bool]:
    """item_id (as text) -> (majority set, union of annotator labels, has annotators).

    Second return value: the annotators field is present in at least one record.
    """
    dataset = load_jsonl_indexed(path, fields.id_field)
    check_gold(dataset, path, fields)
    g = {}
    any_ann = False
    for iid, r in dataset.items():
        if fields.annotators_field in r:
            any_ann = True
        per_annotator = r.get(fields.annotators_field) or {}
        u = set()
        for labs in per_annotator.values():
            u |= set(labs)
        g[iid] = (set(r.get(fields.majority_field) or []), u, bool(per_annotator))
    return g, any_ann


def load_run(path: str, ids: set, cats: list[str]) -> dict:
    p = {}
    for iid, r in load_jsonl_indexed(path, "item_id").items():
        if iid in ids:
            p[iid] = {k for k, v in (r.get("predictions") or {}).items() if v and k in cats}
    return p


def run_categories(paths: list[str]) -> list[str]:
    cats: set[str] = set()
    for path in paths:
        for r in read_jsonl_records(path, "item_id"):
            cats.update((r.get("predictions") or {}).keys())
    return sorted(cats)


def arm_name(path: str) -> str:
    """Arm label of a run file: its file name without a trailing .jsonl."""
    name = Path(path).name
    return name[: -len(".jsonl")] if name.endswith(".jsonl") else name


def per_item(gold, pred, ids, cats):
    n, k = len(ids), len(cats)
    tp = np.zeros((n, k))
    fp = np.zeros((n, k))
    fn = np.zeros((n, k))
    oor_i = np.zeros(n)
    oor_n = np.zeros(n)
    hit = np.zeros(n)
    tot = np.zeros(n)
    jac = np.zeros(n)
    ann = np.zeros(n)
    catset = set(cats)
    for i, iid in enumerate(ids):
        maj, uni, has_ann = gold[iid]
        maj, uni = maj & catset, uni & catset
        ann[i] = 1 if has_ann else 0
        pr = pred[iid]
        for j, c in enumerate(cats):
            if c in pr and c in maj:
                tp[i, j] = 1
            elif c in pr:
                fp[i, j] = 1
            elif c in maj:
                fn[i, j] = 1
        out = pr - uni
        oor_i[i] = 1 if out else 0
        oor_n[i] = len(out)
        hit[i] = len(pr & uni)
        tot[i] = len(uni)
        u = pr | maj
        jac[i] = len(pr & maj) / len(u) if u else 1.0
    return dict(tp=tp, fp=fp, fn=fn, oor_i=oor_i, oor_n=oor_n, hit=hit, tot=tot, jac=jac, ann=ann)


def metrics(d, idx, with_annotators: bool = True):
    tp = d["tp"][idx].sum(0)
    fp = d["fp"][idx].sum(0)
    fn = d["fn"][idx].sum(0)
    den = 2 * tp + fp + fn
    f1c = np.divide(2 * tp, den, out=np.zeros_like(den), where=den > 0)
    T, F, N = tp.sum(), fp.sum(), fn.sum()
    gold_only = [
        f1c.mean(),
        2 * T / (2 * T + F + N) if (2 * T + F + N) else 0.0,
        d["jac"][idx].mean(),
    ]
    if not with_annotators:
        return np.array(gold_only)
    tot = d["tot"][idx].sum()
    if d["ann"].all():
        oor_pct, oor_per_item = 100 * d["oor_i"][idx].mean(), d["oor_n"][idx].mean()
    else:  # items without annotator labels are left out of the OOR measures
        a = d["ann"][idx]
        n_ann = a.sum()
        oor_pct = 100 * (d["oor_i"][idx] * a).sum() / n_ann if n_ann else np.nan
        oor_per_item = (d["oor_n"][idx] * a).sum() / n_ann if n_ann else np.nan
    return np.array([
        oor_pct,
        oor_per_item,
        100 * d["hit"][idx].sum() / tot if tot else 0.0,
        *gold_only,
    ])


def pval(diff_boot, diff_obs):
    """J&M paired bootstrap test: how often does a null-centred replicate
    reach twice the observed effect. Two-sided, symmetrised. NaN when the difference is undefined."""
    if np.isnan(diff_obs):
        return float("nan")
    if diff_obs == 0:
        return 1.0
    centred = diff_boot - diff_obs
    p = np.mean(centred >= abs(diff_obs)) + np.mean(centred <= -abs(diff_obs))
    return min(1.0, p)


@dataclass
class BootstrapResult:
    names: list[str]
    arms: list[str]
    n_items: int
    resamples: int
    point: dict[str, np.ndarray]
    boot: dict[str, np.ndarray]
    jac: dict[str, np.ndarray]
    with_annotators: bool
    lines: list[str] = field(default_factory=list)


def run_bootstrap(sample: str, runs: list[str], *, resamples: int = 5000, seed: int = 42,
                  categories: list[str] | None = None, fields: GoldFields = DEFAULT_FIELDS) -> BootstrapResult:
    rng = np.random.default_rng(seed)
    cats = list(categories) if categories else run_categories(runs)
    gold, with_ann = load_gold(sample, fields)
    names = ALL_NAMES if with_ann else GOLD_ONLY_NAMES
    names_seen: dict[str, str] = {}
    for r in runs:
        a = arm_name(r)
        if a in names_seen:
            raise EvalDataError(f"two run files have the same name '{a}': {names_seen[a]} and {r}. "
                                "Rename one of them, the name labels the run in the output.")
        names_seen[a] = r
    preds = {arm_name(r): load_run(r, set(gold), cats) for r in runs}
    ids = sorted(set(gold) & set.intersection(*[set(p) for p in preds.values()]))
    n = len(ids)
    if n == 0:
        raise EvalDataError("no item is present in the gold file and in every run, so nothing can be scored. "
                            "Check that the runs were made on this dataset and that --id-field is right.")
    lines = []
    if not with_ann:
        lines.append(f"Note: no '{fields.annotators_field}' field in the gold data. OOR and coverage are not reported.")
    else:
        n_unannotated = sum(1 for iid in ids if not gold[iid][2])
        if n_unannotated:
            lines.append(f"Note: {n_unannotated} item(s) have no annotator labels. They are left out of OOR% and "
                         "OOR/item (coverage is unaffected).")
    lines.append(f"items scored: {n}   resamples: {resamples}\n")

    data = {a: per_item(gold, p, ids, cats) for a, p in preds.items()}
    point = {a: metrics(d, np.arange(n), with_ann) for a, d in data.items()}
    boot = {a: np.empty((resamples, len(names))) for a in data}
    for b in range(resamples):
        idx = rng.integers(0, n, n)
        for a, d in data.items():
            boot[a][b] = metrics(d, idx, with_ann)

    for a in data:
        lo, hi = np.percentile(boot[a], [2.5, 97.5], axis=0)
        lines.append(a)
        for j, nm in enumerate(names):
            lines.append(f"   {nm:<11}{point[a][j]:>8.3f}   [{lo[j]:.3f}, {hi[j]:.3f}]")
        lines.append("")

    base = list(data)[0]
    lines.append("=" * 66)
    lines.append(f"PAIRED DIFFERENCES vs {base}\n")
    for a in list(data)[1:]:
        d = boot[a] - boot[base]
        pt = point[a] - point[base]
        lo, hi = np.percentile(d, [2.5, 97.5], axis=0)
        lines.append(a)
        lines.append(f"   {'metric':<11}{'diff':>9}{'95% CI':>22}{'p':>10}")
        for j, nm in enumerate(names):
            p = pval(d[:, j], pt[j])
            floor = 1.0 / resamples
            ps = f"<{floor:.4f}" if p < floor else f"{p:.4f}"
            lines.append(f"   {nm:<11}{pt[j]:>+9.3f}   [{lo[j]:+.3f}, {hi[j]:+.3f}]{ps:>10}")

        if with_ann:
            # efficiency: OOR labels avoided per point of coverage lost
            dn = point[base][2] - point[a][2]
            dr = point[base][1] - point[a][1]
            bn = boot[base][:, 2] - boot[a][:, 2]
            br = boot[base][:, 1] - boot[a][:, 1]
            ok = bn > 0.5
            if dn > 0.5 and ok.mean() > 0.9:
                eff = br[ok] / bn[ok]
                elo, ehi = np.percentile(eff, [2.5, 97.5])
                lines.append(f"   {'efficiency':<11}{dr/dn:>+9.4f}   [{elo:+.4f}, {ehi:+.4f}]"
                             f"        (OOR labels avoided per coverage point lost)")
                if ok.mean() < 1.0:
                    lines.append(f"                 [{100*(1-ok.mean()):.1f}% of replicates dropped: denominator near zero]")
            else:
                lines.append(f"   {'efficiency':<11}      n/a   (coverage change too small to divide by)")
        lines.append("")
    lines.append("p-values: J&M paired bootstrap test, two-sided. Floor is 1/B.")
    return BootstrapResult(names=names, arms=list(data), n_items=n, resamples=resamples, point=point, boot=boot,
                           jac={a: data[a]["jac"] for a in data}, with_annotators=with_ann, lines=lines)


def save_npz(result: BootstrapResult, out: str) -> None:
    np.savez(out, names=np.array(result.names), arms=np.array(result.arms),
             **{f"boot_{a}": result.boot[a] for a in result.arms},
             **{f"point_{a}": result.point[a] for a in result.arms},
             **{f"jac_{a}": result.jac[a] for a in result.arms})


def save_csv(result: BootstrapResult, out: str) -> None:
    """Tidy table: one row per (arm, metric) with point estimate, 95% CI and, for non-baseline
    arms, the paired difference vs the baseline with its CI and p-value. A p-value below the
    floor 1/B is written as 1/B, with p_at_floor = True (read it as "p < 1/B")."""
    import csv

    base = result.arms[0]
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "metric", "point", "ci_low", "ci_high", "diff_vs_baseline", "diff_ci_low", "diff_ci_high",
                    "p_value", "p_at_floor"])
        floor = 1.0 / result.resamples
        for a in result.arms:
            lo, hi = np.percentile(result.boot[a], [2.5, 97.5], axis=0)
            if a != base:
                d = result.boot[a] - result.boot[base]
                pt = result.point[a] - result.point[base]
                dlo, dhi = np.percentile(d, [2.5, 97.5], axis=0)
            for j, nm in enumerate(result.names):
                row = [a, nm, result.point[a][j], lo[j], hi[j]]
                if a != base:
                    p = pval(d[:, j], pt[j])
                    at_floor = bool(p < floor)
                    row += [pt[j], dlo[j], dhi[j], max(p, floor) if at_floor else p, at_floor]
                else:
                    row += ["", "", "", "", ""]
                w.writerow(row)
