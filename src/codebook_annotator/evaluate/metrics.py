"""Metric functions."""

from __future__ import annotations

import numpy as np


def compute_krippendorff_alpha_multi(items_ratings: list[list[int | None]]) -> float:
    """Compute nominal/binary Krippendorff's alpha for multiple coders.

    Each element of items_ratings is a list of ratings (0, 1, or None) for a single item.
    At least 2 ratings must be present per item to contribute to observed disagreement.
    """
    total_votes = 0
    p0 = 0
    p1 = 0
    observed_sum = 0.0

    for ratings in items_ratings:
        valid_ratings = [r for r in ratings if r is not None]
        m = len(valid_ratings)
        if m < 2:
            continue

        m0 = sum(1 for r in valid_ratings if r == 0)
        m1 = sum(1 for r in valid_ratings if r == 1)

        total_votes += m
        p0 += m0
        p1 += m1

        observed_sum += (2.0 * m0 * m1) / (m - 1)

    if total_votes <= 1:
        return 0.0

    d_o = observed_sum / total_votes
    d_e = (2.0 * p0 * p1) / (total_votes * (total_votes - 1))

    if d_e == 0.0:
        return 1.0 if d_o == 0.0 else 0.0

    return 1.0 - (d_o / d_e)


def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculate classification metrics (micro/macro F1, Jaccard, Hamming loss).

    y_true and y_pred are binary matrices of shape (N, num_classes).
    """
    N, num_classes = y_true.shape
    if N == 0:
        return {"micro_f1": 0.0, "macro_f1": 0.0, "jaccard": 0.0, "hamming_loss": 0.0}

    # Hamming loss
    hamming_loss = np.sum(y_true != y_pred) / (N * num_classes)

    # Class-wise TP, FP, FN
    tp = np.sum((y_true == 1) & (y_pred == 1), axis=0)
    fp = np.sum((y_true == 0) & (y_pred == 1), axis=0)
    fn = np.sum((y_true == 1) & (y_pred == 0), axis=0)

    # Macro F1
    macro_f1_list = []
    for i in range(num_classes):
        denom = 2 * tp[i] + fp[i] + fn[i]
        if denom == 0:
            val = 0.0  # undefined F1 (no gold and no predicted positives) counts as 0
        else:
            val = (2 * tp[i]) / denom
        macro_f1_list.append(val)
    macro_f1 = np.mean(macro_f1_list)

    # Micro F1
    sum_tp = np.sum(tp)
    sum_fp = np.sum(fp)
    sum_fn = np.sum(fn)
    micro_f1 = (2 * sum_tp) / (2 * sum_tp + sum_fp + sum_fn) if (2 * sum_tp + sum_fp + sum_fn) > 0 else 0.0

    # Jaccard similarity
    jaccard_list = []
    for i in range(N):
        intersection = np.sum((y_true[i] == 1) & (y_pred[i] == 1))
        union = np.sum((y_true[i] == 1) | (y_pred[i] == 1))
        jaccard_list.append(intersection / union if union > 0 else 1.0)
    jaccard = np.mean(jaccard_list)

    return {
        "micro_f1": float(micro_f1),
        "macro_f1": float(macro_f1),
        "jaccard": float(jaccard),
        "hamming_loss": float(hamming_loss),
    }


def single_category_metrics(y_true_cat: np.ndarray, y_pred_cat: np.ndarray, n_items: int) -> dict:
    """Per-category F1 / Jaccard / Hamming as computed for Sheet1 category rows."""
    tp = np.sum((y_true_cat == 1) & (y_pred_cat == 1))
    fp = np.sum((y_true_cat == 0) & (y_pred_cat == 1))
    fn = np.sum((y_true_cat == 1) & (y_pred_cat == 0))
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    jacc = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 1.0
    h_loss = (fp + fn) / n_items if n_items > 0 else 0.0
    return {"f1": float(f1), "jaccard": float(jacc), "hamming_loss": float(h_loss)}


# ── Ordinal scores (optional "+ordinal" prompt condition) ──────────────────────

def ordinal_metrics(pairs: list[tuple[int, int | None]]) -> dict:
    """MAE and accuracy-within-1 for (gold, predicted) score pairs with non-null gold.

    A null prediction counts as a miss for Acc<=1 and is excluded from MAE
    (reported as n_pred_null).
    """
    n_pairs = len(pairs)
    scored = [(g, p) for g, p in pairs if p is not None]
    n_pred_null = n_pairs - len(scored)
    mae = float(np.mean([abs(g - p) for g, p in scored])) if scored else None
    acc1 = float(sum(1 for g, p in scored if abs(g - p) <= 1) / n_pairs) if n_pairs else None
    return {"n_pairs": n_pairs, "n_pred_null": n_pred_null, "mae": mae, "acc_within_1": acc1}


# ── Evidence quotes (optional "+evidence" prompt condition) ────────────────────

def evidence_summary(records: list[dict], categories: list[str]) -> dict:
    """Evidence statistics pooled over ``categories`` (pass one category for a per-category row).

    n_present: predictions marked present. n_quotes / n_verified: quotes given and
    verified verbatim. share_verified = n_verified / n_quotes. share_present_supported =
    present predictions with >= 1 verified quote / n_present.
    """
    n_present = n_quotes = n_verified = n_supported = 0
    for rec in records:
        preds = rec.get("predictions") or {}
        evidence = rec.get("evidence") or {}
        for cat in categories:
            quotes = evidence.get(cat) or []
            nv = sum(1 for q in quotes if isinstance(q, dict) and q.get("verified"))
            n_quotes += len(quotes)
            n_verified += nv
            if preds.get(cat, False):
                n_present += 1
                if nv > 0:
                    n_supported += 1
    return {
        "n_present": n_present,
        "n_quotes": n_quotes,
        "n_verified": n_verified,
        "share_verified": (n_verified / n_quotes) if n_quotes else None,
        "share_present_supported": (n_supported / n_present) if n_present else None,
    }
