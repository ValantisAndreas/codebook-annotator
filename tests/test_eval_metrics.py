"""Hand-checked metric values on tiny synthetic examples."""

import numpy as np
import pytest

from codebook_annotator.evaluate.metrics import (
    calculate_classification_metrics,
    compute_krippendorff_alpha_multi,
    evidence_summary,
    ordinal_metrics,
    single_category_metrics,
)


def test_alpha_two_coders_hand_computed():
    # coincidence matrix: o11=2, o00=2, o01=o10=1, n0=n1=3, n=6
    # D_o = 2/6, D_e = 2*3*3/(6*5) = 0.6 -> alpha = 1 - (1/3)/0.6 = 4/9
    assert compute_krippendorff_alpha_multi([[1, 1], [0, 0], [1, 0]]) == pytest.approx(4 / 9)


def test_alpha_three_coders_hand_computed():
    # n=9, p1=5, p0=4. observed: item1 2*1*2/2 = 2 -> D_o = 2/9, D_e = 2*4*5/(9*8) = 5/9 -> alpha = 0.6
    assert compute_krippendorff_alpha_multi([[1, 1, 0], [0, 0, 0], [1, 1, 1]]) == pytest.approx(0.6)


def test_alpha_edge_cases():
    assert compute_krippendorff_alpha_multi([[1, 1], [1, 1]]) == 1.0  # no expected disagreement, none observed
    assert compute_krippendorff_alpha_multi([[1], [0]]) == 0.0  # no item with >= 2 ratings
    # None ratings are ignored, a single remaining rating -> item skipped
    assert compute_krippendorff_alpha_multi([[1, None], [1, 1, None], [0, 0]]) == compute_krippendorff_alpha_multi([[1, 1], [0, 0]])


def test_classification_metrics_hand_computed():
    y_true = np.array([[1, 0], [0, 1], [1, 1]])
    y_pred = np.array([[1, 1], [0, 0], [1, 0]])
    m = calculate_classification_metrics(y_true, y_pred)
    assert m["macro_f1"] == pytest.approx(0.5)  # class0 F1=1, class1 F1=0
    assert m["micro_f1"] == pytest.approx(4 / 7)  # tp=2 fp=1 fn=2
    assert m["jaccard"] == pytest.approx(1 / 3)  # rows: 1/2, 0/1, 1/2
    assert m["hamming_loss"] == pytest.approx(0.5)  # 3 wrong cells of 6


def test_classification_metrics_empty_class_and_empty_input():
    y = np.array([[1, 0], [0, 0]])
    m = calculate_classification_metrics(y, y.copy())
    assert m["macro_f1"] == 0.5  # class0 F1 = 1, class1 undefined (never in gold, never predicted) counts as 0
    assert calculate_classification_metrics(np.zeros((0, 2)), np.zeros((0, 2)))["micro_f1"] == 0.0


def test_single_category_metrics():
    m = single_category_metrics(np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0]), 4)
    assert m == {"f1": pytest.approx(0.5), "jaccard": pytest.approx(1 / 3), "hamming_loss": pytest.approx(0.5)}


def test_ordinal_metrics_hand_computed():
    # pairs (gold, pred): |8-7| = 1, null = miss, |5-9| = 4
    m = ordinal_metrics([(8, 7), (3, None), (5, 9)])
    assert m == {"n_pairs": 3, "n_pred_null": 1, "mae": pytest.approx(2.5), "acc_within_1": pytest.approx(1 / 3)}
    assert ordinal_metrics([]) == {"n_pairs": 0, "n_pred_null": 0, "mae": None, "acc_within_1": None}
    assert ordinal_metrics([(4, None)])["mae"] is None


def test_evidence_summary_hand_computed():
    recs = [
        {"predictions": {"A": True, "B": False},
         "evidence": {"A": [{"quote": "x", "verified": True}, {"quote": "y", "verified": False}]}},
        {"predictions": {"A": True, "B": True},
         "evidence": {"A": [{"quote": "z", "verified": False}], "B": [{"quote": "w", "verified": True}]}},
        {"predictions": {"A": False, "B": True}, "evidence": {}},
    ]
    a = evidence_summary(recs, ["A"])
    assert a == {"n_present": 2, "n_quotes": 3, "n_verified": 1,
                 "share_verified": pytest.approx(1 / 3), "share_present_supported": pytest.approx(0.5)}
    pooled = evidence_summary(recs, ["A", "B"])
    assert pooled["n_present"] == 4 and pooled["n_quotes"] == 4 and pooled["n_verified"] == 2
    assert pooled["share_present_supported"] == pytest.approx(2 / 4)
    assert evidence_summary([], ["A"])["share_verified"] is None


def test_undefined_f1_is_zero_everywhere():
    from codebook_annotator.evaluate.metrics import single_category_metrics
    nothing = np.zeros((3, 2))
    m = calculate_classification_metrics(nothing, nothing.copy())
    assert m["macro_f1"] == 0.0 and m["micro_f1"] == 0.0 and m["jaccard"] == 1.0
    assert single_category_metrics(np.zeros(3), np.zeros(3), 3)["f1"] == 0.0
