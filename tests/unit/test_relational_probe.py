"""DR-3 relational probe statistics (pre-registration, section 5).

Hand-built point sets with known kNN overlap and known MST edges,
tie rules, preprocessing invariances, exclusion and zero-distance
rules. Hand sets use ``preprocess="raw"`` so that the geometry stays
as drawn (L2 normalisation would collapse 1-D points to +-1).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse.csgraph import minimum_spanning_tree

from kiki_oniric.eval.relational_probe import (
    calibrated_difference,
    check_representation,
    geometry_weights,
    knn_indices,
    mutual_knn,
    mutual_knn_tau,
    mst_edge_overlap,
    mst_edge_overlap_tau,
    pairwise_distances,
    preprocess,
    prim_mst_edges,
    relational_statistics,
    zero_distance_pairs,
)


def _line(*xs: float) -> np.ndarray:
    return np.array(xs, dtype=np.float64).reshape(-1, 1)


SQUARE = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)


# --- kNN and mutual-kNN -------------------------------------------------


def test_knn_ties_broken_by_lower_index() -> None:
    # point 0 is equidistant from 1 and 2; lower index wins
    d = pairwise_distances(_line(0.0, 1.0, -1.0, 50.0))
    assert knn_indices(d, 1)[0].tolist() == [1]
    assert knn_indices(d, 2)[0].tolist() == [1, 2]


def test_knn_excludes_self() -> None:
    d = pairwise_distances(np.random.default_rng(0).normal(size=(12, 3)))
    nn = knn_indices(d, 4)
    assert all(i not in nn[i] for i in range(12))


def test_mutual_knn_known_half_overlap() -> None:
    x = _line(0, 1, 10, 11)  # NN: 0->1 1->0 2->3 3->2
    y = _line(0, 1, 10, -10)  # NN: 0->1 1->0 2->1 3->0
    assert mutual_knn(x, y, k=1, preprocess="raw") == pytest.approx(0.5)


def test_mutual_knn_known_zero_overlap() -> None:
    x = _line(0, 1, 10, 11)
    y = _line(0, 10, 1, 11)  # NN: 0->2 2->0 1->3 3->1
    assert mutual_knn(x, y, k=1, preprocess="raw") == 0.0


def test_mutual_knn_k2_hand_count() -> None:
    # x: 1-D 0,1,2,3,4 ; k=2 neighbour sets
    # 0:{1,2} 1:{0,2} 2:{1,3}(tie 1,3 -> both, k=2) 3:{2,4} 4:{3,2}
    x = _line(0, 1, 2, 3, 4)
    # y = x with the last two swapped: positions 0,1,2,4,3
    y = _line(0, 1, 2, 4, 3)
    # y neighbour sets: 0:{1,2} 1:{0,2}
    # 2 (pos 2): idx 1 (pos 1) d1, idx 4 (pos 3) d1 -> {1,4}
    #   (idx 0 and idx 3 are at d2, so they are not neighbours)
    # 3 (pos 4): nearest index 4 (pos 3) d1, then index 2 (pos 2) d2
    # 4 (pos 3): nearest index 3 (pos 4) d1, then index 2 (pos 2) d1
    ny = {0: {1, 2}, 1: {0, 2}, 2: {1, 4}, 3: {4, 2}, 4: {3, 2}}
    nx = {0: {1, 2}, 1: {0, 2}, 2: {1, 3}, 3: {2, 4}, 4: {3, 2}}
    for ref, pts in ((nx, x), (ny, y)):
        nn = knn_indices(pairwise_distances(pts), 2)
        assert {i: set(nn[i].tolist()) for i in range(5)} == ref
    expected = np.mean([len(nx[i] & ny[i]) / 2 for i in range(5)])
    assert mutual_knn(x, y, k=2, preprocess="raw") == pytest.approx(expected)


def test_identical_representations_give_maximal_agreement() -> None:
    x = np.random.default_rng(1).normal(size=(40, 5))
    s = relational_statistics(x, x.copy(), k=5, tau=0.1)
    assert s["S_mKNN"] == 1.0
    assert s["S_H0"] == 1.0
    assert s["S_mKNN_tau"] == pytest.approx(1.0)
    assert s["S_H0_tau"] == pytest.approx(1.0)


def test_permuted_rows_much_lower_than_aligned() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(size=(60, 6))
    perm = rng.permutation(60)
    aligned = relational_statistics(x, x, k=5)
    permuted = relational_statistics(x, x[perm], k=5)
    for key in ("S_mKNN", "S_H0", "S_mKNN_tau", "S_H0_tau"):
        assert permuted[key] < 0.3 * aligned[key] + 1e-12


# --- MST (Prim, pre-registered tie rule) ---------------------------------


def test_prim_line_edges() -> None:
    edges = prim_mst_edges(pairwise_distances(_line(0, 1, 2, 3)))
    assert edges == {(0, 1), (1, 2), (2, 3)}


def test_prim_tie_rule_unit_square() -> None:
    # hand-traced: 0-1 (lowest index among tied keys), then 1-2
    # (strictly smaller than parent 0), then 3 keeps parent 0
    # because 1.0 is not strictly smaller than 1.0.
    edges = prim_mst_edges(pairwise_distances(SQUARE))
    assert edges == {(0, 1), (1, 2), (0, 3)}


def test_prim_total_weight_matches_scipy() -> None:
    rng = np.random.default_rng(3)
    d = pairwise_distances(rng.normal(size=(50, 4)))
    edges = prim_mst_edges(d)
    assert len(edges) == 49
    ours = sum(d[i, j] for i, j in edges)
    theirs = minimum_spanning_tree(d).sum()
    assert ours == pytest.approx(theirs)


def test_h0_overlap_normalised_by_n_minus_1_not_jaccard() -> None:
    x = _line(0, 1, 2, 3)  # edges 01 12 23
    y = _line(0, 2, 1, 3)  # order 0,2,1,3 -> edges 02 21 13
    assert mst_edge_overlap(x, y, preprocess="raw") == pytest.approx(1 / 3)


def test_h0_overlap_with_ties_square_vs_itself() -> None:
    assert mst_edge_overlap(SQUARE, SQUARE, preprocess="raw") == 1.0


# --- geometry weights, zero-distance rule, tau limit ---------------------


def test_geometry_weights_match_loop_oracle() -> None:
    rng = np.random.default_rng(4)
    dx = pairwise_distances(rng.normal(size=(15, 3)))
    dy = pairwise_distances(rng.normal(size=(15, 3)))
    tau = 0.1
    w = geometry_weights(dx, dy, tau)

    def q9(d: np.ndarray) -> float:
        off = d[~np.eye(len(d), dtype=bool)]
        return float(np.quantile(off[off > 0], 0.9))

    qx, qy = q9(dx), q9(dy)
    for i in range(15):
        for j in range(15):
            if i == j:
                continue
            ref = np.exp(-abs(np.log(dx[i, j] / qx) - np.log(dy[i, j] / qy)) / tau)
            assert w[i, j] == pytest.approx(ref)


def test_geometry_weights_uniform_rescale_gives_one() -> None:
    dx = pairwise_distances(np.random.default_rng(5).normal(size=(20, 3)))
    assert np.allclose(geometry_weights(dx, 7.0 * dx, 0.01)[~np.eye(20, dtype=bool)], 1.0)


def test_zero_distance_rule() -> None:
    # rows 0 and 1 coincide in x; rows 0 and 2 coincide in y
    x = np.array([[0.0], [0.0], [3.0], [7.0]])
    y = np.array([[0.0], [5.0], [0.0], [9.0]])
    dx, dy = pairwise_distances(x), pairwise_distances(y)
    w = geometry_weights(dx, dy, 0.1)
    assert w[0, 1] == 0.0  # zero in x only
    assert w[0, 2] == 0.0  # zero in y only
    assert zero_distance_pairs(dx) == 1
    assert zero_distance_pairs(dy) == 1
    both = np.array([[0.0], [0.0], [3.0], [7.0]])
    w2 = geometry_weights(dx, pairwise_distances(both), 0.1)
    assert w2[0, 1] == 1.0  # zero in both


def test_tau_to_infinity_recovers_structure_scores() -> None:
    rng = np.random.default_rng(6)
    x, y = rng.normal(size=(50, 4)), rng.normal(size=(50, 4))
    s = relational_statistics(x, y, k=5, tau=1e12)
    assert s["S_mKNN_tau"] == pytest.approx(s["S_mKNN"], abs=1e-9)
    assert s["S_H0_tau"] == pytest.approx(s["S_H0"], abs=1e-9)
    assert mutual_knn_tau(x, y, k=5, tau=1e12) == pytest.approx(s["S_mKNN"], abs=1e-9)
    assert mst_edge_overlap_tau(x, y, tau=1e12) == pytest.approx(s["S_H0"], abs=1e-9)


def test_geometry_scores_never_exceed_structure_scores() -> None:
    rng = np.random.default_rng(7)
    x, y = rng.normal(size=(50, 4)), rng.normal(size=(50, 4))
    s = relational_statistics(x, y, k=5, tau=0.1)
    assert s["S_mKNN_tau"] <= s["S_mKNN"] + 1e-12
    assert s["S_H0_tau"] <= s["S_H0"] + 1e-12


# --- preprocessing --------------------------------------------------------


def test_l2_rows_unit_norm_and_no_centering() -> None:
    x = np.random.default_rng(8).normal(size=(10, 3)) + 5.0
    p = preprocess(x, "l2")
    assert np.allclose(np.linalg.norm(p, axis=1), 1.0)
    assert not np.allclose(p.mean(axis=0), 0.0)


def test_center_l2_removes_translation() -> None:
    x = np.random.default_rng(9).normal(size=(30, 4))
    assert np.allclose(preprocess(x, "center_l2"), preprocess(x + 3.0, "center_l2"))


def test_statistics_invariant_to_row_scaling_and_rotation_under_l2() -> None:
    rng = np.random.default_rng(10)
    x, y = rng.normal(size=(40, 4)), rng.normal(size=(40, 4))
    q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    scale = rng.uniform(0.5, 4.0, size=(40, 1))
    base = relational_statistics(x, y, k=5)
    moved = relational_statistics((x * scale) @ q, y, k=5)
    for key in base:
        if key.startswith("S_"):
            assert moved[key] == pytest.approx(base[key])


def test_raw_pipeline_is_scale_invariant_for_all_scores() -> None:
    rng = np.random.default_rng(11)
    x, y = rng.normal(size=(40, 4)), rng.normal(size=(40, 4))
    a = relational_statistics(x, y, k=5, preprocess="raw")
    b = relational_statistics(3.0 * x, y, k=5, preprocess="raw")
    for key in ("S_mKNN", "S_H0", "S_mKNN_tau", "S_H0_tau"):
        assert b[key] == pytest.approx(a[key])


def test_unknown_preprocess_rejected() -> None:
    with pytest.raises(ValueError, match="preprocess"):
        preprocess(np.ones((3, 2)), "whiten")


# --- exclusion rule and input validation ---------------------------------


def test_check_representation_ok() -> None:
    assert check_representation(np.random.default_rng(12).normal(size=(10, 3))) is None


def test_all_coincident_rows_excluded() -> None:
    x = np.tile(np.array([[1.0, 2.0, 3.0]]), (10, 1))
    reason = check_representation(x)
    assert reason is not None and "coincide" in reason


def test_zero_norm_row_excluded() -> None:
    x = np.random.default_rng(13).normal(size=(10, 3))
    x[4] = 0.0
    reason = check_representation(x)
    assert reason is not None and "zero norm" in reason


def test_non_finite_excluded() -> None:
    x = np.random.default_rng(14).normal(size=(10, 3))
    x[2, 1] = np.nan
    reason = check_representation(x)
    assert reason is not None and "non-finite" in reason


def test_statistics_raise_on_excluded_representation() -> None:
    x = np.tile(np.array([[1.0, 2.0]]), (10, 1))
    y = np.random.default_rng(15).normal(size=(10, 2))
    with pytest.raises(ValueError, match="coincide"):
        relational_statistics(x, y, k=3)


def test_rejects_bad_shapes_and_k() -> None:
    x = np.random.default_rng(16).normal(size=(10, 3))
    with pytest.raises(ValueError, match="2-D"):
        relational_statistics(x[0], x[0], k=3)
    with pytest.raises(ValueError, match="same number of rows"):
        relational_statistics(x, x[:8], k=3)
    with pytest.raises(ValueError, match="k"):
        relational_statistics(x, x, k=10)
    with pytest.raises(ValueError, match="k"):
        relational_statistics(x, x, k=0)
    with pytest.raises(ValueError, match="tau"):
        relational_statistics(x, x, k=3, tau=0.0)


def test_calibrated_difference_is_a_difference_not_a_ratio() -> None:
    trained = {"S_mKNN": 0.2, "S_H0": 0.1}
    null = {"S_mKNN": 0.0, "S_H0": 0.05}
    d = calibrated_difference(trained, null)
    assert d["S_mKNN"] == pytest.approx(0.2)  # null = 0 stays finite
    assert d["S_H0"] == pytest.approx(0.05)
