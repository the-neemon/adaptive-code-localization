import numpy as np
import polars as pl
import pytest

from acl.actions import expand, fusion, order, ranks, rrf, stop
from acl.splits import make_splits


def test_order_is_descending_and_stable():
    assert order(np.array([0.2, 0.9, 0.2, 0.5])).tolist() == [1, 3, 0, 2]


def test_ranks_inverts_order():
    o = np.array([2, 0, 1])
    assert ranks(o).tolist() == [2, 3, 1]


def test_rrf_hand_computed():
    # chunk 0: ranks 1 and 3; chunk 1: 2 and 1; chunk 2: 3 and 2 (k = 60)
    s = rrf([np.array([0, 1, 2]), np.array([1, 2, 0])], k=60)
    assert s.tolist() == pytest.approx([1 / 61 + 1 / 63, 1 / 62 + 1 / 61, 1 / 63 + 1 / 62])


def test_fusion_rewards_agreement():
    bm25 = np.array([3.0, 2.0, 1.0, 0.0])
    dense = np.array([0.0, 0.8, 0.1, 0.9])
    # chunk 1 is 2nd in both lists (2/62); chunks 0 and 3 are 1st in one list and
    # 4th in the other (1/61 + 1/64, smaller) and tie, so chunk order decides.
    assert fusion(bm25=bm25, dense=dense).tolist() == [1, 0, 3, 2]
    assert stop(bm25=bm25).tolist() == [0, 1, 2, 3]


def test_splits_are_repository_disjoint_and_complete():
    tasks = [{"instance_id": f"{r}-{i}", "repo": r} for r, n in
             [("django/django", 5), ("pydata/xarray", 2), ("psf/requests", 1), ("pallets/flask", 1)]
             for i in range(n)]
    s = make_splits(tasks)
    assert set(s["dev"]["ids"]) == {"pydata/xarray-0", "pydata/xarray-1"}
    assert s["folds"]["small"]["repos"] == ["psf/requests", "pallets/flask"]
    ids = s["dev"]["ids"] + [i for f in s["folds"].values() for i in f["ids"]]
    assert sorted(ids) == sorted(t["instance_id"] for t in tasks)


def edges(rows):
    return pl.DataFrame(rows, schema=["src", "dst", "type"], orient="row")


IDS = ["a", "b", "c", "d", "e"]
# BM25 raw scores; min-max normalised: a=1.0, b=0.75, c=0.5, d=0.25, e=0.0
BM25 = np.array([4.0, 3.0, 2.0, 1.0, 0.0])


def test_expand_without_neighbours_is_stop():
    e = edges([("c", "d", "inherits")])  # wrong group, so ignored
    for direction in ("out", "both"):
        got = expand(BM25, IDS, e, "calls", direction, lam=0.9, k=2)
        assert got.tolist() == stop(bm25=BM25).tolist()


def test_expand_out_boosts_callee_of_top_parent():
    # a (1.0) calls e (0.0): e gets 0.9 * 1.0 = 0.9, above b (0.75)
    got = expand(BM25, IDS, edges([("a", "e", "invokes")]), "calls", "out", lam=0.9, k=2)
    assert got.tolist() == [0, 4, 1, 2, 3]


def test_expand_keeps_higher_own_score_and_best_parent():
    # e is called by a (1.0) and b (0.75): best parent wins, 0.6 * 1.0 = 0.6 (< b, > c)
    # d calls nothing; c is a callee of b but its own 0.5 beats 0.6 * 0.75 = 0.45
    e = edges([("a", "e", "invokes"), ("b", "e", "invokes"), ("b", "c", "invokes")])
    got = expand(BM25, IDS, e, "calls", "out", lam=0.6, k=2)
    assert got.tolist() == [0, 1, 4, 2, 3]


def test_expand_direction_and_parent_count():
    # e calls a: only "both" follows the edge backwards to e
    e = edges([("e", "a", "invokes")])
    assert expand(BM25, IDS, e, "calls", "out", lam=0.9, k=2).tolist() == [0, 1, 2, 3, 4]
    assert expand(BM25, IDS, e, "calls", "both", lam=0.9, k=2).tolist() == [0, 4, 1, 2, 3]
    # d is not among the k = 2 parents, so its callee e is not boosted
    e = edges([("d", "e", "invokes")])
    assert expand(BM25, IDS, e, "calls", "out", lam=0.9, k=2).tolist() == [0, 1, 2, 3, 4]


def test_imports_group_includes_containment():
    got = expand(BM25, IDS, edges([("a", "e", "contains")]), "imports", "out", lam=0.9, k=1)
    assert got.tolist() == [0, 4, 1, 2, 3]
