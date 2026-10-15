import numpy as np
import pytest

from acl.actions import fusion, order, ranks, rrf, stop
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
