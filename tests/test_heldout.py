import math

import numpy as np

from acl.analysis import BOOL_FEATURES, NUM_FEATURES, Data
from acl.heldout import evaluate, gap_closed, held_out_choices


def planted(seed=16, n_repos=10, per_repo=40):
    """Fusion is best on average; expand_calls is much better whenever has_traceback = 1."""
    rng = np.random.default_rng(seed)
    n = n_repos * per_repo
    repo = np.repeat([f"r{i}" for i in range(n_repos)], per_repo)
    split = np.where(np.isin(repo, ["r0", "r1"]), "dev", repo)
    feats = {k: rng.integers(0, 2, n).astype(float) for k in BOOL_FEATURES}
    feats.update({k: rng.random(n) for k in NUM_FEATURES})
    tb = feats["has_traceback"]
    noise = lambda: 0.02 * rng.standard_normal(n)
    score = {"stop": 0.30 + noise(), "fusion": 0.40 + noise(),
             "expand_calls": 0.35 + 0.15 * tb - 0.05 * (1 - tb) + noise()}
    return Data([f"i{j}" for j in range(n)], repo, split, score, feats)


def test_gap_closed():
    assert gap_closed(0.25, 0.20, 0.30) == 0.5
    assert gap_closed(0.20, 0.20, 0.30) == 0.0
    assert math.isnan(gap_closed(0.2, 0.3, 0.3))


def test_planted_rule_generalises_to_held_out_repos():
    d = planted()
    h = held_out_choices(d)
    assert h["held"].sum() == 320  # 8 non-dev repos x 40
    for r in h["routers"].values():
        assert r.default == "fusion"
        assert r.rules[0].action == "expand_calls" and r.rules[0].cond[0].feature == "has_traceback"
    e = evaluate(d, h["held"], h["choice"])
    assert e["router"]["gain_vs_fixed"] > 0.04
    assert 0.5 < e["router"]["gap_closed"] <= 1.0
    assert e["router"]["gain_ci"][0] > 0


def test_held_out_scores_never_influence_their_own_routing():
    d = planted()
    before = held_out_choices(d)
    # Corrupt fold r5's scores so that, if they leaked, its rules would flip.
    m = d.split == "r5"
    d.score["expand_calls"][m] = 0.0
    d.score["stop"][m] = 1.0
    after = held_out_choices(d)
    r5 = d.split[before["held"]] == "r5"
    for s in ("fixed", "router", "router_1"):
        assert list(after["choice"][s][r5]) == list(before["choice"][s][r5])
    # ...while other folds, whose training data now includes r5, may change.
    assert after["routers"]["r5"].to_json() == before["routers"]["r5"].to_json()
