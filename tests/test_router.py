import numpy as np
import pytest

from acl.analysis import BOOL_FEATURES, NUM_FEATURES, Data, bins, boot_means, cond_mask, conditions
from acl.router import fit


def test_boot_means_single_repo_is_the_mean():
    v = np.array([1.0, 2.0, 6.0])
    assert np.allclose(boot_means(v, np.array(["r"] * 3), n_boot=50), 3.0)


def test_boot_means_resamples_whole_repos():
    # repo a: mean 0 over 3 issues, repo b: mean 1 over 1 issue. Every draw is
    # aa (0), bb (1) or ab/ba (1/4), never an issue-level mix.
    v = np.array([0.0, 0.0, 0.0, 1.0])
    m = boot_means(v, np.array(["a", "a", "a", "b"]), n_boot=500)
    assert set(np.round(m, 6)) <= {0.0, 0.25, 1.0}


def feats(n, rng):
    f = {k: rng.integers(0, 2, n).astype(float) for k in BOOL_FEATURES}
    f.update({k: rng.random(n) for k in NUM_FEATURES})
    return f


def test_bins_bools_and_thirds():
    f = {k: np.zeros(9) for k in BOOL_FEATURES + NUM_FEATURES}
    f["has_traceback"] = np.array([1, 0] * 4 + [1.0])
    f["issue_tokens"] = np.arange(9.0)
    labels = {c.label for c in bins(f)}
    assert {"has_traceback", "not has_traceback"} <= labels
    assert {"issue_tokens <= 3", "3 < issue_tokens <= 5", "issue_tokens > 5"} <= labels
    # a constant feature gets no bin at all; bins of a feature partition the issues
    assert not any(c.feature == "seed_files" for c in bins(f))
    tok = [c for c in bins(f) if c.feature == "issue_tokens"]
    assert sum(c.mask(f).sum() for c in tok) == 9


def test_pairs_never_combine_a_feature_with_itself():
    rng = np.random.default_rng(0)
    for cond in conditions(feats(60, rng)):
        assert len({c.feature for c in cond}) == len(cond)


def test_fit_recovers_a_planted_rule():
    rng = np.random.default_rng(1)
    n = 240
    f = feats(n, rng)
    tb = f["has_traceback"] == 1
    # fusion is best overall; on traceback issues expand_calls is better by 0.1
    score = {"stop": np.full(n, 0.2), "fusion": np.full(n, 0.3),
             "expand_calls": np.where(tb, 0.4, 0.1)}
    d = Data([f"i{k}" for k in range(n)], np.array([f"r{k % 6}" for k in range(n)]),
             np.array(["x"] * n), score, f)
    r = fit(d)
    assert r.default == "fusion"
    assert len(r.rules) == 1 and r.rules[0].action == "expand_calls"
    assert [c.label for c in r.rules[0].cond] == ["has_traceback"]
    assert r.rules[0].mean_gain == pytest.approx(0.1)
    routed = r.route(f)
    assert (routed[tb] == "expand_calls").all() and (routed[~tb] == "fusion").all()
    assert cond_mask(r.rules[0].cond, f).sum() == tb.sum()


def test_fit_without_signal_has_no_rules():
    rng = np.random.default_rng(2)
    n = 120
    f = feats(n, rng)
    score = {"stop": np.full(n, 0.2), "fusion": np.full(n, 0.3), "expand_calls": np.full(n, 0.25)}
    d = Data([f"i{k}" for k in range(n)], np.array([f"r{k % 4}" for k in range(n)]),
             np.array(["x"] * n), score, f)
    assert fit(d).rules == []
