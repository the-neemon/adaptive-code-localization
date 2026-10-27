import numpy as np

from acl.tier2 import held_out_choices, labels
from tests.test_heldout import planted


def test_labels_break_ties_toward_the_cheaper_action():
    d = planted()
    for a in d.score:
        d.score[a][:] = 0.5
    d.score["fusion"][0] = 0.9
    y = labels(d)
    assert y[0] == "fusion"
    assert set(y[1:]) == {"stop"}  # all tied: stop is cheapest


def test_tier2_learns_a_planted_signal_on_held_out_repos():
    d = planted()
    choice, held, chosen = held_out_choices(d)
    tb = d.feats["has_traceback"][held] == 1
    # expand_calls wins when has_traceback, fusion otherwise
    assert np.mean(choice[tb] == "expand_calls") > 0.9
    assert np.mean(choice[~tb] == "fusion") > 0.9
    assert set(chosen) == set(d.split) - {"dev"}


def test_held_out_labels_never_influence_their_own_routing():
    d = planted()
    before, held, _ = held_out_choices(d)
    m = d.split == "r5"
    d.score["expand_calls"][m] = 0.0
    d.score["stop"][m] = 1.0
    after, _, _ = held_out_choices(d)
    r5 = d.split[held] == "r5"
    assert list(after[r5]) == list(before[r5])
