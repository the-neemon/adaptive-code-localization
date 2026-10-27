"""Tier 2 router: a lightweight classifier over the post-seed features (proposal 3.3).

Protocol, fixed on 2026-10-27 before any Tier 2 result on the real data:

- Features: the 16 post-seed features Tier 1 uses (acl.analysis BOOL_FEATURES
  and NUM_FEATURES). Nothing reads the patch or the ground truth.
- Label: the best action per training issue, argmax_a Delta(i, a) on nDCG at
  the budget, ties broken toward the cheaper action
  (stop < expand(e) < dense < fusion, acl.analysis.COST_ORDER).
- Grid, in this order (ties in the dev score go to the earlier, simpler one):
    logistic regression on standardised features, C in {0.01, 0.1, 1, 10};
    gradient-boosted trees, depth in {1, 2, 3}, 50 or 100 trees, learning
    rate 0.1; random_state 16.
- Tuning on dev only. For each held-out fold f: fit every setting on the
  issues outside f and outside dev, route the dev issues, and keep the setting
  with the best mean nDCG of its routed actions on dev. Refit that setting on
  every issue outside f (the other folds plus dev) and route f.
- Reported on the same 444 held-out issues as Tier 1, with acl.heldout's
  GapClosed, repository-clustered intervals and help/hurt shares, next to
  Tier 1's router and router_1, plus the paired gain over router_1.

Usage: python -m acl.tier2  (writes data/results/tier2.md and tier2.json)
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from acl import analysis, heldout
from acl.analysis import BOOL_FEATURES, COST_ORDER, NUM_FEATURES, RESULTS, Data

SEED = 16
FEATURES = BOOL_FEATURES + NUM_FEATURES
GRID = ([("logreg", {"C": c}) for c in (0.01, 0.1, 1, 10)]
        + [("gbt", {"max_depth": dep, "n_estimators": n}) for dep in (1, 2, 3) for n in (50, 100)])


def labels(d: Data) -> np.ndarray:
    """Best action per issue; the cost order breaks ties because max() keeps the first."""
    acts = d.actions  # already in COST_ORDER
    best = [max(acts, key=lambda a: d.score[a][i]) for i in range(len(d.ids))]
    return np.array(best, dtype=object)


def matrix(d: Data) -> np.ndarray:
    return np.column_stack([d.feats[f] for f in FEATURES])


def make(kind: str, params: dict):
    if kind == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(C=params["C"], max_iter=5000))
    return GradientBoostingClassifier(learning_rate=0.1, random_state=SEED, **params)


def fit_predict(kind: str, params: dict, train: Data, test: Data) -> np.ndarray:
    y = labels(train)
    if len(set(y)) == 1:
        return np.array([y[0]] * len(test.ids), dtype=object)
    model = make(kind, params).fit(matrix(train), y)
    return model.predict(matrix(test)).astype(object)


def mean_score(d: Data, choice: np.ndarray) -> float:
    return float(np.mean([d.score[a][i] for i, a in enumerate(choice)]))


def held_out_choices(d: Data) -> tuple[np.ndarray, np.ndarray, dict]:
    """Tier 2 action per held-out issue, the held-out mask and each fold's chosen setting."""
    choice = np.empty(len(d.ids), dtype=object)
    held = np.zeros(len(d.ids), dtype=bool)
    chosen = {}
    dev = d.split == "dev"
    for fold in sorted(set(d.split) - {"dev"}):
        test = d.split == fold
        inner, dev_d = d.subset(~test & ~dev), d.subset(dev)
        scores = [mean_score(dev_d, fit_predict(k, p, inner, dev_d)) for k, p in GRID]
        best = int(np.argmax(scores))  # first maximum = simplest
        kind, params = GRID[best]
        choice[test] = fit_predict(kind, params, d.subset(~test), d.subset(test))
        held |= test
        chosen[fold] = {"model": kind, **params, "dev_ndcg": scores[best]}
    return choice[held], held, chosen


def main() -> None:
    report, out = ["# Tier 2 router (classifier) on held-out repositories", ""], {}
    for budget in heldout.BUDGETS:
        d = analysis.load(budget, "ndcg")
        tier1 = heldout.held_out_choices(d)
        t2, held, chosen = held_out_choices(d)
        assert (held == tier1["held"]).all()
        choice = {**tier1["choice"], "tier2": t2}
        e = heldout.evaluate(d, held, choice, systems=("router", "router_1", "tier2"))
        idx = np.flatnonzero(held)
        diff = (heldout.scores_of(d, t2, held) - heldout.scores_of(d, choice["router_1"], held))
        b = analysis.boot_means(diff, d.repo[held])
        e["tier2_vs_router_1"] = {"gain": float(diff.mean()),
                                  "ci": [float(x) for x in np.quantile(b, [0.025, 0.975])]}
        oracle_action = labels(d)[idx]
        e["tier2_picks_oracle_action"] = float(np.mean(t2 == oracle_action))
        e["tier2_action_shares"] = {a: float(np.mean(t2 == a)) for a in COST_ORDER}
        e["chosen_settings"] = chosen
        out[f"ndcg@{budget}"] = e
        m = e["mean"]
        report += [f"### nDCG@{budget} ({e['n']} held-out issues)", "", "| system | mean |", "|---|---|"]
        report += [f"| {k} | {m[k]:.4f} |" for k in ("stop", "fixed", "router", "router_1", "tier2", "oracle")]
        report.append("")
        for k in ("router", "router_1", "tier2"):
            r = e[k]
            report.append(f"- {k}: gain vs fixed {r['gain_vs_fixed']:+.4f} [{r['gain_ci'][0]:+.4f}, "
                          f"{r['gain_ci'][1]:+.4f}]; GapClosed {r['gap_closed']:+.3f} [{r['gap_closed_ci'][0]:+.3f}, "
                          f"{r['gap_closed_ci'][1]:+.3f}]; helps {r['helps']:.0%}, hurts {r['hurts']:.0%}; "
                          f"routes {r['routed_away_from_default']:.0%} away from the default")
        g = e["tier2_vs_router_1"]
        report += [f"- tier2 vs router_1: {g['gain']:+.4f} [{g['ci'][0]:+.4f}, {g['ci'][1]:+.4f}]",
                   f"- tier2 picks the oracle's action on {e['tier2_picks_oracle_action']:.0%} of issues; "
                   "actions: " + ", ".join(f"{a} {v:.0%}" for a, v in e["tier2_action_shares"].items() if v),
                   "- settings chosen on dev: " + "; ".join(
                       f"{f}: {c['model']} " + ", ".join(f"{k}={v}" for k, v in c.items() if k not in ("model", "dev_ndcg"))
                       for f, c in chosen.items()), ""]
        if budget == 300:
            # Same decisions, other metrics at B = 300.
            for metric, keep in (("recall", None), ("file_recall_patch", "verified")):
                dm = analysis.load(budget, metric)
                assert dm.ids == d.ids
                mask = held if keep is None else held & (dm.feats["is_pro"] == 0)
                sub = {k: c[mask[held]] for k, c in choice.items()}
                em = heldout.evaluate(dm, mask, sub, systems=("router", "router_1", "tier2"))
                out[f"{metric}@{budget}"] = em
                report += heldout.fmt_main(em, f"{metric}@{budget}" + (" (SWE-bench Verified only)" if keep else ""))
                t = em["tier2"]
                report += [f"- tier2: mean {em['mean']['tier2']:.4f}, gain vs fixed {t['gain_vs_fixed']:+.4f} [{t['gain_ci'][0]:+.4f}, "
                           f"{t['gain_ci'][1]:+.4f}], GapClosed {t['gap_closed']:+.3f}", ""]
    (RESULTS / "tier2.md").write_text("\n".join(report))
    (RESULTS / "tier2.json").write_text(json.dumps(out, indent=1, default=float))
    print("\n".join(report))
    print(f"-> {RESULTS / 'tier2.md'}, {RESULTS / 'tier2.json'}")


if __name__ == "__main__":
    main()
