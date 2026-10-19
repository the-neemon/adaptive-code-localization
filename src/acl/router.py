"""Tier 1 router: an ordered if/else rule list, produced by a fixed procedure.

The procedure (decided 2026-10-19, fixed before any held-out evaluation):

1. Default action: the best fixed action on the training issues (highest mean
   nDCG@300; ties go to the cheaper action, stop < expand(e) < dense < fusion).
2. Conditions: every feature bin and every pair of bins of two different
   features (acl.analysis.conditions), with bin edges (thirds) computed on the
   training issues only.
3. Greedy rule list. Among the training issues not yet covered by an earlier
   rule, pick the (condition, action) with the largest total gain over the
   default, sum of Score(action) - Score(default), subject to:
     - at least MIN_SUPPORT issues from at least MIN_REPOS repositories;
     - a repository-clustered bootstrap lower bound (one-sided, LEVEL) of the
       mean gain above zero.
   Add it, remove the issues it covers, and repeat up to MAX_RULES times or
   until no candidate qualifies.
4. Routing: the first rule whose condition holds picks the action; otherwise
   the default.

For evaluation (Stage 5) the procedure is fit on all folds but one and applied
to the held-out repository; here it is fit on every training set to see which
rules it finds and how stable they are. Held-out issues are never scored.

Usage: python -m acl.router [--per-fold]  (rules on all issues; --per-fold: on each fold's training set)
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field

import numpy as np

from acl import analysis
from acl.analysis import COST_ORDER, Cond, Data, boot_means, cond_label, cond_mask, conditions

MIN_SUPPORT = 20
MIN_REPOS = 3
MAX_RULES = 5
LEVEL = 0.95
OBJECTIVE = ("ndcg", 300)


@dataclass
class Rule:
    cond: tuple[Cond, ...]
    action: str
    n: int
    mean_gain: float
    lower: float

    def __str__(self) -> str:
        return (f"if {cond_label(self.cond)} -> {self.action}  "
                f"(n={self.n}, gain vs default {self.mean_gain:+.4f}, lower bound {self.lower:+.4f})")


@dataclass
class Router:
    default: str
    rules: list[Rule] = field(default_factory=list)

    def route(self, feats: dict[str, np.ndarray]) -> np.ndarray:
        n = len(next(iter(feats.values())))
        out = np.array([self.default] * n, dtype=object)
        free = np.ones(n, dtype=bool)
        for r in self.rules:
            m = free & cond_mask(r.cond, feats)
            out[m] = r.action
            free &= ~m
        return out

    def to_json(self) -> dict:
        return {"default": self.default,
                "rules": [{"if": [{"feature": c.feature, "lo": c.lo, "hi": c.hi, "label": c.label}
                                  for c in r.cond], "then": r.action, "n": r.n,
                           "mean_gain": r.mean_gain, "lower": r.lower} for r in self.rules]}


def best_fixed(d: Data) -> str:
    return max(d.actions, key=lambda a: (d.score[a].mean(), -COST_ORDER.index(a)))


def fit(d: Data) -> Router:
    default = best_fixed(d)
    router = Router(default)
    candidates = [(c, cond_mask(c, d.feats)) for c in conditions(d.feats)]
    gains = {a: d.score[a] - d.score[default] for a in d.actions if a != default}
    free = np.ones(len(d.ids), dtype=bool)
    for _ in range(MAX_RULES):
        ranked = []
        for cond, m in candidates:
            m = m & free
            n = int(m.sum())
            if n < MIN_SUPPORT or len(set(d.repo[m])) < MIN_REPOS:
                continue
            for a, g in gains.items():
                total = float(g[m].sum())
                if total > 0:
                    ranked.append(((total, -COST_ORDER.index(a)), cond, a, m))
        # The largest total gain among candidates whose lower bound clears zero.
        chosen = None
        for _, cond, a, m in sorted(ranked, key=lambda x: x[0], reverse=True):
            lower = float(np.quantile(boot_means(gains[a][m], d.repo[m]), 1 - LEVEL))
            if lower > 0:
                chosen = (cond, a, m, lower)
                break
        if chosen is None:
            break
        cond, a, m, lower = chosen
        router.rules.append(Rule(cond, a, int(m.sum()), float(gains[a][m].mean()), lower))
        free &= ~m
    return router


def describe(router: Router, d: Data) -> str:
    routed = router.route(d.feats)
    score = np.array([d.score[a][i] for i, a in enumerate(routed)])
    lines = [f"default: {router.default}"] + [f"  {i + 1}. {r}" for i, r in enumerate(router.rules)]
    lines.append(f"  on its own training issues: router {score.mean():.4f} vs default "
                 f"{d.score[router.default].mean():.4f} (in-sample, optimistic)")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-fold", action="store_true")
    args = ap.parse_args()
    d = analysis.load(OBJECTIVE[1], OBJECTIVE[0])
    out = {}
    print(f"== rules fit on all {len(d.ids)} issues")
    r = fit(d)
    print(describe(r, d))
    out["all"] = r.to_json()
    if args.per_fold:
        for fold in sorted(set(d.split) - {"dev"}):
            train = d.subset(d.split != fold)
            r = fit(train)
            print(f"\n== fold {fold}: fit on the other {len(train.ids)} issues (held-out issues not scored)")
            print(describe(r, train))
            out[fold] = r.to_json()
    path = analysis.RESULTS / "router_rules.json"
    path.write_text(json.dumps(out, indent=1, default=float))
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
