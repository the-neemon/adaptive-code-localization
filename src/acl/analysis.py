"""Phase A analysis: per-action gains over the seed, breakdowns, signature table.

The gain of action a on issue i is the proposal's eq. 2,
Delta(i, a) = Score(S_i^a) - Score(S_i^stop), by default for our nDCG@300.

Uncertainty is a repository-clustered bootstrap (the proposal): whole
repositories are resampled with replacement, because issues of one repository
are not independent (django alone is 41% of the tasks).

The signature table bins every post-seed feature (booleans as they are,
numbers into thirds) and reports, for every bin and every pair of bins of two
different features, each action's mean gain, its interval and the share of
issues it helps and hurts. With hundreds of cells, about 5% of them will
exclude zero by chance at the 95% level; the table describes, and the router
(acl.router) is what gets tested on held-out repositories.

Usage: python -m acl.analysis [--budget 300] [--metric ndcg]
  writes data/results/phase_a_summary.md and data/results/signature_table.csv
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


RESULTS = Path("data/results")
# Proposal's cost order for ties: stop < expand(e) < dense < fusion.
COST_ORDER = ["stop", "expand_inherit", "expand_imports", "expand_calls", "dense", "fusion"]
BOOL_FEATURES = ["has_traceback", "names_py_path", "names_repo_file", "has_env_dump", "is_pro"]
NUM_FEATURES = ["n_known_identifiers", "issue_tokens", "bm25_margin", "bm25_topk_margin",
                "seed_files", "top_degree", "top_degree_calls", "top_degree_imports",
                "top_degree_inherit", "topk_degree_mean", "bm25_dense_overlap"]
INT_FEATURES = {"n_known_identifiers", "issue_tokens", "seed_files", "top_degree",
                "top_degree_calls", "top_degree_imports", "top_degree_inherit"}
N_BOOT = 2000
SEED = 0


@dataclass
class Data:
    ids: list[str]
    repo: np.ndarray                 # (n,) repository of each issue
    split: np.ndarray                # (n,) "dev" or fold name
    score: dict[str, np.ndarray]     # action -> (n,) metric value
    feats: dict[str, np.ndarray]     # feature -> (n,) value (bools as 0/1)

    @property
    def actions(self) -> list[str]:
        return [a for a in COST_ORDER if a in self.score]

    def delta(self, a: str) -> np.ndarray:
        return self.score[a] - self.score["stop"]

    def subset(self, mask: np.ndarray) -> "Data":
        return Data([i for i, m in zip(self.ids, mask) if m], self.repo[mask], self.split[mask],
                    {a: v[mask] for a, v in self.score.items()},
                    {f: v[mask] for f, v in self.feats.items()})


def load(budget: int = 300, metric: str = "ndcg", root: Path = RESULTS) -> Data:
    score: dict[str, dict[str, float]] = defaultdict(dict)
    meta: dict[str, tuple[str, str]] = {}
    for line in open(root / "phase_a.jsonl"):
        r = json.loads(line)
        if r["budget"] == budget:
            score[r["action"]][r["instance_id"]] = r["metrics"][metric]
            meta[r["instance_id"]] = (r["repo"], r["split"])
    feats = {json.loads(l)["instance_id"]: json.loads(l) for l in open(root / "features.jsonl")}
    ids = sorted(i for i in meta if i in feats and all(i in s for s in score.values()))
    f = {k: np.array([float(feats[i][k]) for i in ids]) for k in BOOL_FEATURES[:-1] + NUM_FEATURES}
    f["is_pro"] = np.array([float(feats[i]["source"] == "pro") for i in ids])
    return Data(ids, np.array([meta[i][0] for i in ids]), np.array([meta[i][1] for i in ids]),
                {a: np.array([s[i] for i in ids]) for a, s in score.items()}, f)


# ---------------------------------------------------------------- statistics

def boot_means(values: np.ndarray, repos: np.ndarray, n_boot: int = N_BOOT,
               rng: np.random.Generator | None = None) -> np.ndarray:
    """Repository-clustered bootstrap distribution of the mean of `values`."""
    rng = rng or np.random.default_rng(SEED)
    uniq, inv = np.unique(repos, return_inverse=True)
    sums = np.bincount(inv, weights=values, minlength=len(uniq))
    counts = np.bincount(inv, minlength=len(uniq)).astype(float)
    w = rng.multinomial(len(uniq), np.full(len(uniq), 1 / len(uniq)), size=n_boot)
    return (w @ sums) / (w @ counts)


def ci(values: np.ndarray, repos: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    if len(values) == 0:
        return (math.nan, math.nan)
    m = boot_means(values, repos)
    a = (1 - level) / 2
    return float(np.quantile(m, a)), float(np.quantile(m, 1 - a))


def helps_hurts(d: np.ndarray, eps: float = 1e-9) -> tuple[float, float]:
    return float(np.mean(d > eps)), float(np.mean(d < -eps))


# ---------------------------------------------------------------- binning

@dataclass(frozen=True)
class Cond:
    """One feature bin: lo < x <= hi (bools: x == value)."""
    feature: str
    lo: float
    hi: float
    label: str

    def mask(self, feats: dict[str, np.ndarray]) -> np.ndarray:
        x = feats[self.feature]
        return (x > self.lo) & (x <= self.hi)


def _fmt(feature: str, x: float) -> str:
    return f"{x:.0f}" if feature in INT_FEATURES else f"{x:.2f}"


def bins(feats: dict[str, np.ndarray]) -> list[Cond]:
    """Booleans split on their value; numbers into thirds at cut points from `feats`.

    Repeated cut points (e.g. a feature that is 0 for most issues) are merged,
    so such a feature gets two bins or none.
    """
    out = []
    for f in BOOL_FEATURES:
        out += [Cond(f, 0.5, math.inf, f), Cond(f, -math.inf, 0.5, f"not {f}")]
    for f in NUM_FEATURES:
        cuts = sorted(set(np.quantile(feats[f], [1 / 3, 2 / 3]).tolist()))
        cuts = [c for c in cuts if c < feats[f].max()]
        if not cuts:
            continue  # constant: one bin holding every issue says nothing
        edges = [-math.inf] + cuts + [math.inf]
        for lo, hi in zip(edges, edges[1:]):
            if lo == -math.inf:
                label = f"{f} <= {_fmt(f, hi)}"
            elif hi == math.inf:
                label = f"{f} > {_fmt(f, lo)}"
            else:
                label = f"{_fmt(f, lo)} < {f} <= {_fmt(f, hi)}"
            out.append(Cond(f, lo, hi, label))
    return [c for c in out if c.mask(feats).any()]


def conditions(feats: dict[str, np.ndarray]) -> list[tuple[Cond, ...]]:
    """Every single bin, and every pair of bins of two different features."""
    single = bins(feats)
    pairs = [(a, b) for a, b in itertools.combinations(single, 2) if a.feature != b.feature]
    return [(c,) for c in single] + pairs


def cond_mask(cond: tuple[Cond, ...], feats: dict[str, np.ndarray]) -> np.ndarray:
    m = np.ones(len(next(iter(feats.values()))), dtype=bool)
    for c in cond:
        m &= c.mask(feats)
    return m


def cond_label(cond: tuple[Cond, ...]) -> str:
    return " AND ".join(c.label for c in cond)


# ---------------------------------------------------------------- reports

def summary(d: Data, budget: int, metric: str) -> str:
    acts = d.actions
    means = {a: float(d.score[a].mean()) for a in acts}
    best = max(acts, key=lambda a: (means[a], -COST_ORDER.index(a)))
    oracle = np.max(np.stack([d.score[a] for a in acts]), axis=0)
    lines = [f"## Phase A, {metric}@{budget}, {len(d.ids)} issues", "",
             "| action | mean | gain vs stop | 95% CI (repo bootstrap) | helps | hurts |",
             "|---|---|---|---|---|---|"]
    for a in acts:
        dl = d.delta(a)
        lo, hi = ci(dl, d.repo)
        h, u = helps_hurts(dl)
        lines.append(f"| {a} | {means[a]:.4f} | {dl.mean():+.4f} | [{lo:+.4f}, {hi:+.4f}] | {h:.0%} | {u:.0%} |")
    lo, hi = ci(oracle, d.repo)
    lines += ["", f"Best fixed action: **{best}** ({means[best]:.4f}). Per-issue oracle over "
                  f"{len(acts)} actions: {oracle.mean():.4f} [{lo:.4f}, {hi:.4f}], "
                  f"+{oracle.mean() / means[best] - 1:.0%} over the best fixed action.", ""]
    wins = {a: 0 for a in acts}
    for i in range(len(d.ids)):
        top = max(d.score[a][i] for a in acts)
        # the cheapest action reaching the per-issue best
        wins[next(a for a in acts if d.score[a][i] >= top - 1e-12)] += 1
    lines += ["Oracle choice (cheapest action reaching the per-issue best): " +
              ", ".join(f"{a} {n / len(d.ids):.0%}" for a, n in wins.items()), ""]
    return "\n".join(lines)


def breakdown(d: Data, groups: dict[str, np.ndarray], title: str) -> str:
    acts = [a for a in d.actions if a != "stop"]
    lines = [f"### Gain vs stop by {title}", "",
             "| group | n | stop | " + " | ".join(acts) + " |", "|---|---|---|" + "---|" * len(acts)]
    for g, m in groups.items():
        if not m.any():
            continue
        cells = [f"{d.delta(a)[m].mean():+.4f}" for a in acts]
        lines.append(f"| {g} | {m.sum()} | {d.score['stop'][m].mean():.4f} | " + " | ".join(cells) + " |")
    return "\n".join(lines + [""])


def signature_table(d: Data) -> list[dict]:
    rows = []
    for cond in conditions(d.feats):
        m = cond_mask(cond, d.feats)
        if m.sum() < 5:
            continue
        for a in d.actions:
            if a == "stop":
                continue
            dl = d.delta(a)[m]
            lo, hi = ci(dl, d.repo[m])
            h, u = helps_hurts(dl)
            rows.append({"condition": cond_label(cond), "n_conditions": len(cond), "action": a,
                         "n": int(m.sum()), "n_repos": len(set(d.repo[m])), "mean_gain": float(dl.mean()),
                         "ci_lo": lo, "ci_hi": hi, "helps": h, "hurts": u})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=300)
    ap.add_argument("--metric", default="ndcg")
    args = ap.parse_args()
    d = load(args.budget, args.metric)

    parts = [summary(d, args.budget, args.metric)]
    off_dev = d.subset(d.split != "dev")
    parts += ["(Outside dev only, since lambda and k were tuned on dev.)", summary(off_dev, args.budget, args.metric)]
    parts.append(breakdown(d, {r: d.repo == r for r in sorted(set(d.repo), key=lambda r: -(d.repo == r).sum())},
                           "repository"))
    parts.append(breakdown(d, {"Pro": d.feats["is_pro"] == 1, "Verified": d.feats["is_pro"] == 0}, "source"))
    parts.append(breakdown(d, {"names a repo file": d.feats["names_repo_file"] == 1,
                               "does not": d.feats["names_repo_file"] == 0}, "whether the issue names a target file"))
    report = "\n".join(parts)
    (RESULTS / "phase_a_summary.md").write_text(report)
    print(report)

    rows = signature_table(d)
    keys = list(rows[0])
    with (RESULTS / "signature_table.csv").open("w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(f'"{r[k]}"' if k == "condition" else
                             (f"{r[k]:.5f}" if isinstance(r[k], float) else str(r[k])) for k in keys) + "\n")
    for n_cond in (1, 2):
        sub = [r for r in rows if r["n_conditions"] == n_cond and r["n_repos"] >= 3 and r["n"] >= 20]
        sig = [r for r in sub if r["ci_lo"] > 0 or r["ci_hi"] < 0]
        print(f"\n{'single' if n_cond == 1 else 'pair'} cells (n >= 20, >= 3 repos): {len(sub)}; "
              f"CI excludes 0: {len(sig)} (about {0.05 * len(sub):.0f} expected by chance)")
        for r in sorted(sig, key=lambda r: -abs(r["mean_gain"]) * math.sqrt(r["n"]))[:15]:
            print(f"  {r['action']:15s} {r['mean_gain']:+.4f} [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] "
                  f"n={r['n']:3d} repos={r['n_repos']:2d}  {r['condition']}")
    print(f"\n-> {RESULTS / 'phase_a_summary.md'}, {RESULTS / 'signature_table.csv'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
