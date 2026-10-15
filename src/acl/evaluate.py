"""Evaluation harness: score a ranked region list for one task at a line budget.

Two families of metrics are reported side by side.

Official: SWE-Explore's released scorer (`acl.swe_explore_eval`, vendored
verbatim), called on the same predictions. These are comparable with the
paper's tables. Two of its choices make them a poor gain signal for us:
  - core ranges are not clipped to the file, so lines past the end of a file
    (`sed -n 1,240p` on a 146-line file) stay in the recall denominator;
  - nDCG's ideal DCG is the best reordering of the *returned* regions, so a
    list whose only core hit is at rank 1 scores 1.0 and finding more core
    code is not rewarded.

Ours (the primary metrics):
  - Core lines are clipped to the file's length at the pre-fix commit, and
    lines in files absent from the commit are dropped. Everything left can in
    principle be retrieved (except non-.py files, which we keep: they are part
    of the task, and missing them is a real limit of a .py index).
  - nDCG@B follows the paper's Appendix definition: the ideal DCG greedily
    places ground-truth core regions, each cut to the remaining budget, in
    order of marginal gain (ties: shorter, then path). The ideal is fixed per
    task and budget, so a gain in nDCG is a gain in DCG, and covering more core
    code counts. Line gains use the official weights (1.5 for lines in
    `main_files`, else 1.0); a line counts once even if regions repeat it.

Predictions are truncated to the budget before scoring, cutting the region
that crosses it (the official recall@budget convention, and what
`acl.bm25.fill_budget` produces).
"""
from __future__ import annotations

import math
from collections import defaultdict

from acl import swe_explore_eval as official

Region = tuple[str, int, int]
MAIN_WEIGHT = 1.5


def truncate(regions: list[Region], budget: int) -> list[Region]:
    out, used = [], 0
    for p, s, e in regions:
        if used >= budget:
            break
        take = min(e - s + 1, budget - used)
        out.append((p, s, s + take - 1))
        used += take
    return out


def clip(region: Region | dict, counts: dict[str, int]) -> Region | None:
    """Resolve a ground-truth region as the official scorer does, then clip it to the file.

    Returns None when the file is absent at the commit or nothing is left.
    """
    p, s, e = official._normalize_region(region)
    n = counts.get(p)
    if n is None:
        return None
    resolved = official._resolve_interval(p, s, e, counts)
    if resolved is None:
        return None
    s, e = resolved[0], min(resolved[1], n)
    return (p, s, e) if s <= e else None


def lines(regions: list[Region]) -> set[tuple[str, int]]:
    return {(p, ln) for p, s, e in regions for ln in range(s, e + 1)}


class Gold:
    """A task's ground truth, resolved against its file line counts."""

    def __init__(self, task: dict, counts: dict[str, int], blank: dict[str, list[int]] | None = None):
        self.task = task
        self.counts = counts
        self.main_files = set(task["main_files"] or [])
        self.core_regions = [r for r in (clip(x, counts) for x in task["core_regions"]) if r]
        self.core = lines(self.core_regions)
        # Lines a .py chunk index can return: no blank lines (chunks trim the
        # ones between definitions), no other file types.
        blank_set = {(p, ln) for p, lns in (blank or {}).items() for ln in lns}
        self.core_indexable = {(p, ln) for p, ln in self.core if p.endswith(".py")} - blank_set
        opt = [r for r in (clip(x, counts) for x in task["optional_regions"]) if r]
        self.useful = self.core | lines(opt)
        self.core_files = {r["path"] for r in task["core_regions"]}
        self.optional_files = {r["path"] for r in task["optional_regions"]}
        self.patch_files = set(task.get("patch_files") or [])
        self._idcg: dict[int, float] = {}
        # The official scorer's view of the same record.
        self.bench_gt = {
            "read_core_files": task["core_files"],
            "read_core_regions": task["core_regions"],
            "read_optional_files_map": {"all": sorted(self.optional_files)},
            "read_optional_regions_map": {"all": task["optional_regions"]},
            "main_files": task["main_files"],
        }

    def weight(self, path: str) -> float:
        return MAIN_WEIGHT if path in self.main_files else 1.0

    def idcg(self, budget: int) -> float:
        """Ideal DCG at `budget` (paper, Appendix "app:idcg"): greedy by marginal gain."""
        if budget not in self._idcg:
            remaining, covered, dcg, rank = budget, set(), 0.0, 0
            pool = list(self.core_regions)
            while remaining > 0 and pool:
                def gain(r: Region) -> tuple[float, list[int]]:
                    new = [ln for ln in range(r[1], r[2] + 1) if (r[0], ln) not in covered][:remaining]
                    return self.weight(r[0]) * len(new), new
                best = max(pool, key=lambda r: (gain(r)[0], -(r[2] - r[1] + 1), _neg_str(r[0])))
                g, new = gain(best)
                if g <= 0:
                    break
                pool.remove(best)
                # The ideal region is the run of uncovered lines it adds, cut to the budget.
                covered.update((best[0], ln) for ln in new)
                remaining -= len(new)
                dcg += g / math.log2(rank + 2)
                rank += 1
            self._idcg[budget] = dcg
        return self._idcg[budget]


def _neg_str(s: str) -> tuple[int, ...]:
    # max() with ascending path order as the last tie-break.
    return tuple(-ord(c) for c in s)


def score(gold: Gold, regions: list[Region], budget: int) -> dict[str, float]:
    preds = truncate(regions, budget)
    pred_lines = lines(preds)
    core = gold.core
    m: dict[str, float] = {}

    # Coverage
    hit = pred_lines & core
    m["recall"] = len(hit) / len(core) if core else math.nan
    # Perfect recall (proposal: "all core regions covered"): every core region
    # is touched by a returned line. Strict variant: every non-blank .py core
    # line is returned (all-core-lines coverage is unreachable by construction).
    m["perfect_recall"] = float(bool(gold.core_regions) and all(
        lines([r]) & pred_lines for r in gold.core_regions))
    m["perfect_recall_lines"] = float(bool(gold.core_indexable) and gold.core_indexable <= pred_lines)
    m["recall_cap"] = min(1.0, budget / len(core)) if core else math.nan
    m["precision"] = len(hit) / len(pred_lines) if pred_lines else 0.0
    m["context_efficiency"] = len(pred_lines & gold.useful) / len(pred_lines) if pred_lines else 0.0

    # Ranking: DCG over the truncated list against the paper's ideal
    covered, dcg, first = set(), 0.0, math.nan
    for i, (p, s, e) in enumerate(preds):
        new = ({(p, ln) for ln in range(s, e + 1)} & core) - covered
        if new and math.isnan(first):
            first = i + 1
        covered |= new
        dcg += gold.weight(p) * len(new) / math.log2(i + 2)
    idcg = gold.idcg(budget)
    m["ndcg"] = min(dcg / idcg, 1.0) if idcg > 0 else math.nan
    m["first_hit_rank"] = first  # 1-based; NaN when no region touches core

    # Noise: share of returned regions touching neither core nor optional lines
    if preds:
        m["noise_region"] = sum(1 for r in preds if not (lines([r]) & gold.useful)) / len(preds)
    else:
        m["noise_region"] = 0.0

    # Files
    files = {p for p, _, _ in preds}
    m["hit_file"] = len(files & gold.core_files) / len(gold.core_files) if gold.core_files else math.nan
    m["file_recall_patch"] = (len(files & gold.patch_files) / len(gold.patch_files)
                              if gold.patch_files else math.nan)

    # Official SWE-Explore scorer on the same truncated list
    ev = official.ExploreEvaluator.__new__(official.ExploreEvaluator)
    ev._current_file_line_counts = gold.counts
    g = gold.bench_gt
    m["off_recall"] = ev._recall_at_line_budget(preds, g, budget)
    m["off_ndcg"] = ev._ndcg_at_line_budget(preds, g, budget)
    m["off_precision"] = ev.evaluate_precision(preds, g)
    m["off_context_efficiency"] = ev.evaluate_context_efficiency(preds, g)
    m["off_hit_file"] = ev.evaluate_hit_file_rate(preds, g)
    m["off_noise_file"] = ev.evaluate_noise_file_rate(preds, g)
    m["off_noise_region"] = ev.evaluate_noise_region_rate(preds, g)
    m["off_hit_region"] = ev.evaluate_hit_region_rate(preds, g)
    m["off_first_useful_hit"] = ev.evaluate_first_useful_hit(preds, g)
    m["n_core"] = len(core)
    m["n_pred_lines"] = len(pred_lines)
    return m


def summarise(rows: list[dict], keys: list[str]) -> dict[tuple, dict[str, float]]:
    """Mean of every numeric metric per group (NaNs skipped)."""
    acc: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for k, v in r["metrics"].items():
            if isinstance(v, (int, float)) and not math.isnan(v):
                acc[tuple(r[k2] for k2 in keys)][k].append(v)
    return {g: {k: sum(v) / len(v) for k, v in ms.items()} for g, ms in sorted(acc.items())}
