"""Stage 5: the Tier 1 router on held-out repositories.

For each of the 8 folds, the frozen rule procedure (acl.router.fit) is fit on
every other issue (the other folds plus dev) and applied to the held-out
repository's issues, with bin edges from the training issues only. No
held-out issue influences the rules that route it.

Systems compared on the 444 held-out issues (proposal section 4):
- stop: the BM25 seed
- fixed: the best fixed action on that fold's training issues (the router's default)
- router: the full rule list (up to MAX_RULES rules), the main result
- router_1: the same list cut after its first rule, a secondary analysis
  declared before any held-out result existed (docs/stage4.md)
- oracle: the best action per issue in hindsight

GapClosed = (Score(router) - Score(fixed)) / (Score(oracle) - Score(fixed)),
on means over the held-out issues (proposal eq. 3). Intervals are
repository-clustered bootstraps (acl.analysis): whole repositories are
resampled and every system is scored on the same draw.

The router is fit on nDCG at each budget B and scored at that B. At B = 300 its
choices are also scored on line recall and, for SWE-bench Verified issues, on
file-level recall of the developer patch.

Secondary benchmark (SWE-bench Verified, file-level recall of the developer
patch). If data/results/verified_extra/ holds Phase A rows and features for the
90 Verified issues outside the study set (acl.data --verified-extra), the
procedure is fit once on all 500 study issues and routes those 90, which no
tuning or rule fitting ever saw. The Verified issues outside dev are then
reported together (464): the 374 held-out study issues routed by their fold's
router, the 90 by the router fit on the study set. The 36 dev issues (xarray,
pytest) tune the actions and train every router, so they are left out.

Usage: python -m acl.heldout  (writes data/results/heldout.md and heldout.json)
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from acl import analysis, router
from acl.analysis import N_BOOT, RESULTS, SEED, Data

BUDGETS = (100, 300, 500)
SYSTEMS = ("stop", "fixed", "router", "router_1", "oracle")
# Extra retrieval calls beyond the BM25 seed; expansions only read the graph.
EXTRA = RESULTS / "verified_extra"
RETRIEVAL_CALLS = {"stop": 0, "expand_inherit": 0, "expand_imports": 0, "expand_calls": 0,
                   "dense": 1, "fusion": 1}


def held_out_choices(d: Data) -> dict:
    """Per held-out issue: the action each system picks, plus the fitted routers."""
    choice = {s: np.empty(len(d.ids), dtype=object) for s in SYSTEMS if s != "oracle"}
    held = np.zeros(len(d.ids), dtype=bool)
    routers = {}
    for fold in sorted(set(d.split) - {"dev"}):
        test = d.split == fold
        r = router.fit(d.subset(~test))
        feats = {k: v[test] for k, v in d.feats.items()}
        choice["stop"][test] = "stop"
        choice["fixed"][test] = r.default
        choice["router"][test] = r.route(feats)
        choice["router_1"][test] = router.Router(r.default, r.rules[:1]).route(feats)
        held |= test
        routers[fold] = r
    return {"choice": {s: c[held] for s, c in choice.items()}, "held": held, "routers": routers}


def scores_of(d: Data, choice: np.ndarray, held: np.ndarray) -> np.ndarray:
    idx = np.flatnonzero(held)
    return np.array([d.score[a][i] for a, i in zip(choice, idx)])


def boot_pooled(arrs: dict[str, np.ndarray], repos: np.ndarray) -> dict[str, np.ndarray]:
    """Repository-clustered bootstrap means of several per-issue arrays on shared draws."""
    rng = np.random.default_rng(SEED)
    uniq, inv = np.unique(repos, return_inverse=True)
    counts = np.bincount(inv, minlength=len(uniq)).astype(float)
    w = rng.multinomial(len(uniq), np.full(len(uniq), 1 / len(uniq)), size=N_BOOT)
    return {k: (w @ np.bincount(inv, weights=v, minlength=len(uniq))) / (w @ counts)
            for k, v in arrs.items()}


def gap_closed(router_: float, fixed: float, oracle: float) -> float:
    return (router_ - fixed) / (oracle - fixed) if oracle > fixed else float("nan")


def evaluate(d: Data, held: np.ndarray, choice: dict[str, np.ndarray]) -> dict:
    s = {k: scores_of(d, c, held) for k, c in choice.items()}
    idx = np.flatnonzero(held)
    s["oracle"] = np.array([max(d.score[a][i] for a in d.actions) for i in idx])
    repos = d.repo[held]
    b = boot_pooled(s, repos)
    out = {"n": int(held.sum()), "mean": {k: float(v.mean()) for k, v in s.items()}}
    for k in ("router", "router_1"):
        diff = s[k] - s["fixed"]
        out[k] = {
            "gain_vs_fixed": float(diff.mean()),
            "gain_ci": [float(x) for x in np.quantile(b[k] - b["fixed"], [0.025, 0.975])],
            "gain_vs_stop": float((s[k] - s["stop"]).mean()),
            "gap_closed": gap_closed(s[k].mean(), s["fixed"].mean(), s["oracle"].mean()),
            "gap_closed_ci": [float(x) for x in np.nanquantile(
                [gap_closed(r, f, o) for r, f, o in zip(b[k], b["fixed"], b["oracle"])], [0.025, 0.975])],
            "helps": float(np.mean(diff > 1e-9)), "hurts": float(np.mean(diff < -1e-9)),
            "routed_away_from_default": float(np.mean(choice[k] != choice["fixed"])),
        }
    return out


def breakdown(d: Data, held: np.ndarray, choice: dict, groups: dict[str, np.ndarray]) -> list[str]:
    lines = ["| group | n | stop | fixed | router | router_1 | oracle | GapClosed |", "|---|---|---|---|---|---|---|---|"]
    for name, g in groups.items():
        sub = held.copy()
        sub[held] = g
        if sub.sum() == 0:
            continue
        e = evaluate(d, sub, {k: c[g] for k, c in choice.items()})
        m = e["mean"]
        lines.append(f"| {name} | {e['n']} | {m['stop']:.4f} | {m['fixed']:.4f} | {m['router']:.4f} | "
                     f"{m['router_1']:.4f} | {m['oracle']:.4f} | {e['router']['gap_closed']:+.2f} |")
    return lines


def load_cost(budget: int) -> dict[str, dict[str, float]]:
    """Mean seconds and graph nodes visited per (issue, action), from the Phase A rows."""
    cost: dict[tuple[str, str], dict] = {}
    for line in open(RESULTS / "phase_a.jsonl"):
        r = json.loads(line)
        if r["budget"] == budget:
            cost[(r["instance_id"], r["action"])] = {"secs": r["secs"], "nodes": r.get("n_visited", 0)}
    return cost


def cost_of(choice: np.ndarray, ids: list[str], cost: dict) -> dict[str, float]:
    rows = [cost[(i, a)] for i, a in zip(ids, choice)]
    return {"retrieval_calls": float(np.mean([RETRIEVAL_CALLS[a] for a in choice])),
            "graph_nodes": float(np.mean([r["nodes"] for r in rows])),
            "ms": 1000 * float(np.mean([r["secs"] for r in rows]))}


def fmt_main(e: dict, label: str, what: str = "held-out issues") -> list[str]:
    m = e["mean"]
    lines = [f"### {label} ({e['n']} {what})", "",
             "| system | mean |", "|---|---|"] + [f"| {k} | {m[k]:.4f} |" for k in SYSTEMS]
    for k in ("router", "router_1"):
        r = e[k]
        lines.append("")
        lines.append(f"- {k}: gain vs fixed {r['gain_vs_fixed']:+.4f} "
                     f"[{r['gain_ci'][0]:+.4f}, {r['gain_ci'][1]:+.4f}], vs stop {r['gain_vs_stop']:+.4f}; "
                     f"GapClosed {r['gap_closed']:+.3f} [{r['gap_closed_ci'][0]:+.3f}, {r['gap_closed_ci'][1]:+.3f}]; "
                     f"helps {r['helps']:.0%}, hurts {r['hurts']:.0%} of issues vs fixed; "
                     f"routes {r['routed_away_from_default']:.0%} away from the default")
    return lines + [""]


def verified_extra(d: Data, held: np.ndarray, choice: dict, out: dict) -> list[str]:
    """File-level recall on SWE-bench Verified outside dev (374 held-out study issues + 90 extra)."""
    r_all = router.fit(d)  # all 500 study issues; the extra issues were never seen
    de = analysis.load(300, "file_recall_patch", EXTRA)
    all_e = np.ones(len(de.ids), dtype=bool)
    ce = {"stop": np.array(["stop"] * len(de.ids), dtype=object),
          "fixed": np.array([r_all.default] * len(de.ids), dtype=object),
          "router": r_all.route(de.feats),
          "router_1": router.Router(r_all.default, r_all.rules[:1]).route(de.feats)}
    e_extra = evaluate(de, all_e, ce)
    out["file_recall_patch@300_verified_extra"] = e_extra
    lines = fmt_main(e_extra, "file_recall_patch@300, extra Verified issues, router fit on all 500 study issues",
                     "issues")

    # Verified outside dev: per-issue scores for each system, study and extra together.
    ds = analysis.load(300, "file_recall_patch")
    ver = held & (ds.feats["is_pro"] == 0)
    sys_scores = {k: np.concatenate([scores_of(ds, c[ver[held]], ver), scores_of(de, ce[k], all_e)])
                  for k, c in choice.items()}
    fixed_actions = {a: np.concatenate([ds.score[a][ver], de.score[a]]) for a in ds.actions}
    n_ver = int(ver.sum()) + len(de.ids)
    lines += [f"### SWE-bench Verified, file recall@300 of the developer patch ({n_ver} issues: "
              f"{int(ver.sum())} held-out study issues + {len(de.ids)} extra)", "",
              "| system | mean |", "|---|---|"]
    lines += [f"| {k} | {v.mean():.4f} |" for k, v in sys_scores.items()]
    lines += [f"| fixed: {a} | {v.mean():.4f} |" for a, v in fixed_actions.items()]
    lines.append("")
    lines.append(f"(dev issues are excluded: {int((ds.feats['is_pro'] == 0).sum()) - int(ver.sum())} "
                 "Verified issues tune the actions and are not held out)")
    out["file_recall_patch@300_verified_all"] = {"n": n_ver, **{k: float(v.mean()) for k, v in sys_scores.items()},
                                                 **{f"fixed_{a}": float(v.mean()) for a, v in fixed_actions.items()}}
    return lines + [""]


def main() -> None:
    report, out = ["# Stage 5: Tier 1 router on held-out repositories", ""], {}
    for budget in BUDGETS:
        d = analysis.load(budget, "ndcg")
        h = held_out_choices(d)
        held, choice = h["held"], h["choice"]
        e = evaluate(d, held, choice)
        out[f"ndcg@{budget}"] = e
        report += fmt_main(e, f"nDCG@{budget}")
        if budget == 300:
            ids = [i for i, m in zip(d.ids, held) if m]
            out["routers"] = {f: r.to_json() for f, r in h["routers"].items()}
            out["choices"] = {s: dict(zip(ids, map(str, c))) for s, c in choice.items()}
            folds = d.split[held]
            report += ["### By held-out fold (nDCG@300)", ""] + breakdown(
                d, held, choice, {f: folds == f for f in sorted(set(folds))}) + [""]
            names = d.feats["names_repo_file"][held] == 1
            report += ["### By whether the issue names a target file (nDCG@300)", ""] + breakdown(
                d, held, choice, {"names a repo file": names, "does not": ~names}) + [""]
            cost = load_cost(budget)
            report += ["### Cost per issue (B = 300)", "", "| system | extra retrieval calls | graph nodes visited | ms |",
                       "|---|---|---|---|"]
            for s in ("stop", "fixed", "router", "router_1"):
                c = cost_of(choice[s], ids, cost)
                out.setdefault("cost", {})[s] = c
                report.append(f"| {s} | {c['retrieval_calls']:.2f} | {c['graph_nodes']:.1f} | {c['ms']:.1f} |")
            report.append("")
            rules = defaultdict(int)
            for r in h["routers"].values():
                for rule in r.rules:
                    rules[(analysis.cond_label(rule.cond), rule.action)] += 1
            report += ["### Rules across the 8 per-fold routers", "", "| rule | action | folds |", "|---|---|---|"]
            report += [f"| {c} | {a} | {n} |" for (c, a), n in sorted(rules.items(), key=lambda kv: -kv[1])] + [""]
            # Same routing decisions, other metrics at B = 300.
            for metric, keep in (("recall", None), ("file_recall_patch", "verified")):
                dm = analysis.load(budget, metric)
                assert dm.ids == d.ids, "metric tables must cover the same issues"
                mask = held if keep is None else held & (dm.feats["is_pro"] == 0)
                sub = {k: c[mask[held]] for k, c in choice.items()}
                em = evaluate(dm, mask, sub)
                out[f"{metric}@{budget}"] = em
                report += fmt_main(em, f"{metric}@{budget}" + (" (SWE-bench Verified only)" if keep else ""))
        if budget == 300 and (EXTRA / "phase_a.jsonl").exists():
            report += verified_extra(d, held, choice, out)
    (RESULTS / "heldout.md").write_text("\n".join(report))
    (RESULTS / "heldout.json").write_text(json.dumps(out, indent=1, default=float))
    print("\n".join(report))
    print(f"-> {RESULTS / 'heldout.md'}, {RESULTS / 'heldout.json'}")


if __name__ == "__main__":
    main()
