# Stage 5: the Tier 1 router on held-out repositories

Status as of 2026-10-27: done. The protocol below was fixed on 2026-10-25,
before any held-out result existed. The run (`scripts/ada_stage5.sbatch`)
happened on ada on 2026-10-26, and its results are at the end.

## Protocol (`acl.heldout`)

- For each of the 8 folds, the frozen rule procedure (`acl.router.fit`,
  docs/stage4.md) is fit on every other issue (the other 7 folds plus dev) and
  routes the held-out repository's issues. Bin edges come from the training
  issues only. A unit test corrupts one fold's scores and checks that its own
  routing does not change.
- Systems on the 444 held-out issues: `stop` (the seed), `fixed` (the best
  fixed action on the fold's training issues, which is also the router's
  default), `router` (up to 5 rules, the main result), `router_1` (the same
  list cut after its first rule, the declared secondary analysis) and
  `oracle` (the best action per issue in hindsight).
- Headline: GapClosed (proposal eq. 3) on nDCG@300, computed on means over
  the held-out issues, with a 95% repository-clustered bootstrap interval;
  gain over `fixed` with its interval; shares of issues helped and hurt.
- Also: nDCG at B = 100 and 500 (the router is fit at that budget), line
  recall@300 and SWE-bench file recall@300 (Verified only) for the same
  routing decisions, breakdowns by fold and by whether the issue names a
  target file, cost (extra retrieval calls, graph nodes visited, time), and
  which rules each fold's router picked.
- No setting of the procedure changes after this point. If a result looks
  wrong, the fix is reported as a fix, with both numbers.

## Secondary benchmark: SWE-bench Verified at file level

The proposal names SWE-bench Verified, scored at file level against the
developer patch, as the secondary benchmark. 410 of its 500 issues are in the
study set. The other 90 (`acl.data --verified-extra`,
`data/tasks_verified_extra.jsonl`) are run through the same pipeline:

| Why outside the study set | Issues |
|---|---|
| not in SWE-Explore | 49 |
| dropped by the study's Python filter: agents also edited a non-.py file (mostly pyproject.toml), while the developer patch is Python | 41 |

By repository: sphinx 32, django 24, astropy 13, sympy 9, matplotlib 5,
xarray 4, requests, pytest and scikit-learn 1 each. They have no SWE-Explore
ground truth, so only file recall of the patch applies. The 41 are a flaw in
the Stage 1 filter, which used the files agents edited instead of the
developer patch. The study set stays as it is, since changing it would
invalidate Phase A, the tuning and the splits.

The router fit on all 500 study issues routes the 90 extra issues, which no
tuning or rule fitting saw. The report then pools the Verified issues
outside dev: 374 held-out study issues (each routed by its fold's router) plus
the 90 extra, 464 in total. The 36 dev Verified issues train every router and
are left out.

## Running it

```bash
# on ada, in ~/ire/project/acl
sbatch scripts/ada_stage5.sbatch
```

One GPU job: the extra issues go through `acl.data`, `acl.snapshot`,
`acl.prepare`, `acl.dense`, `acl.phase_a` and `acl.features` (each with
`--tasks data/tasks_verified_extra.jsonl`), then `acl.heldout` writes
`data/results/heldout.md` and `heldout.json`. The dense step reuses the
per-repository embedding caches, so it only embeds chunks that changed.

## Results

Run on ada on 2026-10-26 (job 40536, gnode061, RTX 2080 Ti, about 1.5 h, no
errors). Full output: `data/results/heldout.md`.

Checks before reading the numbers: the extra-issue task list is identical to
the local one; all 1,080 stop and expand result rows for the 90 extra issues
equal an independent local run exactly; held-out stop and fusion means equal
Stage 4's (.1786, .1872). On ada the project folder is not a git clone, so
the code was copied with scp (its output format matches commit a2660c0), and
`data/raw/` was copied over for `acl.data --verified-extra`. No router
setting was changed.

### Main result: nDCG@300, 444 held-out issues

| system | mean |
|---|---|
| stop (BM25 seed) | .1786 |
| fixed (fusion in every fold) | .1872 |
| router (up to 5 rules) | .1908 |
| router_1 (first rule only) | .1927 |
| oracle | .2618 |

| | gain vs fixed | 95% CI | GapClosed | 95% CI | helps / hurts |
|---|---|---|---|---|---|
| router | +.0035 | [-.0041, +.0086] | +.047 | [-.047, +.114] | 17% / 18% |
| router_1 | +.0055 | [-.0005, +.0077] | +.073 | [-.005, +.107] | 7% / 6% |

The router does not reliably beat the best fixed action. It closes about 5%
of the gap to the oracle and its interval includes zero. The 1-rule router
does slightly better (7%, interval just reaching zero), as Stage 4's
stability check predicted: rules 2 to 5 add noise on unseen repositories.

### Other budgets and metrics

| metric | router vs fixed | router_1 vs fixed |
|---|---|---|
| nDCG@100 | -.0078 [-.0245, +.0159] | -.0001 [-.0068, +.0094] |
| nDCG@500 | -.0031 [-.0191, +.0046] | +.0010 [-.0099, +.0046] |
| line recall@300 | -.0008 [-.0075, +.0034] | +.0023 [+.0002, +.0064] |
| SWE-bench file recall@300 (374 held-out Verified) | -.0125 [-.0360, -.0036] | -.0021 [-.0053, +.0048] |
| SWE-bench file recall@300 (90 extra Verified) | -.0422 [-.1081, +.0417] | -.0278 [-.0932, +.0179] |

- No gain at B = 100 or 500. The 1-rule router's line-recall gain at 300 is
  the only interval above zero, and only just.
- On SWE-bench file recall the full router is reliably worse than fusion. The
  rules route issues to graph expansions, which cover fewer distinct files
  than fusion (fixed expand_calls .639 vs fusion .708 on the 464 Verified
  issues outside dev). The router optimises nDCG, which rewards finding more
  core lines in fewer files, so it trades file coverage for line ranking.
- On the 90 extra Verified issues, which no tuning or rule fitting saw, the
  same pattern holds with wide intervals.

### By fold, target file and cost

- GapClosed by fold ranges from -.16 (matplotlib) to +.16 (ansible); django,
  ansible and the small fold gain, matplotlib and qutebrowser lose.
- When the issue names a repository file, both routers gain more (router
  GapClosed +.13, router_1 .2123 vs fixed .1943) than when it does not (+.02).
- Cost per issue at B = 300: fusion always makes one extra (dense) retrieval
  call; the router makes 0.63 on average and visits 31 graph nodes. All
  systems take milliseconds once indexes exist.

### Rules

The stable rule (top hit with many call edges and BM25/dense top-10 overlap
<= 0.2, route to expand_calls) is chosen in all 8 folds. `heldout.md` lists
it as two rows because each fold computes its own bin edges: "> 7" call edges
in 7 folds and "> 6" in sympy's. Every other rule appears in at most 4 folds.

### Verdict

The proposal's target claim, that the router beats the best fixed action on
held-out repositories, is not supported. What stands is the proposal's
fallback: per-issue gains are large (the oracle is 40% above fusion) but the
cheap post-seed features predict only a small part of them, the strongest
being the graph degree of the top seed hit. One rule generalises and closes
about 7% of the gap. That is a conditional, partly negative result, which the
proposal anticipated.
