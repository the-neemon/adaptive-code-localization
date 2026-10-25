# Stage 5: the Tier 1 router on held-out repositories

Status as of 2026-10-25: code and protocol done and tested, run pending on ada
(`scripts/ada_stage5.sbatch`). This file fixes the protocol before any
held-out result exists. Results go below when the job returns.

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

Pending.
