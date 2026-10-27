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

## Tier 2: a classifier router (stretch goal)

Pursued because Tier 1 was evaluated before November 1, as the proposal
requires. Protocol fixed on 2026-10-27, after the Tier 1 results above and
before any Tier 2 result on the real data (`acl.tier2`):

- Same 16 post-seed features as Tier 1. Label per training issue: its best
  action on nDCG at the budget, ties broken toward the cheaper action.
- Grid, simplest first: logistic regression on standardised features with
  C in {0.01, 0.1, 1, 10}, then gradient-boosted trees of depth 1 to 3 with 50
  or 100 trees (learning rate 0.1, random_state 16).
- Tuning on dev only. For each held-out fold: fit every setting on the
  issues outside the fold and outside dev, keep the one with the best mean
  nDCG on dev (ties to the simpler one), refit it on every issue outside the
  fold and route the fold. Unit tests check that a planted signal is learned
  and that corrupting a fold's own scores leaves its routing unchanged.
- Reported on the same 444 held-out issues and budgets as Tier 1, next to
  router and router_1, with the paired gain over router_1.

Tier 2 is designed after seeing Tier 1's result, which the write-up must say.
Its settings and selection rule above do not change after its own result.

### Tier 2 results

Run locally on 2026-10-27 from the ada Phase A files (job 28702), about 90 s.
Full output: `data/results/tier2.md`. With the same files the laptop
reproduces Stage 5's `heldout.md` byte for byte, so Tier 2 and Tier 1 are
scored on identical data.

| nDCG@300, 444 held-out issues | mean | gain vs fixed | 95% CI | GapClosed |
|---|---|---|---|---|
| fixed (fusion) | .1872 | | | |
| router (Tier 1) | .1908 | +.0035 | [-.0041, +.0086] | +.047 |
| router_1 (Tier 1, first rule) | .1927 | +.0055 | [-.0005, +.0077] | +.073 |
| tier2 (classifier) | .1798 | -.0074 | [-.0255, +.0006] | -.099 |

- Tier 2 is worse than the fixed action at every budget (nDCG@100 -.0130,
  @500 -.0103 with an interval below zero) and reliably worse than router_1
  at B = 300 (-.0129 [-.0261, -.0058]).
- It moves 78% of issues away from fusion and helps 32% of issues while
  hurting 39%. It picks the oracle's action on 24% of issues; always picking
  fusion matches it on 21%.
- On SWE-bench file recall it loses .054 [-.103, -.043] against fusion, much
  more than Tier 1 (-.013), because it routes more issues to graph expansion.
- Dev picked gradient-boosted trees in every fold at B = 300, mostly depth 3.
  With 56 dev issues, the most flexible setting wins on dev and then
  overfits the held-out repositories.

Reading: a learned classifier over the same 16 features does not find more
signal than the rule list. It finds less, because it spends its capacity on
patterns that do not transfer between repositories. This supports the
Stage 5 verdict: the features carry little transferable information about
which action wins, and the one rule that does transfer is best used alone.

### Data note

The ada Phase A files and an independent laptop run agree exactly on 5,982
of 6,000 BM25 and graph-expansion rows. The other 18 are two openlibrary
issues (dev split), whose BM25 scores differ slightly between the two
machines (nDCG@300 by up to .022), presumably because those two snapshots were
built a little differently. Feature values otherwise differ only by float32
rounding. No held-out issue is affected. All reported numbers use the ada
files.
