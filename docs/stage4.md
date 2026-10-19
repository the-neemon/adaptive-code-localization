# Stage 4: Phase A analysis, signature table, Tier 1 rules

Status as of 2026-10-19: draft findings done (the proposal's checkpoint for
this stage). Started a week early, during exam week, because Phase A finished
on ada on 2026-10-19 (job 28702: all 6 actions x 500 tasks x 3 budgets, RRF
k = 100 tuned on dev) and the analysis needs nothing else. No held-out
repository has been scored by the router yet; that is Stage 5.

All numbers are nDCG@300 (ours, docs/stage2.md) unless stated, and gains are
the proposal's eq. 2: Delta(i, a) = Score(action a) - Score(stop). Intervals
are 95% repository-clustered bootstrap intervals (whole repositories are
resampled, since issues of one repository are not independent; django is 41%
of the tasks).

```bash
python -m acl.analysis           # data/results/phase_a_summary.md, signature_table.csv
python -m acl.router --per-fold  # data/results/router_rules.json
```

## Phase A: the marginal value of each action

All 500 issues:

| Action | nDCG@300 | gain vs stop | 95% CI | helps | hurts |
|---|---|---|---|---|---|
| stop (BM25 seed) | .1720 | | | | |
| expand_inherit | .1721 | +.0001 | [-.0001, +.0002] | 3% | 2% |
| expand_imports | .1698 | -.0022 | [-.0163, +.0074] | 38% | 42% |
| expand_calls | .1775 | +.0055 | [-.0086, +.0133] | 41% | 37% |
| dense | .1523 | -.0197 | [-.0470, -.0057] | 45% | 48% |
| fusion | **.1844** | +.0124 | [+.0021, +.0306] | 52% | 35% |
| per-issue oracle | .2582 | | | | |

- On average only fusion beats the seed, and only dense is reliably worse.
  None of the graph expansions moves the average.
- Per issue, every action except inherit helps and hurts large shares of
  issues. The oracle (best action per issue, in hindsight) is 40% above the
  best fixed action. The oracle's choice is spread evenly: stop 17%,
  expand_imports 20%, expand_calls 19%, dense 22%, fusion 21%. No action
  dominates, which is the premise of the router.
- Outside dev (444 issues, since lambda and k were tuned on dev) the picture is
  the same: fusion .1872 is the best fixed action, oracle .2618 (+40%).
- `expand_inherit` changes nDCG on only 5% of issues (inheritance edges are
  rare); it is effectively `stop`.
- By repository the gains differ in sign (e.g. expand_calls: matplotlib
  +.032, scikit-learn -.046; fusion: xarray +.112, pylint -.046). Pro issues
  gain more from every action than Verified ones (fusion +.028 vs +.009).
  Full tables: `data/results/phase_a_summary.md`.

## The proposal's hypotheses

| Proposal's expectation | Result | Verdict |
|---|---|---|
| A traceback makes expand(calls) help | +.009 [-.015, +.034] with a traceback (66 issues) vs +.005 without | not supported |
| An issue naming the target file needs no expansion (seed already right, expansion adds noise) | expansion helps slightly *more* there: calls +.011 [-.002, +.016], imports +.013 [-.001, +.029] | not supported; if anything the opposite |
| A hub top hit makes expansion flood the budget | top hit with degree > 10: calls -.014 [-.036, +.001]; degree <= 4: calls +.017 [+.009, +.031] | supported for calls |
| BM25/dense disagreement calls for fusion | fusion +.014 [-.003, +.048] when they disagree vs +.012 when they agree | not supported |
| A concentrated seed means stop | no difference across bm25_margin bins | not supported |

The graph degree of the top seed hit is the strongest single signal in the
whole table, and not only for expansion: with a hub top hit (degree > 10)
dense loses .062 [-.110, -.036] and fusion .017; with a poorly connected top
hit (<= 4) fusion gains .040 [+.022, +.080]. A plausible reading is that
BM25 lands on a central, well-connected definition when the issue names the
right code, and then any change hurts.

## Signature table (`data/results/signature_table.csv`)

Every post-seed feature is binned (booleans by value, numbers into thirds),
and for each bin and each pair of bins of two different features the table
gives each action's mean gain over stop, its interval and the shares helped
and hurt (4,070 rows).

| Cells with n >= 20 issues from >= 3 repositories | Cells | interval excludes 0 | expected by chance (5%) |
|---|---|---|---|
| single features | 205 | 46 | about 10 |
| pairs of features | 3,525 | 668 | about 176 |

So there is real structure, about 4 times what chance would produce, but any
single cell may be one of the chance ones. The table describes; only the
held-out test in Stage 5 can say whether rules built from it generalise.

## Tier 1 router: the rule procedure (`acl.router`)

Decided 2026-10-19 (Yash): rules come from a fixed procedure, fit per fold,
not hand-written after reading the table, so that Stage 5 tests rules that
never saw the held-out repository. Rules have at most 2 conditions, at most 5
rules.

1. Default action = the best fixed action on the training issues (ties go to
   the cheaper action: stop < expand(e) < dense < fusion).
2. Candidate conditions = every feature bin and every pair of bins, with bin
   edges (thirds) computed on the training issues.
3. Greedy ordered rule list: among the issues not yet covered, take the
   (condition, action) with the largest total gain over the default, among
   candidates with at least 20 issues from at least 3 repositories whose
   repository-clustered one-sided 95% lower bound of the mean gain is above
   zero. Repeat up to 5 times.
4. Routing: the first matching rule picks the action, else the default.

Settings (20, 3, 5, 95%) were fixed before any rule was fit. They live as
constants in `acl/router.py`.

### Rules fit on all 500 issues (descriptive)

Default: fusion.

| # | Condition | Action | Issues | Gain vs fusion |
|---|---|---|---|---|
| 1 | top hit has > 7 call edges AND BM25/dense top-10 overlap <= 0.2 | expand_calls | 64 | +.053 |
| 2 | issue names a .py path AND top-hit degree in (4, 10] | expand_imports | 46 | +.045 |
| 3 | seed spans <= 5 files AND top-hit degree > 10 | expand_inherit (in effect stop) | 62 | +.026 |
| 4 | Verified AND top hit has inheritance edges | expand_imports | 33 | +.021 |
| 5 | top-k margin in (.15, .25] AND BM25/dense overlap > 0.4 | dense | 22 | +.028 |

In-sample (on the same 500 issues it was fit on) the router scores .2011
against fusion's .1844, about 23% of the way to the oracle (.2582). That is
an optimistic upper estimate: rules were selected on these issues.

**Rule 1 is the finding.** On its 64 issues (10 repositories): stop .202,
expand_calls .213, fusion .161, dense .121. When the top hit is a call hub
and dense disagrees with BM25, dense is wrong, fusion inherits its mistakes,
and the right move is to keep the seed and follow its call edges. This
reverses the proposal's guess that disagreement calls for fusion.

### Stability across folds (training sets only)

Fitting the procedure on each fold's training set (the held-out issues are
not scored): rule 1 is found in all 8 folds (sympy's version uses > 6 call
edges). Every other rule appears in at most 4 of 8 folds, and their lower
bounds sit close to zero (some at +.0004). With thousands of (condition, action)
candidates to choose from, the bootstrap bound on the *selected* rule is
optimistic (winner's curse), so rules 2 to 5 are likely partly noise. Expect
the held-out gain to come mostly from rule 1. Full per-fold output:
`data/results/router_rules.json`.

## What Stage 5 will do

- Freeze the procedure (commit it), then for each of the 8 folds fit on the
  other folds plus dev and route the held-out repository's issues.
- Report the router against stop, the best fixed action (fusion) and the
  oracle on held-out issues: GapClosed (proposal eq. 3), repository-clustered
  intervals, paired per-issue comparisons, breakdowns by repository and by
  whether the issue names a target file, and cost (retrieval calls, graph
  nodes visited, time).
- Also at B = 100 and 500, and on recall and SWE-bench file-level recall.

## Open questions for the team

- Rules 2 to 5 are unstable. Keeping MAX_RULES = 5 is what was decided; the
  alternative (only rules that recur across most training folds) would have
  to be fixed before Stage 5 runs, not after seeing its result.
- `expand_inherit` is effectively `stop`. It stays in the action set (the
  router picks it only through ties), but the write-up can merge the two.
