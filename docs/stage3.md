# Stage 3: graph expansion, dev tuning, router features, Phase A launch

Status as of 2026-10-27: done. The three `expand(e)` actions, their tuning on
dev, the post-seed features and a local Phase A run of every BM25-only action
were finished on 2026-10-19. The full Phase A batch with all six actions
(`scripts/ada_phase_a.sbatch`) ran on ada on 2026-10-19 (job 28702), one day
after the proposal's launch date, and tuned the fusion constant to k = 100 on
dev. Its analysis is in docs/stage4.md.

## Graph expansion (`acl.actions.expand`)

As in the proposal (eq. 1). Parents are the top k = 10 seed chunks. Every
depth-1 neighbour n along the edge group gets
s(n) = max(own min-max BM25, lambda * best parent score); all other chunks keep
their BM25 score; the merged list is sorted (ties keep the seed order) and cut
to the budget by `fill_budget`. With no neighbours the action equals `stop`
exactly (tested).

| Action | Stored edge types |
|---|---|
| `expand_calls` | invokes |
| `expand_imports` | imports, contains (the proposal groups imports with containment) |
| `expand_inherit` | inherits |

**Edge direction** was the open question from Stage 1. It is tuned on dev per
group as a hyperparameter, alongside lambda: "out" follows edges from a parent
(callees, imported names, members, bases); "both" also follows them backwards
(callers, importers, containers, subclasses).

## Tuning on dev (`acl.tune`, `configs/tuned.json`)

Dev = xarray, pytest, openlibrary (56 tasks); objective mean nDCG@300 (ours);
grid lambda in {0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99} x direction in {out,
both}. Exact ties go to "out" and the larger lambda: a tie means extra
expansion costs nothing on dev, and the smallest lambda would make
`expand_inherit` identical to `stop`, so Phase A could not measure it.

| Action | Chosen | dev nDCG@300 | vs stop (0.1193) |
|---|---|---|---|
| expand_calls | both, lambda 0.99 | 0.1480 | +0.0287 |
| expand_imports | both, lambda 0.99 | 0.1336 | +0.0143 |
| expand_inherit | out, lambda 0.7 | 0.1193 | 0.0000 |

- Calls and imports improve monotonically with lambda up to the top of the
  grid, and "both" beats "out" at every lambda >= 0.7. At 0.99 a neighbour
  sits directly behind its parent, so the dev result reads: put the callers and
  callees of a strong seed hit right after it, ahead of weaker BM25 hits.
- Inheritance edges are rare (mean 0.28 at the top seed chunk). At every
  lambda <= 0.7 the dev score equals `stop`, and larger lambdas hurt. Even the
  chosen setting returns the same regions as `stop` on 55 of 56 dev tasks.
- The fusion RRF k (grid 5, 10, 20, 30, 60, 100) is tuned on ada, where the
  dense scores are (first step of the batch script).

## First Phase A numbers for the graph actions (local run)

`python -m acl.phase_a --actions stop expand_calls expand_imports expand_inherit`
over all 500 tasks took about 3 minutes. Per issue at B = 300, nDCG, against
`stop`, on the 444 tasks outside dev (lambda was tuned on dev):

| Action | mean nDCG@300 | mean gain | helps | hurts | no change |
|---|---|---|---|---|---|
| stop | 0.1786 | | | | |
| expand_calls | 0.1812 | +0.0026 | 40% | 39% | 21% |
| expand_imports | 0.1743 | -0.0043 | 37% | 44% | 19% |
| expand_inherit | 0.1787 | +0.0001 | 4% | 2% | 94% |
| per-issue oracle over these four | 0.2201 | | | | |

- This is the premise of the study in numbers. Averaged over issues,
  expansion barely moves nDCG, but it helps about 40% of issues and hurts
  about 40%. Choosing per issue (the oracle) would be 21% above the best fixed
  action.
- The dev gain does not carry over in full: +0.029 on dev against +0.003 on
  the folds for `expand_calls`, with folds ranging from -0.046
  (scikit-learn) to +0.032 (matplotlib). 56 dev tasks from three repositories
  are a small tuning set; the per-fold spread is a result to report, not a
  bug.
- Cost (`n_visited`, neighbours scored): `expand_calls` median 41, 90th
  percentile 193, max 1,923; `expand_imports` median 16, max 2,370. The
  large values are hubs, which the proposal expected to flood the budget; the
  top-hit degree is a router feature for this reason.

## Post-seed features (`acl.features`, `data/results/features.jsonl`)

The proposal's list, computed only from the issue text, the seed, the graph
and the dense scores; nothing reads the patch or the ground truth.

| Feature | Share or mean over 500 |
|---|---|
| has_traceback | 13.2% |
| names_py_path / names_repo_file (the path exists in the snapshot) | 36.2% / 17.8% |
| n_known_identifiers (code-like names in the issue that name a repo chunk) | 5.2 |
| has_env_dump (pip list, versions block, or 3+ "name: 1.2" lines) | 8.4% |
| issue_tokens | 195 |
| bm25_margin (top-1 vs runner-up), bm25_topk_margin | 0.096, 0.215 |
| seed_files (distinct files in the top 10) | 5.8 |
| top_degree (all, calls, imports, inherit), topk_degree_mean | 19.8 (15.8, 3.7, 0.28), 16.7 |
| bm25_dense_overlap (top-10 agreement) | computed on ada |
| source | Verified or Pro |

`has_env_dump` replaces the Stage 1 question of stripping pasted environment
dumps from the query: the query stays as written, and the router sees whether
a dump is there. A first version of the detector missed the scikit-learn
"System: python: 3.7.2" template (49 issues); the version-line rule fixes it
without matching `x = 1.5` lines in reproduction scripts or matplotlib's empty
"_No response_" template (tested).

## Launching Phase A on ada

```bash
# on ada, in ~/ire/project/acl after pulling this commit
sbatch scripts/ada_phase_a.sbatch    # tune k on dev, all 6 actions x 500 tasks, features
```

It overwrites Stage 2's `data/results/phase_a.jsonl` with a superset (the
stop and dense rows are unchanged; fusion changes only if the tuned k differs
from 60). Copy back `data/results/` and `configs/tuned.json`.
