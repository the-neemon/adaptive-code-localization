# Stage 2: evaluation harness, dense retriever, fusion, splits

Status as of 2026-10-15: complete. Checkpoint from the proposal ("metrics match
golden examples"): met, see "Harness checks" below. Stage 1 was re-verified
first.

## Stage 1 verification

- `acl.data` reproduces the task list exactly: 500 Python tasks, 410 Verified
  and 90 Pro, same per-repository counts as docs/stage1.md.
- `acl.prepare` was re-run from scratch on all 500 tasks (on ada; 104 of them
  also on the laptop). The 104 tasks built on both machines are
  content-identical (chunks, spans, BM25 scores, edges, seed regions), across
  different Python (3.10 vs 3.11), polars and scipy versions. Stage 1 is
  deterministic.
- The `stop` action in the new runner reproduces every stored seed region list
  at B = 100, 300 and 500 for all 500 tasks (the runner asserts this).
- The hand-check on the 10 tasks in handcheck_ids.txt gives the same seed line
  recall and the same ranks for the target code as the stage1.md table, for
  all 10. Mean seed line recall@300 under the official scorer is 11.55%,
  matching stage1.md's 11.6%.

## Evaluation harness (`acl.evaluate`)

Scores a ranked region list for one task at a line budget B. Predictions are
first cut to B lines; the region that crosses B is cut at B (the official
recall@budget convention, and what `fill_budget` produces).

**Two families of metrics, side by side.**

*Official*: SWE-Explore's released `eval.py`, vendored verbatim as
`acl.swe_explore_eval` (MIT, commit 5602f03) and called on the same
predictions. Keys are prefixed `off_`. These are comparable with the
SWE-Explore paper.

*Ours (primary)*: the official scorer has two properties that make it a poor
gain signal for this study, so the primary metrics differ:

1. **Clipped ground truth.** Core and optional ranges are clipped to the file's
   length at the pre-fix commit (agents ran `sed -n 1,240p` on 146-line
   files), and regions on files absent from the commit are dropped. The
   official scorer keeps these phantom lines in the recall denominator.
   Non-.py core lines are kept: they are part of the task, and missing them is
   a real limit of a .py index.
2. **nDCG@B with a ground-truth ideal.** The paper (Appendix "app:idcg")
   defines the ideal DCG by greedily placing ground-truth core regions, each
   cut to the remaining budget, in order of marginal gain (ties: shorter
   region, then path). The released code instead reorders the regions the
   system itself returned, so a list whose only core hit is at rank 1 scores
   1.0. At B = 300, 75 of the 500 BM25 seeds get official nDCG = 1.0 with a
   mean line recall of only 14%. We implement the paper's definition (decided
   2026-10-14). The ideal is a constant per task and budget, so a gain in nDCG
   is a gain in DCG, and finding more core code is rewarded. Line gains keep
   the official weights (1.5 for lines in `main_files`, else 1.0), and a line
   counts once.

| Key | Meaning |
|---|---|
| `recall` | returned core lines / core lines (clipped) |
| `ndcg` | DCG over the cut list / paper's ideal DCG (primary gain metric) |
| `perfect_recall` | every core region is touched by a returned line (the proposal's "all core regions covered") |
| `perfect_recall_lines` | every non-blank .py core line is returned (strict variant) |
| `first_hit_rank` | 1-based rank of the first region touching core; NaN if none |
| `noise_region` | share of returned regions touching neither core nor optional lines |
| `precision`, `context_efficiency` | returned lines that are core / core or optional |
| `hit_file` | share of core files with at least one returned line |
| `file_recall_patch` | share of files edited by the developer patch that are returned (the SWE-bench file-level metric; patch files are stored per task in `tasks.jsonl` as `patch_files`, new files excluded) |
| `recall_cap` | min(1, B / core lines): the best any ranker could reach |
| `off_*` | the official scorer's recall@B, nDCG@B, precision, efficiency, hit/noise file and region rates, first useful hit |

**Why two perfect-recall variants.** Requiring every core line is unreachable
by construction: chunks trim the blank lines between definitions, so of the 67
tasks whose core fits in 300 lines only 1 has every core line inside a chunk
(1,036 blank and 32 non-.py lines are missing). That metric was 0 for every
action. The region-level version is reachable (13 to 20% of tasks) and
differs across actions; the line-level one skips blank and non-.py lines
(decided 2026-10-15).

**Line counts** (`acl.linecounts`). Resolving `end = -1` and negative starts
(27 regions), and clipping, need each ground-truth file's length at its commit.
516 ground-truth regions point at non-.py files, whose blobs Stage 1 never
fetched, so this step fetches just those (about 180 blobs). 16 ground-truth
paths do not exist at their commit (build artefacts, vendored submodules).
Also writes the blank lines of ground-truth .py files.

### Harness checks (the checkpoint)

`tests/test_evaluate.py`:
- Golden examples with hand-derived values for every primary metric,
  including budget cutting, the paper vs official nDCG on the same list,
  phantom lines, `-1`/negative ranges, missing files, blank lines and no-hit
  cases.
- A randomised check (300 instances x 3 budgets) that our recall, precision,
  context efficiency and hit-file equal the official scorer's whenever the
  definitions coincide (no phantom lines, no main files).
- On real data, ours and the official recall differ only through clipping:
  0.1204 vs 0.1155 for the seed at B = 300.

## Dense retriever (`acl.dense`)

- Model: `nomic-ai/CodeRankEmbed` (137M bi-encoder, CoRNStack; Suresh et al.,
  2025), frozen, fp16 on one 2080 Ti. The only candidate (decided
  2026-10-14), so there was no model selection on dev.
- Document = path + chunk source, max 512 tokens. Query = the required prefix
  "Represent this query for searching relevant code: " + the issue text, max
  1024 tokens. Scores are cosine similarities.
- Documents are embedded once per distinct text per repository: 207 django
  snapshots (about 30k chunks each) collapse to 88.7k distinct chunks, all 500
  snapshots to about 476k. Embedding everything took about 25 GPU-minutes;
  chunking the snapshots took longer than embedding them.
- Scores are stored per task in chunk order with a hash of the chunk ids, and
  `load_scores` refuses scores that are out of sync with `chunks.parquet`.
- **transformers 5 compatibility.** The model's remote code calls
  `get_extended_attention_mask`, which transformers 5 removed. ada's env needs
  transformers 5 (its `huggingface_hub` is 1.x), so `acl.dense` restores the
  transformers 4 behaviour. Checked: embeddings on ada (transformers 5.17 +
  shim) match the laptop's (transformers 4.57, no shim) with cosine 1.0 in
  fp32 and at least 0.9997 in fp16.

## Fusion (`acl.actions`)

Reciprocal-rank fusion of the full BM25 and dense rankings,
score(c) = sum of 1 / (k + rank), with k = 60 (Cormack et al., 2009) for now;
k is to be tuned on dev only. Every action ranks all chunks and
`fill_budget` cuts to B, so no action can win by returning more code. The
`expand(e)` actions plug into the same `ACTIONS` registry in Stage 3.

## Splits (`acl.splits`, `configs/splits.json`)

Decided 2026-10-14: a fixed dev group plus leave-one-repository-out.

| Split | Repositories | Tasks |
|---|---|---|
| dev (tuning only) | xarray, pytest, openlibrary | 56 |
| fold django | django | 207 |
| fold sympy | sympy | 66 |
| fold ansible | ansible | 40 |
| fold scikit-learn | scikit-learn | 31 |
| fold qutebrowser | qutebrowser | 30 |
| fold matplotlib | matplotlib | 29 |
| fold small | pylint, astropy, requests, seaborn, flask | 29 |
| fold sphinx | sphinx | 12 |

The dev set mixes Verified and Pro (openlibrary is Pro).

## First results (all 500 tasks)

`python -m acl.phase_a` writes `data/results/phase_a.jsonl` (one row per task,
action and budget). Means:

| Action | B | recall | nDCG | perfect (regions) | noise (regions) | patch-file recall | off recall | off nDCG |
|---|---|---|---|---|---|---|---|---|
| stop | 100 | .063 | .264 | .092 | .441 | .522 | .059 | .584 |
| dense | 100 | .060 | .213 | .114 | .424 | .548 | .056 | .571 |
| fusion | 100 | .068 | .268 | .108 | .369 | .558 | .064 | .607 |
| stop | 300 | .120 | .172 | .134 | .529 | .630 | .116 | .594 |
| dense | 300 | .128 | .152 | .170 | .524 | .671 | .123 | .578 |
| fusion | 300 | .137 | .185 | .162 | .470 | .707 | .131 | .619 |
| stop | 500 | .154 | .144 | .156 | .582 | .688 | .149 | .595 |
| dense | 500 | .164 | .128 | .192 | .579 | .729 | .158 | .578 |
| fusion | 500 | .183 | .157 | .196 | .524 | .765 | .177 | .619 |

nDCG falls as B grows because the ideal grows faster than what chunk lists
can reach; compare actions within a budget, not across.

**Per-issue gains over the seed at B = 300 (nDCG):**

| Action | mean gain | helps | hurts | no change |
|---|---|---|---|---|
| dense | -0.020 | 45% | 48% | 6% |
| fusion | +0.013 | 52% | 35% | 12% |

| nDCG@300 | stop | dense | fusion | per-issue oracle |
|---|---|---|---|---|
| all 500 | .172 | .152 | .185 | .231 |
| dev 56 | .119 | .108 | .161 | .190 |

This already supports the study's premise: dense is worse than BM25 on
average but better on 45% of issues, and the per-issue oracle over only these
three actions is 25% above the best fixed action (fusion). Pro issues score
higher than Verified ones (seed nDCG .217 vs .162), consistent with their
templated, more specific text.

Our nDCG tracks coverage better than the official one: across seeds at
B = 300, corr(nDCG, recall) is 0.55 for ours and 0.42 for the official one.
The mean `recall_cap` at B = 300 is 0.395 (stage1.md: 0.390 before clipping).

## Compute notes

- ada: the code, mirrors and outputs live in `~/ire/project/acl`; the env is
  `totto` (Python 3.11, torch 2.6 + cu124), to which `sentence-transformers`,
  `transformers` 5.17 and `einops` were added without changing any existing
  package. Batch scripts: `scripts/ada_dense.sbatch`,
  `scripts/ada_prepare.sbatch`. Account `research` may not use the `u22-cpu`
  partition, so CPU work runs on `u22` without a GPU; the QoS caps a user at
  40 CPUs, 4 GPUs and about 125 GB.
- gnode069 and gnode077 failed with `CUDA unknown error` on 2026-10-15; jobs
  exclude them (`-x`). Request 2080 Ti nodes with `--constraint=2080ti`;
  `u22` otherwise also hands out 1080 Ti GPUs (slow in fp16).
- Prepared outputs (551 MB) stay on ada; the link from ada to the laptop ran
  at about 110 KB/s. `acl.phase_a` runs on ada and only results come back.
- Locally the env is `mtsbench` (polars, pytest and einops added).

## Open items for Stage 3

- `expand(e)` actions: edge direction (outgoing only, or callers and
  subclasses too) and the decay lambda (tune on dev).
- Tune the RRF k on dev.
- Whether to strip pasted environment dumps from the query (stage1.md); the
  dense query is cut at 1024 tokens, which already drops the tail of long
  issues.
- Post-seed features for the router, then launch the Phase A batch by Oct 18.
