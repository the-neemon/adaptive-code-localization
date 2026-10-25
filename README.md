# Adaptive code localization

When does graph expansion help? A measurement-driven study of retrieval actions
(BM25 seed, dense, fusion, edge-typed graph expansion) for repository-level
issue localization on the Python subset of SWE-Explore, followed by an
interpretable per-issue router.

## Setup

Any Python 3.10+ environment with the dependencies works (locally we use the
`mtsbench` conda env; on ada, `totto`):

```bash
pip install -e '.[dev]'          # add '.[dense]' on the GPU node for acl.dense
python -m pytest
```

## Data

Everything under `data/` is gitignored. The SWE-Explore data is CC BY-NC-ND, so
it is never committed; fetch it yourself:

```bash
mkdir -p data/raw && cd data/raw
HF=https://huggingface.co
curl -LO $HF/datasets/SWE-Explore-Bench/SWE-Explore-Bench/resolve/main/bench.final.public.jsonl
curl -L -o swebench_verified.parquet $HF/api/datasets/princeton-nlp/SWE-bench_Verified/parquet/default/test/0.parquet
curl -L -o swebench_pro.parquet     $HF/api/datasets/ScaleAI/SWE-bench_Pro/parquet/default/test/0.parquet
curl -L -o swebench_pro_v1.parquet  $HF/api/datasets/ScaleAI/SWE-bench_Pro/parquet/v1/test/0.parquet
```

## Pipeline (Stage 1)

```bash
python -m acl.data       # join SWE-Explore with SWE-bench -> data/tasks.jsonl (500 Python tasks)
python -m acl.snapshot   # fetch the pre-fix commits (.py contents only) -> data/mirrors/
python -m acl.prepare --workers 4   # chunks, BM25 seed, graph -> data/prepared/<instance_id>/
python -m acl.inspect_task <instance_id> ...   # hand-check report
```

| Module | What it does |
|---|---|
| `acl.data` | Python filter and join with SWE-bench Verified / Pro for issue text and base commit |
| `acl.snapshot` | Partial, depth-1 fetch of only the needed commits and `.py` blobs; reads a snapshot from git objects |
| `acl.chunking` | Function / class / module chunks with line spans, ids matching LocAgent node names |
| `acl.bm25` | Code-aware tokenizer, BM25, filling a ranked list to a line budget |
| `acl.graph` | LocAgent's graph builder (Apache-2.0), adapted; maps edges onto chunks |
| `acl.prepare` | Runs the above per task, resumable |

See [docs/stage1.md](docs/stage1.md) for dataset counts, design decisions and the
hand-check.

## Pipeline (Stage 2)

```bash
python -m acl.linecounts   # GT file lengths at each pre-fix commit -> data/line_counts.json
python -m acl.splits       # dev repos + leave-one-repo-out folds -> configs/splits.json
# on an ada GPU node (see scripts/ada_dense.sbatch); copy data/dense/ back afterwards
python -m acl.dense        # CodeRankEmbed scores per chunk -> data/dense/coderankembed/
python -m acl.phase_a      # stop / dense / fusion on every task, B = 100/300/500 -> data/results/
```

| Module | What it does |
|---|---|
| `acl.evaluate` | Metrics at a line budget: ours (clipped GT, paper nDCG) and SWE-Explore's official ones |
| `acl.swe_explore_eval` | SWE-Explore's `eval.py`, vendored verbatim (MIT) as the reference scorer |
| `acl.linecounts` | Line counts of every ground-truth file, fetching the non-.py blobs |
| `acl.dense` | Frozen CodeRankEmbed embeddings, deduplicated across snapshots |
| `acl.actions` | Rankings for each action (stop, dense, RRF fusion) |
| `acl.splits` | Repository-disjoint splits |
| `acl.phase_a` | Runs actions on every task and scores them |

See [docs/stage2.md](docs/stage2.md) and the draft [related work](docs/related_work.md).

## Phase A (Stage 3)

```bash
python -m acl.tune --what expand   # lambda and edge direction per edge group, dev only -> configs/tuned.json
python -m acl.features             # post-seed router features -> data/results/features.jsonl
sbatch scripts/ada_phase_a.sbatch  # on ada: tune RRF k on dev, all actions x all tasks, features
```

| Module | What it does |
|---|---|
| `acl.actions` | adds `expand_calls`, `expand_imports`, `expand_inherit` (proposal eq. 1) |
| `acl.tune` | grid search on dev for the expansion and fusion hyperparameters |
| `acl.features` | cheap issue and seed signals for the router (no patch or ground truth) |

See [docs/stage3.md](docs/stage3.md).

## Phase A analysis and Tier 1 rules (Stage 4)

```bash
python -m acl.analysis           # gains vs seed with repo-bootstrap CIs, breakdowns, signature table
python -m acl.router --per-fold  # Tier 1 rule procedure, fit on all issues and on each fold's training set
```

| Module | What it does |
|---|---|
| `acl.analysis` | Phase A summary (`data/results/phase_a_summary.md`) and signature table (`signature_table.csv`) |
| `acl.router` | the fixed rule procedure: greedy ordered rule list over feature bins and bin pairs |

See [docs/stage4.md](docs/stage4.md).

## Held-out evaluation (Stage 5)

```bash
python -m acl.data --verified-extra   # the 90 SWE-bench Verified issues outside the study set
sbatch scripts/ada_stage5.sbatch      # on ada: those 90 end to end, then acl.heldout
python -m acl.heldout                 # router per fold on held-out repos -> data/results/heldout.md
```

| Module | What it does |
|---|---|
| `acl.heldout` | fits the rule procedure per fold, routes the held-out repository, GapClosed with repo-bootstrap CIs, cost, SWE-bench Verified file recall |

`acl.snapshot`, `acl.prepare`, `acl.dense`, `acl.phase_a` and `acl.features` take
`--tasks PATH` to run on another task file. See [docs/stage5.md](docs/stage5.md).
