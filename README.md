# Adaptive code localization

When does graph expansion help? A measurement-driven study of retrieval actions
(BM25 seed, dense, fusion, edge-typed graph expansion) for repository-level
issue localization on the Python subset of SWE-Explore, followed by an
interpretable per-issue router.

## Setup

```bash
uv venv .venv
uv pip install --python .venv/bin/python -e '.[dev]'
.venv/bin/python -m pytest
```

Code targets Python 3.10+ (the cluster's compute nodes run 3.10).

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
