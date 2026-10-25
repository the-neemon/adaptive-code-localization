"""Build the Python task list by joining SWE-Explore with its SWE-bench sources.

SWE-Explore records carry the ground-truth regions but no issue text, repo
name or commit. Those come from SWE-bench Verified and SWE-bench Pro, joined
on instance_id. (SWE-bench Multilingual has no Python tasks.) 21 Pro tasks were
dropped from Pro's current public split and survive only in its `v1` config,
so that file is read as a fallback.

The study set is fixed at these 500 tasks. For the secondary benchmark
(SWE-bench Verified, scored at file level against the developer patch, as the
proposal says), `--verified-extra` writes the 90 Verified issues outside it:
49 that SWE-Explore does not include and 41 that the Python filter above drops
because agents also edited a non-.py file (mostly pyproject.toml), although
the developer patch is Python. They have no SWE-Explore ground truth, only
`patch_files`.

Usage: python -m acl.data [--verified-extra]
  (reads data/raw/, writes data/tasks.jsonl or data/tasks_verified_extra.jsonl)
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import polars as pl

RAW = Path("data/raw")
# Later files fill in ids missing from earlier ones.
SOURCES = {"verified": ["swebench_verified.parquet"],
           "pro": ["swebench_pro.parquet", "swebench_pro_v1.parquet"]}


def is_python(gt: dict) -> bool:
    """A task is Python if every file the fix touched is .py.

    52 records have no modified_core_files (the successful trajectories did not
    agree on an edit); for those we fall back to the core files that were read.
    """
    files = gt["modified_core_files"] or gt["read_core_files"]
    return bool(files) and all(f.endswith(".py") for f in files)


def optional_regions(gt: dict) -> list[dict]:
    """Union of the per-model optional regions, deduplicated."""
    seen, out = set(), []
    for regions in gt["read_optional_regions_map"].values():
        for r in regions:
            key = (r["path"], r["start"], r["end"])
            if key not in seen:
                seen.add(key)
                out.append(r)
    return out


_DIFF_OLD = re.compile(r"^--- (?:a/(.+)|/dev/null)$", re.M)


def patch_files(patch: str) -> list[str]:
    """Files the developer patch changes that already exist at the pre-fix commit.

    Files the patch creates (`--- /dev/null`) are left out: they are not in the
    snapshot, so no retriever can return them.
    """
    return sorted({m.group(1) for m in _DIFF_OLD.finditer(patch) if m.group(1)})


def build_tasks(bench_path: Path, raw: Path = RAW) -> list[dict]:
    meta = {}
    for fname in (f for files in SOURCES.values() for f in files):
        df = pl.read_parquet(raw / fname, columns=["instance_id", "repo", "base_commit", "problem_statement", "patch"])
        for row in df.iter_rows(named=True):
            # SWE-bench Pro ids carry an "instance_" prefix that SWE-Explore drops.
            meta.setdefault(row["instance_id"].removeprefix("instance_"), row)

    tasks = []
    for line in bench_path.open():
        rec = json.loads(line)
        gt = rec["ground_truth"]
        if rec["dataset"] not in SOURCES or not is_python(gt):
            continue
        src = meta[rec["instance_id"]]  # KeyError here means the join broke
        tasks.append({
            "instance_id": rec["instance_id"],
            "source": rec["dataset"],
            "repo": src["repo"],
            "base_commit": src["base_commit"],
            "problem_statement": src["problem_statement"],
            "core_files": gt["read_core_files"],
            "core_regions": gt["read_core_regions"],
            "optional_regions": optional_regions(gt),
            "modified_core_files": gt["modified_core_files"],
            "main_files": gt["main_files"],
            "patch_files": patch_files(src["patch"]),
        })
    return tasks


def build_verified_extra(bench_path: Path, study: list[dict], raw: Path = RAW) -> list[dict]:
    in_study = {t["instance_id"] for t in study}
    in_bench = {json.loads(line)["instance_id"] for line in bench_path.open()}
    df = pl.read_parquet(raw / SOURCES["verified"][0],
                         columns=["instance_id", "repo", "base_commit", "problem_statement", "patch"])
    return [{
        "instance_id": row["instance_id"], "source": "verified", "repo": row["repo"],
        "base_commit": row["base_commit"], "problem_statement": row["problem_statement"],
        "extra_reason": "python_filter" if row["instance_id"] in in_bench else "not_in_swe_explore",
        # No SWE-Explore ground truth: only the file-level patch metric applies.
        "core_files": [], "core_regions": [], "optional_regions": [],
        "modified_core_files": [], "main_files": [],
        "patch_files": patch_files(row["patch"]),
    } for row in df.iter_rows(named=True) if row["instance_id"] not in in_study]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verified-extra", action="store_true")
    args = ap.parse_args()
    tasks = build_tasks(RAW / "bench.final.public.jsonl")
    out = Path("data/tasks.jsonl")
    if args.verified_extra:
        tasks = build_verified_extra(RAW / "bench.final.public.jsonl", tasks)
        out = Path("data/tasks_verified_extra.jsonl")
        print("by reason:", dict(Counter(t["extra_reason"] for t in tasks)))
    with out.open("w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")
    print(f"{len(tasks)} tasks -> {out}")
    print("by source:", dict(Counter(t["source"] for t in tasks)))
    for repo, n in Counter(t["repo"] for t in tasks).most_common():
        print(f"  {n:4d}  {repo}")


if __name__ == "__main__":
    main()
