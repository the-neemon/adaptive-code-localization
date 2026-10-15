"""Build the Python task list by joining SWE-Explore with its SWE-bench sources.

SWE-Explore records carry the ground-truth regions but no issue text, repo
name or commit. Those come from SWE-bench Verified and SWE-bench Pro, joined
on instance_id. (SWE-bench Multilingual has no Python tasks.) 21 Pro tasks were
dropped from Pro's current public split and survive only in its `v1` config,
so that file is read as a fallback.

Usage: python -m acl.data  (reads data/raw/, writes data/tasks.jsonl)
"""
from __future__ import annotations

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


def main() -> None:
    tasks = build_tasks(RAW / "bench.final.public.jsonl")
    out = Path("data/tasks.jsonl")
    with out.open("w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")
    print(f"{len(tasks)} Python tasks -> {out}")
    print("by source:", dict(Counter(t["source"] for t in tasks)))
    for repo, n in Counter(t["repo"] for t in tasks).most_common():
        print(f"  {n:4d}  {repo}")


if __name__ == "__main__":
    main()
