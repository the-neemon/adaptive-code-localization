"""Human-readable report of prepared outputs, for hand-checking Stage 1.

For each task: the issue head, core regions, coverage, the top-10 BM25 chunks
(marked * when they overlap a core line) and the graph neighbours of the top
hit. `seed@300 line recall` is a plain line-overlap sanity number, not the
evaluation harness.

Usage: python -m acl.inspect_task ID [ID ...] > report.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl

from acl.prepare import OUT


def core_lines(task: dict) -> set[tuple[str, int]]:
    return {(r["path"], ln) for r in task["core_regions"] for ln in range(r["start"], r["end"] + 1)}


def report(task: dict, out: Path = OUT) -> str:
    d = out / task["instance_id"]
    seed = json.loads((d / "seed.json").read_text())
    chunks = pl.read_parquet(d / "chunks.parquet")
    edges = pl.read_parquet(d / "edges.parquet")
    core = core_lines(task)
    st = seed["stats"]

    def hits(row) -> bool:
        return any((row["path"], ln) in core
                   for s, e in zip(row["starts"], row["ends"]) for ln in range(s, e + 1))

    seed_lines = {(p, ln) for p, s, e in seed["regions"]["300"] for ln in range(s, e + 1)}
    issue = " ".join(task["problem_statement"].split())[:500]
    lines = [
        f"## {task['instance_id']}  ({task['repo']} @ {task['base_commit'][:10]})",
        f"**Issue:** {issue}",
        "",
        "**Core regions:** " + ", ".join(f"{r['path']}:{r['start']}-{r['end']}" for r in task["core_regions"]),
        f"**Modified:** {', '.join(task['modified_core_files'])}",
        f"**Coverage:** {st['core_indexed']}/{st['core_lines']} core lines indexed "
        f"({st['core_resolvable']} resolvable); {st['n_files']} files, {st['n_chunks']} chunks, "
        f"{st['n_parse_failed']} unparsable; edges {st['n_edges']}",
        f"**seed@300 line recall:** {len(seed_lines & core)}/{len(core)}",
        "",
        "| # | BM25 norm | lines | core? | chunk |",
        "|---|---|---|---|---|",
    ]
    top = chunks.filter(pl.col("id").is_in(seed["top_k"])).sort("bm25", descending=True)
    for i, row in enumerate(top.iter_rows(named=True), 1):
        lines.append(f"| {i} | {row['bm25_norm']:.2f} | {row['n_lines']} | {'*' if hits(row) else ''} | `{row['id']}` |")

    first = seed["top_k"][0]
    nb = edges.filter((pl.col("src") == first) | (pl.col("dst") == first))
    by_id = {r["id"]: r for r in chunks.filter(
        pl.col("id").is_in(pl.concat([nb["src"], nb["dst"]]).unique().to_list())).iter_rows(named=True)}
    lines += ["", f"**Neighbours of top hit `{first}`** ({nb.height} edges, * = overlaps core):"]
    for r in nb.sort("type").head(15).iter_rows(named=True):
        other, arrow = (r["dst"], "->") if r["src"] == first else (r["src"], "<-")
        mark = "*" if other in by_id and hits(by_id[other]) else " "
        lines.append(f"- {mark} {r['type']} {arrow} `{other}`")
    return "\n".join(lines) + "\n"


def main(ids: list[str]) -> None:
    tasks = {t["instance_id"]: t for t in map(json.loads, open("data/tasks.jsonl"))}
    for i in ids:
        print(report(tasks[i]))


if __name__ == "__main__":
    main(sys.argv[1:])
