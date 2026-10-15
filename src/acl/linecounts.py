"""Line counts of every ground-truth file at its task's pre-fix commit.

The scorer needs them to resolve SWE-Explore regions written relative to the
end of a file (end = -1, start < 0) and to clip ranges that run past the end of
a file (agents run `sed -n 1,240p` on 146-line files). Ground truth also names
non-.py files (pyproject.toml, .rst, .html), whose blobs `acl.snapshot` never
fetched, so this step fetches just those.

Counts follow SWE-Explore's runner: `len(text.splitlines())` on the file decoded
with errors ignored. A path missing at the commit gets no entry.

The blank lines of ground-truth .py files are recorded too: no chunk contains
the blank lines between definitions, so line-level perfect recall skips them.

Usage: python -m acl.linecounts  (reads data/tasks.jsonl, writes data/line_counts.json
                                  and data/blank_lines.json)
"""
from __future__ import annotations

import json
import subprocess
from collections import defaultdict
from pathlib import Path

from acl import snapshot

OUT = Path("data/line_counts.json")
BLANK_OUT = Path("data/blank_lines.json")


def gt_paths(task: dict) -> set[str]:
    return {r["path"] for r in task["core_regions"] + task["optional_regions"]}


def _blobs(mirror: Path, commit: str, paths: set[str]) -> dict[str, str]:
    """{path: blob sha} for the given paths that exist as regular files at `commit`."""
    ls = subprocess.run(["git", "-C", str(mirror), "ls-tree", "-r", "-z", commit, "--", *sorted(paths)],
                        capture_output=True, check=True).stdout
    out = {}
    for rec in ls.split(b"\0"):
        if not rec:
            continue
        meta, path = rec.split(b"\t", 1)
        mode, kind, sha = meta.split()
        if kind == b"blob" and mode in (b"100644", b"100755") and path.decode() in paths:
            out[path.decode()] = sha.decode()
    return out


def _count_lines(mirror: Path, shas: list[str]) -> dict[str, tuple[int, list[int]]]:
    """{sha: (line count, 1-based numbers of blank lines)}."""
    batch = subprocess.run(["git", "-C", str(mirror), "cat-file", "--batch"],
                           input="".join(f"{s}\n" for s in shas).encode(),
                           capture_output=True, check=True).stdout
    counts, pos = {}, 0
    for sha in shas:
        header_end = batch.index(b"\n", pos)
        size = int(batch[pos:header_end].split()[2])
        body = batch[header_end + 1: header_end + 1 + size]
        text = body.decode("utf-8", errors="ignore").splitlines()
        counts[sha] = (len(text), [i + 1 for i, ln in enumerate(text) if not ln.strip()])
        pos = header_end + 1 + size + 1
    return counts


def line_counts(tasks: list[dict], mirrors: Path = snapshot.MIRRORS
                ) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, list[int]]]]:
    by_repo: dict[str, list[dict]] = defaultdict(list)
    for t in tasks:
        by_repo[t["repo"]].append(t)
    out: dict[str, dict[str, int]] = {}
    blank: dict[str, dict[str, list[int]]] = {}
    for repo, ts in by_repo.items():
        mirror = snapshot.mirror_path(mirrors, repo)
        blobs = {t["instance_id"]: _blobs(mirror, t["base_commit"], gt_paths(t)) for t in ts}
        # .py blobs are already local; fetch the rest in one request per repo.
        other = sorted({sha for b in blobs.values() for p, sha in b.items() if not p.endswith(".py")})
        if other:
            subprocess.run(["git", "-C", str(mirror), "fetch", "-q", "--no-tags", "--filter=blob:none",
                            "--stdin", "origin"], input="".join(f"{s}\n" for s in other).encode(),
                           capture_output=True, check=True)
        counts = _count_lines(mirror, sorted({s for b in blobs.values() for s in b.values()}))
        for iid, b in blobs.items():
            out[iid] = {p: counts[sha][0] for p, sha in sorted(b.items())}
            blank[iid] = {p: counts[sha][1] for p, sha in sorted(b.items()) if p.endswith(".py")}
        print(f"{repo}: {len(ts)} tasks, {len(other)} non-.py blobs", flush=True)
    return out, blank


def load(path: Path = OUT) -> dict[str, dict[str, int]]:
    return json.loads(path.read_text())


def load_blank(path: Path = BLANK_OUT) -> dict[str, dict[str, list[int]]]:
    return json.loads(path.read_text())


def main() -> None:
    tasks = [json.loads(line) for line in open("data/tasks.jsonl")]
    counts, blank = line_counts(tasks)
    OUT.write_text(json.dumps(counts))
    BLANK_OUT.write_text(json.dumps(blank))
    print(f"-> {OUT}, {BLANK_OUT}")


if __name__ == "__main__":
    main()
