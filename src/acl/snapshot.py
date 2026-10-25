"""Read the Python files of a repository at a given commit from a bare mirror.

Each repository gets one bare repo under `data/mirrors/` holding only the
pre-fix commits we need, fetched shallowly (depth 1) in a single fetch so
files shared between snapshots are downloaded once. Full histories are
several GB and download too slowly on campus. A task's snapshot is then read
straight from git objects, so no per-task checkout is ever written to disk.

Usage: python -m acl.snapshot [--tasks PATH] [repo ...]  (fetches the commits listed in the task file)
"""
from __future__ import annotations

import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

MIRRORS = Path("data/mirrors")


def mirror_path(mirrors: Path, repo: str) -> Path:
    return mirrors / f"{repo.replace('/', '__')}.git"


def _git(path: Path, *args: str, stdin: str | None = None) -> str:
    return subprocess.run(["git", "-C", str(path), *args], input=stdin, text=True,
                          capture_output=True, check=True).stdout


def fetch_commits(mirrors: Path, repo: str, commits: list[str]) -> Path:
    """Fetch the given commits with only their .py file contents.

    Step 1 fetches commits and trees without any file contents (a partial,
    depth-1 fetch). Step 2 lists the .py blobs those trees point to and fetches
    just the missing ones. Skipping everything else (test images, data files)
    makes repos like matplotlib and astropy several times smaller.
    """
    path = mirror_path(mirrors, repo)
    if not path.exists():
        subprocess.run(["git", "init", "-q", "--bare", str(path)], check=True)
    # Mirrors made by an early version of this function have no origin URL
    # (requests, flask, seaborn), so (re)write it rather than only at creation.
    # Not `remote get-url`: with no URL set it prints the remote's name and succeeds.
    if subprocess.run(["git", "-C", str(path), "config", "--get", "remote.origin.url"],
                      capture_output=True).returncode:
        _git(path, "config", "remote.origin.url", f"https://github.com/{repo}.git")
        # Mark origin as a promisor so git tolerates the blobs we never fetch.
        _git(path, "config", "remote.origin.promisor", "true")
        _git(path, "config", "remote.origin.partialclonefilter", "blob:none")
    new = [c for c in commits if not has_commit(path, c)]
    if new:
        _git(path, "fetch", "-q", "--depth", "1", "--no-tags", "--filter=blob:none", "origin", *new)

    # --missing=print reports absent objects as "?<sha>" without fetching them.
    listing = _git(path, "rev-list", "--objects", "--missing=print", *commits)
    absent = {ln[1:] for ln in listing.splitlines() if ln.startswith("?")}
    wanted = {sha for c in commits for sha, _ in _py_entries(path, c)} & absent
    if wanted:
        _git(path, "fetch", "-q", "--no-tags", "--filter=blob:none", "--stdin", "origin",
             stdin="".join(f"{sha}\n" for sha in sorted(wanted)))
    return path


def has_commit(mirror: Path, commit: str) -> bool:
    # Not `cat-file -e`: in a partial clone that lazily fetches the missing
    # commit with its whole history of trees. --missing=print never fetches.
    r = subprocess.run(["git", "-C", str(mirror), "rev-list", "--no-walk", "--missing=print", commit],
                       capture_output=True)
    return r.returncode == 0


def _py_entries(mirror: Path, commit: str) -> list[tuple[str, str]]:
    """(blob sha, path) of every regular .py file at `commit`."""
    ls = subprocess.run(["git", "-C", str(mirror), "ls-tree", "-r", "-z", commit],
                        capture_output=True, check=True).stdout
    entries: list[tuple[str, str]] = []
    for rec in ls.split(b"\0"):
        if not rec:
            continue
        meta, path = rec.split(b"\t", 1)
        mode, kind, sha = meta.split()
        if kind == b"blob" and mode in (b"100644", b"100755") and path.endswith(b".py"):
            entries.append((sha.decode(), path.decode()))
    return entries


def read_python_files(mirror: Path, commit: str) -> dict[str, str]:
    """{repo-relative path: source} for every regular .py file at `commit`.

    Symlinks (mode 120000) and submodules are skipped, as in LocAgent. Files
    that are not valid UTF-8 are decoded with replacement so line numbers stay
    aligned with the ground truth.
    """
    entries = _py_entries(mirror, commit)
    batch = subprocess.run(["git", "-C", str(mirror), "cat-file", "--batch"],
                           input="".join(f"{sha}\n" for sha, _ in entries).encode(),
                           capture_output=True, check=True).stdout
    files: dict[str, str] = {}
    pos = 0
    for _, path in entries:
        header_end = batch.index(b"\n", pos)
        size = int(batch[pos:header_end].split()[2])
        body = batch[header_end + 1: header_end + 1 + size]
        files[path] = body.decode("utf-8", errors="replace")
        pos = header_end + 1 + size + 1  # trailing newline after each object
    return files


def main(argv: list[str]) -> None:
    tasks_path = "data/tasks.jsonl"
    if argv[:1] == ["--tasks"]:
        tasks_path, argv = argv[1], argv[2:]
    repos = argv
    by_repo: dict[str, set[str]] = defaultdict(set)
    for line in open(tasks_path):
        t = json.loads(line)
        by_repo[t["repo"]].add(t["base_commit"])
    for repo in repos or sorted(by_repo, key=lambda r: len(by_repo[r])):
        fetch_commits(MIRRORS, repo, sorted(by_repo[repo]))
        print(f"{repo}: {len(by_repo[repo])} commits", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
