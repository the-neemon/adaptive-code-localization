"""Repository-disjoint splits: a fixed dev group plus leave-one-repository-out folds.

The Python subset has 15 repositories and django alone holds 41% of tasks, so
a single train/dev/test split is not viable (docs/stage1.md).

- dev: xarray, pytest and openlibrary (56 tasks, Verified and Pro). Used only
  to tune hyperparameters (expansion decay lambda, RRF constant, any
  thresholds). Never used to evaluate.
- folds: the other 12 repositories, one per fold, except the five smallest
  (pylint, astropy, requests, seaborn, flask: 29 tasks), which share a fold so
  that no fold has only one or two tasks. The router is trained on all folds
  but one and tested on the held-out one.

Usage: python -m acl.splits  (writes configs/splits.json)
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

OUT = Path("configs/splits.json")
DEV_REPOS = ["pydata/xarray", "pytest-dev/pytest", "internetarchive/openlibrary"]
SMALL_FOLD = ["pylint-dev/pylint", "astropy/astropy", "psf/requests", "mwaskom/seaborn", "pallets/flask"]


def make_splits(tasks: list[dict]) -> dict:
    repos = Counter(t["repo"] for t in tasks)
    folds: dict[str, list[str]] = {}
    for repo, _ in repos.most_common():
        if repo in DEV_REPOS:
            continue
        name = "small" if repo in SMALL_FOLD else repo.split("/")[1]
        folds.setdefault(name, []).append(repo)
    by_repo = lambda rs: sorted(t["instance_id"] for t in tasks if t["repo"] in rs)
    return {
        "dev": {"repos": DEV_REPOS, "ids": by_repo(DEV_REPOS)},
        "folds": {name: {"repos": rs, "ids": by_repo(rs)} for name, rs in folds.items()},
    }


def load(path: Path = OUT) -> dict:
    return json.loads(path.read_text())


def split_of(splits: dict) -> dict[str, str]:
    """{instance_id: "dev" or fold name}."""
    out = {i: "dev" for i in splits["dev"]["ids"]}
    for name, f in splits["folds"].items():
        out.update({i: name for i in f["ids"]})
    return out


def main() -> None:
    tasks = [json.loads(line) for line in open("data/tasks.jsonl")]
    s = make_splits(tasks)
    n = len(s["dev"]["ids"]) + sum(len(f["ids"]) for f in s["folds"].values())
    assert n == len(tasks), "every task must be in exactly one split"
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(s, indent=1) + "\n")
    print(f"dev: {len(s['dev']['ids'])} tasks {s['dev']['repos']}")
    for name, f in s["folds"].items():
        print(f"fold {name}: {len(f['ids'])} tasks {f['repos']}")


if __name__ == "__main__":
    main()
