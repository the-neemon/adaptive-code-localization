import numpy as np
import polars as pl

from acl.features import seed_features, text_features

ISSUE = '''Calling `Model.full_clean()` crashes in django/db/models/base.py

Traceback (most recent call last):
  File "manage.py", line 22, in <module>
    main()
ValueError: bad

<details>
pip list
</details>
'''


def test_text_features():
    f = text_features(ISSUE, ["django/db/models/base.py", "manage.py"], {"Model", "full_clean", "main"})
    assert f["has_traceback"] and f["names_py_path"] and f["names_repo_file"] and f["has_env_dump"]
    # Model.full_clean (dotted) and main( are code-like; "Calling" and "crashes" are prose
    assert f["n_known_identifiers"] == 3


def test_text_features_plain_prose():
    f = text_features("Docs should mention the new option.", ["a.py"], {"option"})
    assert not (f["has_traceback"] or f["names_py_path"] or f["names_repo_file"] or f["has_env_dump"])
    assert f["n_known_identifiers"] == 0


def test_text_features_path_suffix_matches_repo_file():
    f = text_features("see models/base.py", ["django/db/models/base.py"], set())
    assert f["names_repo_file"]
    assert not text_features("see other.py", ["django/db/models/base.py"], set())["names_repo_file"]


def test_seed_features():
    ids = ["a", "b", "c"]
    paths = ["x.py", "x.py", "y.py"]
    bm25 = np.array([4.0, 2.0, 0.0])  # normalised 1.0, 0.5, 0.0
    edges = pl.DataFrame([("a", "b", "invokes"), ("c", "a", "inherits"), ("a", "c", "contains")],
                         schema=["src", "dst", "type"], orient="row")
    f = seed_features(ids, paths, bm25, edges, dense_scores=np.array([0.0, 1.0, 2.0]))
    assert f["bm25_margin"] == 0.5
    assert f["bm25_topk_margin"] == 0.75  # 1 - mean(0.5, 0.0)
    assert f["seed_files"] == 2
    assert f["top_degree"] == 3
    assert (f["top_degree_calls"], f["top_degree_imports"], f["top_degree_inherit"]) == (1, 1, 1)
    assert f["topk_degree_mean"] == 2.0  # a: 3, b: 1, c: 2
    assert f["bm25_dense_overlap"] == 1.0  # k = 10 covers all three chunks in both lists
    assert seed_features(ids, paths, bm25, edges, None)["bm25_dense_overlap"] is None


def test_env_dump_from_version_lines_but_not_templates_or_assignments():
    sklearn = "#### Versions\nSystem:\n    python: 3.7.2 (default)\n     numpy: 1.15.2\n     scipy: 1.1.0\n"
    template = "### Python version\n\n_No response_\n\n### Jupyter version\n\n_No response_\n"
    script = "x = 1.5\ny = 2.5\nz = 3.5\n"
    assert text_features(sklearn, [], set())["has_env_dump"]
    assert not text_features(template, [], set())["has_env_dump"]
    assert not text_features(script, [], set())["has_env_dump"]
