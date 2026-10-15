"""Golden examples for the evaluation harness, with hand-derived values."""
import math
import random

import pytest

from acl.evaluate import Gold, score, truncate

L2, L3, L4 = math.log2(2), math.log2(3), math.log2(4)


def task(core, optional=(), main=(), patch=()):
    regs = lambda rs: [{"path": p, "start": s, "end": e} for p, s, e in rs]
    return {"core_regions": regs(core), "optional_regions": regs(optional),
            "core_files": sorted({p for p, _, _ in core}), "main_files": list(main),
            "patch_files": list(patch)}


# a.py:1-10 is core in a main file (weight 1.5), b.py:5-6 is core, c.py:1-3 optional.
GOLD = Gold(task([("a.py", 1, 10), ("b.py", 5, 6)], [("c.py", 1, 3)], main=["a.py"], patch=["a.py"]),
            {"a.py": 10, "b.py": 20, "c.py": 3})
PREDS = [("x.py", 1, 4), ("a.py", 1, 5), ("b.py", 5, 6)]


def test_golden_coverage_and_noise():
    m = score(GOLD, PREDS, 300)
    # 11 returned lines, 12 core lines, 7 of them returned
    assert m["recall"] == pytest.approx(7 / 12)
    assert m["precision"] == pytest.approx(7 / 11)
    assert m["context_efficiency"] == pytest.approx(7 / 11)
    # Both core regions are touched (region-level perfect), but a.py:6-10 is not returned
    assert m["perfect_recall"] == 1.0 and m["perfect_recall_lines"] == 0.0
    assert m["first_hit_rank"] == 2
    assert m["noise_region"] == pytest.approx(1 / 3)  # x.py touches nothing useful
    assert m["hit_file"] == 1.0 and m["file_recall_patch"] == 1.0


def test_golden_ndcg_paper_vs_official():
    m = score(GOLD, PREDS, 300)
    dcg = 0 / L2 + 1.5 * 5 / L3 + 2 / L4
    # Paper: ideal places a.py:1-10 (gain 15) then b.py:5-6 (gain 2)
    assert m["ndcg"] == pytest.approx(dcg / (15 / L2 + 2 / L3))
    # Official: ideal reorders the returned regions by gain density: a, b, x
    assert m["off_ndcg"] == pytest.approx(dcg / (7.5 / L2 + 2 / L3 + 0 / L4))


def test_golden_budget_cuts_the_crossing_region():
    assert truncate(PREDS, 6) == [("x.py", 1, 4), ("a.py", 1, 2)]
    m = score(GOLD, PREDS, 6)
    assert m["recall"] == pytest.approx(2 / 12)
    assert m["off_recall"] == pytest.approx(2 / 12)
    # Ideal at B=6: six lines of a.py at weight 1.5 in one region
    assert m["ndcg"] == pytest.approx((1.5 * 2 / L3) / 9)
    assert m["recall_cap"] == pytest.approx(6 / 12)


def test_official_ndcg_ignores_coverage():
    # One core line at rank 1 is a "perfect" ranking for the official scorer.
    m = score(GOLD, [("a.py", 1, 1)], 300)
    assert m["off_ndcg"] == 1.0
    assert m["ndcg"] == pytest.approx(1.5 / (15 / L2 + 2 / L3))


def test_phantom_lines_are_clipped_only_in_our_recall():
    # Agent ran `sed -n 1,240p` on a 146-line file.
    g = Gold(task([("d.py", 1, 240)]), {"d.py": 146})
    m = score(g, [("d.py", 1, 146)], 300)
    assert m["recall"] == 1.0 and m["perfect_recall"] == 1.0 and m["perfect_recall_lines"] == 1.0
    assert m["off_recall"] == pytest.approx(146 / 240)
    assert m["ndcg"] == 1.0


def test_end_of_file_and_negative_start_resolve_like_official():
    g = Gold(task([("e.py", 1, -1), ("f.py", -5, -1)]), {"e.py": 50, "f.py": 30})
    assert g.core == {("e.py", i) for i in range(1, 51)} | {("f.py", i) for i in range(26, 31)}
    m = score(g, [("e.py", 1, 50), ("f.py", 26, 30)], 300)
    assert m["recall"] == m["off_recall"] == 1.0


def test_files_missing_at_commit_leave_our_denominator():
    g = Gold(task([("a.py", 1, 10), ("gone.py", 1, 10)]), {"a.py": 10})
    m = score(g, [("a.py", 1, 10)], 300)
    assert m["recall"] == 1.0
    assert m["off_recall"] == pytest.approx(0.5)


def test_line_level_perfect_recall_skips_blank_and_non_py_lines():
    # Core is a.py:1-9 plus a README; lines 4-5 are the blank gap between two
    # functions, which no chunk returns.
    g = Gold(task([("a.py", 1, 9), ("README.md", 1, 3)]), {"a.py": 9, "README.md": 3},
             blank={"a.py": [4, 5]})
    m = score(g, [("a.py", 1, 3), ("a.py", 6, 9)], 300)
    assert m["perfect_recall_lines"] == 1.0
    assert m["perfect_recall"] == 0.0  # the README region is never touched
    assert m["recall"] == pytest.approx(7 / 12)


def test_no_hit():
    m = score(GOLD, [("x.py", 1, 50)], 300)
    assert m["recall"] == 0 and m["ndcg"] == 0 and m["off_ndcg"] == 0
    assert math.isnan(m["first_hit_rank"]) and m["noise_region"] == 1.0


def test_agrees_with_official_when_definitions_coincide():
    # No phantom lines, no main files, no duplicate predictions: recall,
    # precision, efficiency and hit-file must equal the official scorer's.
    rng = random.Random(0)
    files = {f"m{i}.py": rng.randint(20, 400) for i in range(6)}
    for _ in range(300):
        core = []
        for p in rng.sample(sorted(files), 3):
            s = rng.randint(1, files[p]); core.append((p, s, rng.randint(s, files[p])))
        opt = [(p, 1, files[p]) for p in rng.sample(sorted(files), 2)]
        g = Gold(task(core, opt), files)
        preds, seen = [], set()
        for _ in range(rng.randint(1, 12)):
            p = rng.choice(sorted(files)); s = rng.randint(1, files[p]); e = rng.randint(s, min(files[p], s + 80))
            if not any((p, ln) in seen for ln in range(s, e + 1)):
                seen.update((p, ln) for ln in range(s, e + 1)); preds.append((p, s, e))
        for b in (100, 300, 500):
            m = score(g, preds, b)
            for ours, off in [("recall", "off_recall"), ("precision", "off_precision"),
                              ("context_efficiency", "off_context_efficiency"), ("hit_file", "off_hit_file")]:
                assert m[ours] == pytest.approx(m[off]), (ours, b)
            assert 0.0 <= m["ndcg"] <= 1.0
