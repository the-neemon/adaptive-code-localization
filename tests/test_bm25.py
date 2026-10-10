import numpy as np

from acl.bm25 import BM25, fill_budget, minmax, tokenize
from acl.chunking import Chunk


def test_tokenize_splits_identifiers_and_keeps_whole():
    toks = tokenize("call get_FOO_display() on HTTPResponse")
    assert "get_foo_display" in toks
    assert {"get", "foo", "display", "httpresponse", "http", "response", "call"} <= set(toks)
    # stopwords and single characters are dropped
    assert "on" not in toks
    assert tokenize("a b x_y") == ["x_y"]


def test_bm25_prefers_matching_and_rarer_terms():
    docs = [["parse", "url", "host"], ["render", "template"], ["parse", "json"], ["parse", "body"]]
    s = BM25(docs).scores(["url", "parse"])
    assert s.argmax() == 0
    assert s[1] == 0
    # "url" is rarer than "parse", so doc 0 beats docs that only match "parse"
    assert s[0] > s[2] > 0


def test_bm25_unknown_query_terms_score_zero():
    s = BM25([["a1"], ["b1"]]).scores(["zzz"])
    assert np.all(s == 0)


def test_minmax():
    assert minmax(np.array([2.0, 4.0, 3.0])).tolist() == [0.0, 1.0, 0.5]
    assert minmax(np.array([1.0, 1.0])).tolist() == [0.0, 0.0]


def chunk(path, *spans):
    return Chunk(id=path, path=path, kind="function", name="", spans=list(spans))


def test_fill_budget_truncates_the_crossing_region():
    ranked = [chunk("a.py", (1, 50)), chunk("b.py", (10, 109)), chunk("c.py", (1, 5))]
    assert fill_budget(ranked, 100) == [("a.py", 1, 50), ("b.py", 10, 59)]


def test_fill_budget_exact_fit_and_multi_span():
    ranked = [chunk("m.py", (1, 3), (10, 12)), chunk("n.py", (1, 4))]
    assert fill_budget(ranked, 6) == [("m.py", 1, 3), ("m.py", 10, 12)]
    assert fill_budget(ranked, 100) == [("m.py", 1, 3), ("m.py", 10, 12), ("n.py", 1, 4)]
