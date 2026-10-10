from acl.chunking import chunk_file, chunk_repo

SRC = '''\
"""Module docstring."""
import os

X = 1


@decorator
def top(a):
    def inner():
        return a
    return inner


class Outer(Base):
    """Doc."""
    attr = 2

    def method(self):
        return 1

    class Nested:
        y = 3

        async def deep(self):
            pass

    other = 4


Y = 5
'''


def by_id(chunks):
    return {c.id: c for c in chunks}


def test_chunk_ids_and_kinds():
    cs = by_id(chunk_file("pkg/m.py", SRC))
    assert set(cs) == {
        "pkg/m.py",
        "pkg/m.py:top",
        "pkg/m.py:Outer",
        "pkg/m.py:Outer.method",
        "pkg/m.py:Outer.Nested",
        "pkg/m.py:Outer.Nested.deep",
    }
    assert cs["pkg/m.py"].kind == "module"
    assert cs["pkg/m.py:Outer"].kind == "class"
    assert cs["pkg/m.py:Outer.method"].kind == "function"


def test_spans():
    cs = by_id(chunk_file("pkg/m.py", SRC))
    # Decorator line is part of the function; the nested def stays inside it.
    assert cs["pkg/m.py:top"].spans == [(7, 11)]
    # Class chunk = header/doc/attrs, plus the attr after the nested class.
    assert cs["pkg/m.py:Outer"].spans == [(14, 16), (27, 27)]
    assert cs["pkg/m.py:Outer.Nested"].spans == [(21, 22)]
    # The blank line between `import os` and `X = 1` stays inside the span.
    assert cs["pkg/m.py"].spans == [(1, 4), (30, 30)]
    assert cs["pkg/m.py:top"].text.startswith("@decorator\ndef top(a):")


def test_every_nonblank_line_in_exactly_one_chunk():
    chunks = chunk_file("pkg/m.py", SRC)
    seen = [ln for c in chunks for s, e in c.spans for ln in range(s, e + 1)]
    assert len(seen) == len(set(seen))
    lines = SRC.splitlines()
    nonblank = {i + 1 for i, line in enumerate(lines) if line.strip()}
    assert nonblank <= set(seen)


def test_blank_lines_inside_docstring_do_not_split_the_span():
    src = 'class A:\n    """One.\n\n    Two.\n    """\n\n    def f(self):\n        pass\n'
    assert by_id(chunk_file("a.py", src))["a.py:A"].spans == [(1, 5)]


def test_n_lines():
    cs = by_id(chunk_file("pkg/m.py", SRC))
    assert cs["pkg/m.py:Outer"].n_lines == 4


def test_file_with_only_defs_has_no_module_chunk():
    cs = by_id(chunk_file("a.py", "def f():\n    pass\n"))
    assert set(cs) == {"a.py:f"}


def test_repo_reports_unparsable_files():
    chunks, failed = chunk_repo({"ok.py": "x = 1\n", "bad.py": "def (:\n"})
    assert failed == ["bad.py"]
    assert [c.id for c in chunks] == ["ok.py"]


def test_repeated_names_get_unique_ids():
    src = (
        "class A:\n"
        "    @property\n    def x(self):\n        return 1\n"
        "    @x.setter\n    def x(self, v):\n        pass\n"
    )
    cs = by_id(chunk_file("a.py", src))
    assert cs["a.py:A.x"].spans == [(2, 4)]
    assert cs["a.py:A.x#2"].spans == [(5, 7)]
