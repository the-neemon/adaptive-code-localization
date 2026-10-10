from acl.chunking import chunk_repo
from acl.graph import build_graph, chunk_edges

FILES = {
    "src/pkg/__init__.py": "",
    "src/pkg/base.py": (
        "class Base:\n"
        "    def run(self):\n"
        "        return helper()\n"
        "\n"
        "def helper():\n"
        "    return 1\n"
    ),
    "src/pkg/child.py": (
        "from pkg.base import Base, helper\n"
        "\n"
        "class Child(Base):\n"
        "    def go(self):\n"
        "        def local():\n"
        "            return helper()\n"
        "        return local()\n"
    ),
}


def edges():
    chunks, _ = chunk_repo(FILES)
    return set(chunk_edges(build_graph(FILES), chunks))


def test_inherits_and_imports_resolve_under_src():
    e = edges()
    assert ("src/pkg/child.py:Child", "src/pkg/base.py:Base", "inherits") in e
    # file-level `from pkg.base import ...` resolves to src/pkg/base.py
    assert ("src/pkg/child.py", "src/pkg/base.py:helper", "imports") in e


def test_invokes():
    e = edges()
    assert ("src/pkg/base.py:Base.run", "src/pkg/base.py:helper", "invokes") in e


def test_nested_def_maps_to_enclosing_chunk():
    # `local` has no chunk of its own; its call to helper is credited to Child.go
    e = edges()
    assert ("src/pkg/child.py:Child.go", "src/pkg/base.py:helper", "invokes") in e
    assert not any("local" in u or "local" in v for u, v, _ in e)


def test_contains_links_class_to_method():
    e = edges()
    assert ("src/pkg/base.py:Base", "src/pkg/base.py:Base.run", "contains") in e


def test_no_self_loops():
    assert all(u != v for u, v, _ in edges())
