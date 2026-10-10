"""Split Python files into function, class and module chunks.

Chunk kinds:
- function: every def whose parent is the module or a class (methods included).
  Defs nested inside a function stay part of that function's chunk.
- class: the lines of a class not covered by its methods or nested classes
  (header, decorators, docstring, class attributes).
- module: the lines of a file not covered by any top-level def or class.

Every non-blank line of a parsed file belongs to exactly one chunk, so chunks
never overlap. Class and module chunks may be non-contiguous, hence `spans`.
Blank lines are trimmed from the ends of each span (so separators between defs
cost no budget) but kept inside it, so a docstring or import block stays one
region rather than many.

Chunk ids follow LocAgent's node naming, `path:Qualified.name` for functions
and classes and `path` for the module chunk, so graph nodes map onto chunks
directly.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field


@dataclass
class Chunk:
    id: str
    path: str
    kind: str  # "function" | "class" | "module"
    name: str  # qualified name, "" for module chunks
    spans: list[tuple[int, int]] = field(default_factory=list)  # 1-based, inclusive
    text: str = ""

    @property
    def n_lines(self) -> int:
        return sum(e - s + 1 for s, e in self.spans)


def _start(node: ast.AST) -> int:
    # Decorators sit above the def line and belong to the definition.
    decos = getattr(node, "decorator_list", [])
    return min([node.lineno] + [d.lineno for d in decos])


def _runs(lines: list[int], src_lines: list[str]) -> list[tuple[int, int]]:
    """Collapse sorted line numbers into contiguous spans with blank ends trimmed."""
    spans: list[tuple[int, int]] = []
    for ln in lines:
        if spans and ln == spans[-1][1] + 1:
            spans[-1] = (spans[-1][0], ln)
        else:
            spans.append((ln, ln))
    trimmed = []
    for s, e in spans:
        while s <= e and not src_lines[s - 1].strip():
            s += 1
        while e >= s and not src_lines[e - 1].strip():
            e -= 1
        if s <= e:
            trimmed.append((s, e))
    return trimmed


def _text(src_lines: list[str], spans: list[tuple[int, int]]) -> str:
    return "\n".join("\n".join(src_lines[s - 1:e]) for s, e in spans)


def chunk_file(path: str, source: str) -> list[Chunk]:
    """Chunk one file. Raises SyntaxError if the file does not parse."""
    tree = ast.parse(source)
    src_lines = source.splitlines()
    n = len(src_lines)
    chunks: list[Chunk] = []

    def visit_class(node: ast.ClassDef, prefix: str) -> tuple[int, int]:
        qual = f"{prefix}{node.name}"
        start, end = _start(node), node.end_lineno
        covered: set[int] = set()
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                s, e = add_function(child, f"{qual}.")
            elif isinstance(child, ast.ClassDef):
                s, e = visit_class(child, f"{qual}.")
            else:
                continue
            covered.update(range(s, e + 1))
        own = [ln for ln in range(start, end + 1) if ln not in covered]
        spans = _runs(own, src_lines)
        chunks.append(Chunk(f"{path}:{qual}", path, "class", qual, spans, _text(src_lines, spans)))
        return start, end

    def add_function(node: ast.AST, prefix: str) -> tuple[int, int]:
        qual = f"{prefix}{node.name}"
        spans = [(_start(node), node.end_lineno)]
        chunks.append(Chunk(f"{path}:{qual}", path, "function", qual, spans, _text(src_lines, spans)))
        return spans[0]

    covered: set[int] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            s, e = add_function(node, "")
        elif isinstance(node, ast.ClassDef):
            s, e = visit_class(node, "")
        else:
            continue
        covered.update(range(s, e + 1))

    spans = _runs([ln for ln in range(1, n + 1) if ln not in covered], src_lines)
    if spans:
        chunks.append(Chunk(path, path, "module", "", spans, _text(src_lines, spans)))

    # Repeated names (@overload stubs, property setters, conditional redefinitions)
    # get "#2", "#3", ... so ids stay unique. LocAgent merges them into one graph
    # node, which maps to the first definition.
    seen: dict[str, int] = {}
    for c in sorted(chunks, key=lambda c: c.spans[0][0]):
        seen[c.id] = seen.get(c.id, 0) + 1
        if seen[c.id] > 1:
            c.id = f"{c.id}#{seen[c.id]}"
    return chunks


def chunk_repo(files: dict[str, str]) -> tuple[list[Chunk], list[str]]:
    """Chunk every file in {path: source}. Returns (chunks, paths that failed to parse)."""
    chunks: list[Chunk] = []
    failed: list[str] = []
    for path in sorted(files):
        try:
            chunks.extend(chunk_file(path, files[path]))
        except (SyntaxError, ValueError, RecursionError):
            failed.append(path)
    return chunks, failed
