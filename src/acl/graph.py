# Adapted from LocAgent's dependency_graph/build_graph.py (v2.3)
# https://github.com/gersteinlab/LocAgent, Apache License 2.0.
# Changes: reads sources from an in-memory {path: source} dict instead of a
# checkout on disk; resolves absolute imports under src/ and lib/ as well as
# the repo root; drops visualisation; adds `chunk_edges` to map graph nodes
# onto acl.chunking chunks; for speed and memory, analyses each definition's
# AST node (parsing one file at a time) instead of re-parsing its source
# segment, and caches the file-level part of find_all_possible_callee
# (identical edges, about 5x faster on django).
"""Typed repository graph: directories, files, classes, functions with
contains / imports / invokes / inherits edges."""
from __future__ import annotations

import ast
import copy
import posixpath
from collections import defaultdict

import networkx as nx

from acl.chunking import Chunk

NODE_TYPE_DIRECTORY = 'directory'
NODE_TYPE_FILE = 'file'
NODE_TYPE_CLASS = 'class'
NODE_TYPE_FUNCTION = 'function'
EDGE_TYPE_CONTAINS = 'contains'
EDGE_TYPE_INHERITS = 'inherits'
EDGE_TYPE_INVOKES = 'invokes'
EDGE_TYPE_IMPORTS = 'imports'

SKIP_DIRS = ['.github', '.git']
# Package roots tried when resolving absolute imports (LocAgent only tries '').
IMPORT_ROOTS = ['', 'src/', 'lib/']


def is_skip_dir(dirname):
    return any(skip_dir in dirname for skip_dir in SKIP_DIRS)


def find_imports(rel_path, tree, top_level_only=False):
    # top_level_only: only include top level imports for classes/functions
    candidates = ast.iter_child_nodes(tree) if top_level_only else ast.walk(tree)

    imports = []
    for node in candidates:
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append({"type": "import", "module": alias.name, "alias": alias.asname})
        elif isinstance(node, ast.ImportFrom):
            import_entities = []
            for alias in node.names:
                if alias.name == '*':
                    import_entities = [{'name': '*', 'alias': None}]
                    break
                import_entities.append({"name": alias.name, "alias": alias.asname})

            if node.level == 0:
                module_name = node.module
            else:
                # Relative import
                package_parts = rel_path.split('/')
                if len(package_parts) >= node.level:
                    package_parts = package_parts[:-node.level]
                else:
                    package_parts = []
                if node.module:
                    module_name = '.'.join(package_parts + [node.module])
                else:
                    module_name = '.'.join(package_parts)

            imports.append({"type": "from", "module": module_name, "entities": import_entities})
    return imports


class CodeAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.nodes = []
        self.node_name_stack = []
        self.node_type_stack = []

    def visit_ClassDef(self, node):
        full_class_name = '.'.join(self.node_name_stack + [node.name])
        self.nodes.append({
            'name': full_class_name,
            'type': NODE_TYPE_CLASS,
            'ast': node,
            'start_line': node.lineno,
            'end_line': node.end_lineno,
        })
        self.node_name_stack.append(node.name)
        self.node_type_stack.append(NODE_TYPE_CLASS)
        self.generic_visit(node)
        self.node_name_stack.pop()
        self.node_type_stack.pop()

    def visit_FunctionDef(self, node):
        # A class's __init__ is folded into the class node (see analyze_init).
        if self.node_type_stack and self.node_type_stack[-1] == NODE_TYPE_CLASS and node.name == '__init__':
            return
        self._visit_func(node)

    def visit_AsyncFunctionDef(self, node):
        self._visit_func(node)

    def _visit_func(self, node):
        full_function_name = '.'.join(self.node_name_stack + [node.name])
        self.nodes.append({
            'name': full_function_name,
            'parent_type': self.node_type_stack[-1] if self.node_type_stack else None,
            'type': NODE_TYPE_FUNCTION,
            'ast': node,
            'start_line': node.lineno,
            'end_line': node.end_lineno,
        })
        self.node_name_stack.append(node.name)
        self.node_type_stack.append(NODE_TYPE_FUNCTION)
        self.generic_visit(node)
        self.node_name_stack.pop()
        self.node_type_stack.pop()


def analyze_file(source):
    tree = ast.parse(source)
    analyzer = CodeAnalyzer()
    try:
        analyzer.visit(tree)
    except RecursionError:
        pass
    return tree, analyzer.nodes


def resolve_module(module_name, files):
    """Resolve a module name to a repo-relative file path, or None."""
    if not module_name:
        return None
    base = module_name.replace('.', '/')
    for root in IMPORT_ROOTS:
        for cand in (f'{root}{base}.py', f'{root}{base}/__init__.py'):
            if cand in files:
                return cand
    return None


def add_imports(root_node, imports, graph, files):
    for imp in imports:
        if imp['type'] == 'import':
            imp_filename = resolve_module(imp['module'], files)
            if imp_filename and graph.has_node(imp_filename):
                graph.add_edge(root_node, imp_filename, type=EDGE_TYPE_IMPORTS, alias=imp['alias'])
        elif imp['type'] == 'from':
            module_name = imp['module']
            entities = imp['entities']

            if len(entities) == 1 and entities[0]['name'] == '*':
                # Handle 'from module import *' as 'import module' statement
                imp_filename = resolve_module(module_name, files)
                if imp_filename and graph.has_node(imp_filename):
                    graph.add_edge(root_node, imp_filename, type=EDGE_TYPE_IMPORTS, alias=None)
                continue

            for entity in entities:
                entity_name, entity_alias = entity['name'], entity['alias']
                entity_filename = resolve_module(f"{module_name}.{entity_name}", files)
                if entity_filename:
                    # Entity is a submodule
                    if graph.has_node(entity_filename):
                        graph.add_edge(root_node, entity_filename, type=EDGE_TYPE_IMPORTS, alias=entity_alias)
                else:
                    # Entity might be an attribute inside the module
                    imp_filename = resolve_module(module_name, files)
                    if imp_filename:
                        node = f"{imp_filename}:{entity_name}"
                        if graph.has_node(node):
                            graph.add_edge(root_node, node, type=EDGE_TYPE_IMPORTS, alias=entity_alias)
                        elif graph.has_node(imp_filename):
                            graph.add_edge(root_node, imp_filename, type=EDGE_TYPE_IMPORTS, alias=entity_alias)


def build_graph(files, fuzzy_search=True, global_import=False):
    """Build the graph for {repo-relative path: source} (.py files only)."""
    graph = nx.MultiDiGraph()
    file_imports = {}

    ## add nodes
    graph.add_node('/', type=NODE_TYPE_DIRECTORY)
    for filename in sorted(files):
        dirname = posixpath.dirname(filename) or '/'
        if is_skip_dir(dirname):
            continue
        try:
            tree, nodes = analyze_file(files[filename])
        except (SyntaxError, ValueError):
            # Skip files that cannot be parsed (LocAgent adds the bare file node first)
            graph.add_node(filename, type=NODE_TYPE_FILE)
            _add_dir_chain(graph, dirname)
            graph.add_edge(dirname, filename, type=EDGE_TYPE_CONTAINS)
            continue
        # Only the import list is kept: holding every file's AST at once needs
        # gigabytes on sympy (its generated Rubi rules are 26 MB of source).
        file_imports[filename] = find_imports(filename, tree)
        del tree
        graph.add_node(filename, type=NODE_TYPE_FILE)
        _add_dir_chain(graph, dirname)
        graph.add_edge(dirname, filename, type=EDGE_TYPE_CONTAINS)

        # add function/class nodes
        for node in nodes:
            full_name = f'{filename}:{node["name"]}'
            graph.add_node(full_name, type=node['type'],
                           start_line=node['start_line'], end_line=node['end_line'])

        # add edges with type=contains
        for node in nodes:
            full_name = f'{filename}:{node["name"]}'
            name_list = node['name'].split('.')
            if len(name_list) == 1:
                graph.add_edge(filename, full_name, type=EDGE_TYPE_CONTAINS)
            else:
                full_parent_name = f'{filename}:{".".join(name_list[:-1])}'
                graph.add_edge(full_parent_name, full_name, type=EDGE_TYPE_CONTAINS)

    ## add imports edges (file -> class/function)
    for filename, imports in file_imports.items():
        add_imports(filename, imports, graph, files)

    global_name_dict = defaultdict(list)
    if global_import:
        for node in graph.nodes():
            global_name_dict[node.split(':')[-1].split('.')[-1]].append(node)

    ## add edges start from class/function
    file_callees = {}
    cur_file, file_defs = None, {}
    for node, attributes in list(graph.nodes(data=True)):
        if attributes.get('type') not in [NODE_TYPE_CLASS, NODE_TYPE_FUNCTION]:
            continue

        # A file's definitions are contiguous in node order, so each file is
        # parsed once here and its tree freed when the next file starts.
        filename = node.split(':', 1)[0]
        if filename != cur_file:
            # Later definitions of a repeated name win, as add_node does above.
            cur_file, file_defs = filename, {d['name']: d['ast'] for d in analyze_file(files[filename])[1]}

        # LocAgent re-parses ast.get_source_segment(node), which starts at the
        # `def`/`class` line and so drops the definition's own decorators.
        # Analysing a copy without them keeps its edges identical.
        caller_code_tree = copy.copy(file_defs[node.split(':', 1)[1]])
        caller_code_tree.decorator_list = []

        # construct possible callee dict (name -> node) based on graph connectivity
        callee_nodes, callee_alias = find_all_possible_callee(node, graph, file_callees)
        if fuzzy_search:
            # for nodes with the same suffix, keep every nodes
            callee_name_dict = defaultdict(list)
            for callee_node in set(callee_nodes):
                callee_name_dict[callee_node.split(':')[-1].split('.')[-1]].append(callee_node)
            for alias, callee_node in callee_alias.items():
                callee_name_dict[alias].append(callee_node)
        else:
            # for nodes with the same suffix, only keep the nearest node
            callee_name_dict = {
                callee_node.split(':')[-1].split('.')[-1]: callee_node
                for callee_node in callee_nodes[::-1]
            }
            callee_name_dict.update(callee_alias)

        # analysis invokes and inherits, add (top-level) imports edges (class/function -> class/function)
        if attributes.get('type') == NODE_TYPE_CLASS:
            invocations, inheritances = analyze_init(node, caller_code_tree, graph, files)
        else:
            invocations = analyze_invokes(node, caller_code_tree, graph, files)
            inheritances = []

        for edge_type, names in ((EDGE_TYPE_INVOKES, invocations), (EDGE_TYPE_INHERITS, inheritances)):
            for callee_name in set(names):
                callee_node = callee_name_dict.get(callee_name)
                if callee_node:
                    for callee in (callee_node if isinstance(callee_node, list) else [callee_node]):
                        graph.add_edge(node, callee, type=edge_type)
                elif global_import:
                    for global_fuzzy_node in global_name_dict.get(callee_name, []):
                        graph.add_edge(node, global_fuzzy_node, type=edge_type)

    return graph


def _add_dir_chain(graph, dirname):
    while dirname != '/' and not graph.has_node(dirname):
        graph.add_node(dirname, type=NODE_TYPE_DIRECTORY)
        parent = posixpath.dirname(dirname) or '/'
        graph.add_edge(parent, dirname, type=EDGE_TYPE_CONTAINS)
        dirname = parent


def get_inner_nodes(query_node, src_node, graph):
    inner_nodes = []
    for _, dst_node, attr in graph.edges(src_node, data=True):
        if attr['type'] == EDGE_TYPE_CONTAINS and dst_node != query_node:
            inner_nodes.append(dst_node)
            if graph.nodes[dst_node]['type'] == NODE_TYPE_CLASS:  # only include class's inner nodes
                inner_nodes.extend(get_inner_nodes(query_node, dst_node, graph))
    return inner_nodes


def find_all_possible_callee(node, graph, file_cache=None):
    callee_nodes, callee_alias = [], {}
    cur_node = node
    pre_node = node

    def find_parent(_cur_node):
        for predecessor in graph.predecessors(_cur_node):
            for key, attr in graph.get_edge_data(predecessor, _cur_node).items():
                if attr['type'] == EDGE_TYPE_CONTAINS:
                    return predecessor

    while True:
        callee_nodes.extend(get_inner_nodes(pre_node, cur_node, graph))

        if graph.nodes[cur_node]['type'] == NODE_TYPE_FILE:
            # Everything below depends only on the file, and files gain no edges
            # while callees are resolved, so it is computed once per file.
            if file_cache is not None and cur_node in file_cache:
                cached_nodes, cached_alias = file_cache[cur_node]
                callee_nodes.extend(cached_nodes)
                callee_alias.update(cached_alias)
                break
            n_before, alias_before = len(callee_nodes), dict(callee_alias)

            # check recursive imported files
            file_list = []
            file_stack = [cur_node]
            while len(file_stack) > 0:
                for _, dst_node, attr in graph.edges(file_stack.pop(), data=True):
                    if attr['type'] == EDGE_TYPE_IMPORTS and dst_node not in file_list + [cur_node]:
                        if graph.nodes[dst_node]['type'] == NODE_TYPE_FILE and dst_node.endswith('__init__.py'):
                            file_list.append(dst_node)
                            file_stack.append(dst_node)

            for file in file_list:
                callee_nodes.extend(get_inner_nodes(cur_node, file, graph))
                for _, dst_node, attr in graph.edges(file, data=True):
                    if attr['type'] == EDGE_TYPE_IMPORTS:
                        if attr['alias'] is not None:
                            callee_alias[attr['alias']] = dst_node
                        if graph.nodes[dst_node]['type'] in [NODE_TYPE_FILE, NODE_TYPE_CLASS]:
                            callee_nodes.extend(get_inner_nodes(file, dst_node, graph))
                        if graph.nodes[dst_node]['type'] in [NODE_TYPE_FUNCTION, NODE_TYPE_CLASS]:
                            callee_nodes.append(dst_node)

            # check imported functions and classes
            for _, dst_node, attr in graph.edges(cur_node, data=True):
                if attr['type'] == EDGE_TYPE_IMPORTS:
                    if attr['alias'] is not None:
                        callee_alias[attr['alias']] = dst_node
                    if graph.nodes[dst_node]['type'] in [NODE_TYPE_FILE, NODE_TYPE_CLASS]:
                        callee_nodes.extend(get_inner_nodes(cur_node, dst_node, graph))
                    if graph.nodes[dst_node]['type'] in [NODE_TYPE_FUNCTION, NODE_TYPE_CLASS]:
                        callee_nodes.append(dst_node)

            if file_cache is not None:
                assert not alias_before  # aliases are only ever set in this block
                file_cache[cur_node] = (callee_nodes[n_before:], dict(callee_alias))
            break

        pre_node = cur_node
        cur_node = find_parent(cur_node)

    return callee_nodes, callee_alias


def _process_decorator_node(decorator_node, add_invoke):
    if isinstance(decorator_node, ast.Name):
        add_invoke(decorator_node.id)
    else:
        for sub_node in ast.walk(decorator_node):
            if isinstance(sub_node, ast.Call) and isinstance(sub_node.func, ast.Name):
                add_invoke(sub_node.func.id)
            elif isinstance(sub_node, ast.Attribute):
                add_invoke(sub_node.attr)


def analyze_init(node, code_tree, graph, files):
    caller_name = node.split(':')[-1].split('.')[-1]
    rel_path = node.split(':')[0]
    invocations = []
    inheritances = []

    for ast_node in ast.walk(code_tree):
        if isinstance(ast_node, ast.ClassDef) and ast_node.name == caller_name:
            add_imports(node, find_imports(rel_path, ast_node, top_level_only=True), graph, files)

            for inheritance_node in ast_node.bases:
                if isinstance(inheritance_node, ast.Attribute):
                    inheritances.append(inheritance_node.attr)
                if isinstance(inheritance_node, ast.Name):
                    inheritances.append(inheritance_node.id)

            for decorator_node in ast_node.decorator_list:
                _process_decorator_node(decorator_node, invocations.append)

            for body_item in ast_node.body:
                if isinstance(body_item, ast.FunctionDef) and body_item.name == '__init__':
                    add_imports(node, find_imports(rel_path, body_item, top_level_only=True), graph, files)
                    for decorator_node in body_item.decorator_list:
                        _process_decorator_node(decorator_node, invocations.append)
                    for sub_node in ast.walk(body_item):
                        if isinstance(sub_node, ast.Call):
                            if isinstance(sub_node.func, ast.Name):  # function or class
                                invocations.append(sub_node.func.id)
                            if isinstance(sub_node.func, ast.Attribute):  # member function
                                invocations.append(sub_node.func.attr)
                    break
            break

    return invocations, inheritances


def analyze_invokes(node, code_tree, graph, files):
    caller_name = node.split(':')[-1].split('.')[-1]
    rel_path = node.split(':')[0]
    invocations = []

    def traverse_call(_node):
        for child in ast.iter_child_nodes(_node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                # Skip inner function/class definition
                continue
            elif isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    invocations.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    invocations.append(child.func.attr)
            traverse_call(child)

    for ast_node in ast.walk(code_tree):
        if isinstance(ast_node, (ast.FunctionDef, ast.AsyncFunctionDef)) and ast_node.name == caller_name:
            add_imports(node, find_imports(rel_path, ast_node, top_level_only=True), graph, files)
            for decorator_node in ast_node.decorator_list:
                _process_decorator_node(decorator_node, invocations.append)
            traverse_call(ast_node)
            break

    return invocations


def chunk_edges(graph: nx.MultiDiGraph, chunks: list[Chunk]) -> list[tuple[str, str, str]]:
    """Map graph edges onto chunk ids, as deduplicated (src, dst, type) triples.

    Files map to their module chunk; classes and functions to their own chunk.
    Nodes with no chunk of their own (defs nested inside functions) map to the
    nearest enclosing chunk. Directory nodes and self-loops are dropped.
    """
    ids = {c.id for c in chunks}

    def to_chunk(node: str) -> str | None:
        if ':' not in node:
            return node if node in ids else None  # file -> module chunk
        path, qual = node.split(':', 1)
        parts = qual.split('.')
        while parts:
            cid = f'{path}:{".".join(parts)}'
            if cid in ids:
                return cid
            parts.pop()
        return None

    cache: dict[str, str | None] = {}
    out = set()
    for u, v, data in graph.edges(data=True):
        cu = cache[u] if u in cache else cache.setdefault(u, to_chunk(u))
        cv = cache[v] if v in cache else cache.setdefault(v, to_chunk(v))
        if cu and cv and cu != cv:
            out.add((cu, cv, data['type']))
    return sorted(out)
