# Stage 1: data, snapshots, chunking, BM25, graph

Status as of 2026-10-10: complete. All 500 Python tasks prepared. Checkpoint
from the proposal: outputs hand-checked on 10 tasks (see below).

## Dataset

SWE-Explore (`SWE-Explore-Bench/SWE-Explore-Bench` on Hugging Face, file
`bench.final.public.jsonl`) has 848 issues: 451 from SWE-bench Verified, 215 from
SWE-bench Pro, 182 from SWE-bench Multilingual. The records carry the ground
truth but no issue text, repository or commit, so `acl.data` joins them with
SWE-bench Verified and Pro on `instance_id`. Pro ids carry an `instance_`
prefix that SWE-Explore drops, and 21 of SWE-Explore's Pro tasks are no longer
in Pro's default split; they are in its `v1` config.

**Python filter:** every file in `modified_core_files` ends in `.py`. 52
records have no `modified_core_files`; for those the core read files are used.

| | Tasks | Repositories |
|---|---|---|
| Python subset | **500** | **15** |
| from Verified | 410 | 12 |
| from Pro | 90 | 3 (ansible, qutebrowser, openlibrary) |

Per repository: django 207, sympy 66, ansible 40, scikit-learn 31, qutebrowser 30,
matplotlib 29, openlibrary 20, xarray 18, pytest 18, sphinx 12, pylint 10,
astropy 9, requests 7, seaborn 2, flask 1. 499 distinct pre-fix commits.

**Consequence for splits:** the proposal assumed many repositories. With 15,
and django alone at 41% of tasks, a single repository-disjoint
train/dev/test split is not viable. Use leave-one-repository-out (the
proposal's fallback), probably grouping the four smallest repositories.

## Ground truth characteristics that affect the study

1. **Core regions are often whole files.** Agents `cat` entire files, so
   regions like `lib/matplotlib/colorbar.py:1-1637` are common. Median core
   size over the Python subset is about 1,160 lines, and only 12.6% of tasks
   have 300 or fewer core lines (see the coverage table below). At B = 300,
   Recall@B is capped well below 1 for most tasks, and Perfect Recall@B is
   unreachable for about 87% of them.
   SWE-Explore's nDCG@B is normalised against the best reordering of the
   regions a system returned, so it rewards ranking rather than raw coverage
   and should suffer less from this; that argues for keeping it as the
   primary metric, but Stage 2 should confirm it on the seed.
2. **Ranges run past the end of the file.** Agents run `sed -n 1,240p` on
   short files and SWE-Explore stores the requested range unclipped. Its
   official scorer only clips `-1` (end of file), so phantom lines stay in the
   recall denominator. Stage 2 has to decide: replicate the official scorer
   (comparable with the paper) or clip (cleaner); reporting both is cheap.
3. **Non-`.py` core files** (`pyproject.toml`, `.html`, `.js`, `.rst`) are
   outside our index.
4. Pro issue texts are rewritten in a templated style ("## Title: ...
   ## Description") and read differently from Verified's raw GitHub issues.
   Worth a `source` feature or a breakdown in the analysis.
5. Pro's `before_repo_set_cmd` applies the test patch before agents ran. We
   index the clean pre-fix commit (as the proposal says); the unresolvable
   lines we found came from cases 2 and 3, not from test-patch files.

### Coverage over all 500 tasks

"Resolvable" = core lines in a `.py` file that exists at the pre-fix commit,
within the file's length. "Indexed" = also inside some chunk (so not blank
lines outside definitions). "Recall@300 cap" = mean of min(1, 300 / core
lines): the best any ranker could do at B = 300.

| | Tasks | Median core lines | Core <= 300 lines | Resolvable | Indexed | Recall@300 cap | BM25 seed line recall@300 |
|---|---|---|---|---|---|---|---|
| All | 500 | 1,160 | 12.6% | 99.2% | 92.7% | 39.0% | 11.6% |
| Verified | 410 | 1,362 | 10.0% | 100.0% | 93.6% | 35.7% | 10.8% |
| Pro | 90 | 671 | 24.4% | 92.8% | 86.2% | 54.3% | 15.2% |

Snapshots have a median of about 28k chunks (max 56k, sympy).

## Design decisions

**Snapshots.** Full clones were too slow on the campus link (about 70 KB/s
per GitHub stream). `acl.snapshot` makes one bare repo per repository and
fetches only the needed commits at depth 1 with `--filter=blob:none`, then
fetches only the `.py` blobs those commits reference. All 15 repositories
come to 170 MB (prepared outputs: 551 MB). Snapshots are read straight from
git objects, never checked out. Pitfall: `git cat-file -e` in a partial clone lazily fetches a
missing commit with its whole history of trees; existence checks use
`git rev-list --missing=print` instead.

**Chunks** (`acl.chunking`). Function (every def directly under a module or
class, methods included; nested defs stay inside their function), class
(lines of the class not covered by its methods or nested classes) and module
(lines not covered by any top-level def or class). Every non-blank line is in
exactly one chunk, so chunks never overlap. Blank lines are trimmed from the
ends of each span but kept inside it, so a docstring stays one region: this
matters because SWE-Explore's nDCG discounts by region rank. Chunk ids match
LocAgent node names (`path:Class.method`, `path` for the module chunk);
repeated names (`@overload`, property setters) get `#2`, `#3`.

**BM25** (`acl.bm25`). Okapi BM25, k1 = 1.2, b = 0.75, Lucene idf. Tokens are
lowercased identifiers plus their snake_case and camelCase parts, with the
whole identifier kept too. A chunk's document is its path, qualified name and
text. Scores are min-max normalised per query, as the proposal specifies.

**Budget filling.** Ranked chunks become ranked `(path, start, end)` regions in
rank order; the region that crosses the budget is cut at the budget. This is
the convention of SWE-Explore's `recall_at_budget`, so our output can be fed to
their scorer unchanged.

**Graph** (`acl.graph`). LocAgent's `build_graph.py` v2.3 (Apache-2.0), which
uses Python's `ast` module (the proposal says tree-sitter; the write-up should
say `ast`). Changes: reads from the in-memory snapshot; resolves absolute
imports under `src/` and `lib/` as well as the root (otherwise pytest, flask,
requests and ansible get almost no import edges); keeps each definition's
AST node instead of re-parsing its source segment; caches the file-level half of
`find_all_possible_callee`, which LocAgent recomputes for every definition in
a file. On django that took a snapshot from about 100 to 140 s down to about
21 s, and peak memory fell enough that three workers no longer exhaust the
laptop's RAM (three of the original builds did, at 21:30 on 2026-10-09).
Verified edge-for-edge identical against the original on requests, flask,
scikit-learn, astropy and django snapshots. One trap kept in mind: LocAgent's
segments start at the `def` line, so a definition's own decorators never
contribute edges; the rewrite analyses a copy with them removed to match.
Edge types: `contains`, `imports`, `invokes`, `inherits`, stored directed and mapped onto
chunk ids. LocAgent's fuzzy callee matching links a call to every same-named
candidate, so `invokes` is dense (about 55k edges on astropy).

## Outputs

`data/prepared/<instance_id>/`: `chunks.parquet` (every chunk with spans and
BM25 score), `edges.parquet`, `seed.json` (regions at B = 100/300/500, top-10
seed ids, coverage stats, timings). Chunk text is not stored; re-chunking from
the mirror is fast and deterministic.

## Hand-check (checkpoint)

Ten tasks, one from each of the ten largest repositories, drawn with seed 16
([handcheck_ids.txt](handcheck_ids.txt)): 7 Verified, 3 Pro. For each,
`python -m acl.inspect_task <id>` was read against the issue text and the
ground truth. Checked: chunk spans against the real source lines (decorators,
docstrings, class headers, module code), that BM25's top chunks are plausible
for the issue, that graph edges around the top hit exist in the code, and that
the coverage numbers match the region lists. "Line recall" is a plain overlap
count of the seed at B = 300, not the evaluation harness.

| Task | Seed (BM25 top 10) | Line recall @300 | Note |
|---|---|---|---|
| django-13741 | modified class and method at ranks 1 to 3 | 49/473 | `inherits` edges from tests are local subclasses inside test functions: correct |
| sympy-24066 | modified method at rank 2 | 204/665 | top hit (a test) has an `invokes` edge to it |
| ansible-d62496 (Pro) | modified function at rank 1 | 195/310 | 15 of 16 neighbours of the top hit are core |
| scikit-learn-13135 | modified class's methods at ranks 2, 3, 5 | 233/295 | fuzzy `invokes` links one `fit_transform()` call to every class's `fit_transform` |
| qutebrowser-6dd402 (Pro) | method named in the issue at rank 1 | 116/451 | all 3 neighbours of the top hit are core |
| matplotlib-22865 | target class at rank 7, behind tutorials and examples | 0/1637 | core is the whole of `colorbar.py` |
| openlibrary-9c392 (Pro) | target class's methods at ranks 1 to 4 | 39/255 | a core range `1-240` on a 146-line file: 95 phantom lines |
| xarray-2905 | fails: `print_versions`, `doc/conf.py` | 0/2843 | pasted version dump drives the query |
| pytest-7490 | fails: tests rank above `skipping.py` | 0/331 | |
| sphinx-9658 | fails: `doc/conf.py` | 0/3024 | one `imports` edge from the top hit reaches a core function |

Verdict: chunks, BM25, budget filling and the graph behave as intended; no
pipeline bug was left open. Bugs found and fixed during the check: duplicate
chunk ids from `@overload` and property setters, and blank lines splitting
docstrings into many regions. The failures in the table are genuine BM25
failures, and several (version dumps, test files outranking source) are the
kind of signal Phase A is meant to measure.

Files that fail to parse (78 distinct, in 113 tasks) are deliberate fixtures:
django's `tests_syntax_error.py`, pylint's `syntax_error.py` and
`tokenize_error.py`, Python 2 examples. They are left out of the index.

## Running it

`python -m acl.prepare` is resumable (a task is done once its `seed.json`
exists). Graph building dominates: median per snapshot is 1 to 35 s for most
repositories, 29 s for django and 256 s for sympy (its 2018 to 2022 snapshots
include the auto-generated Rubi integration rules, 26 MB of Python). Peak
memory is under 900 MB per worker; 3 workers under a 4 GB cap
(`systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 ...`) ran the
full set on the laptop. The pool is a `ProcessPoolExecutor` because
`multiprocessing.Pool` hangs forever when a worker is killed for memory,
which cost a night on 2026-10-09.

## Open questions for Stage 2

Resolved later: edge direction was tuned on dev per edge group (Stage 3,
"both" for calls and imports); the harness reports both the official and the
clipped scorer (Stage 2); environment dumps stay in the query and became the
`has_env_dump` router feature (Stage 3).

- Expansion direction: follow edges outgoing only (callees, bases, imported
  names) or both directions (also callers, subclasses, importers)?
- Scorer: official SWE-Explore lines vs clipped (see above).
- Whether to strip pasted environment dumps (`pip list`, `<details>` blocks)
  from the BM25 query. They pull the seed toward `conf.py` and
  `print_versions`; arguably that is exactly the signal the router should see.
