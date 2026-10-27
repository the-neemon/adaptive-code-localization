# Related work (draft, Stage 2)

Superseded by the Related Work section of the final report
(reports/final/main.tex), whose references were checked against primary
sources in Stage 6. Kept as the Stage 2 record.

Draft of the report's Related Work section. It extends the proposal's section
with the evaluation benchmark details we now rely on, the dense retriever we
chose, and a strand the proposal did not cover: selective query expansion and
query performance prediction in classic IR, the closest precedent for deciding
per query whether to expand. Citations marked (proposal) are carried over
from the proposal and should be re-checked against the final versions before
submission.

## Graph-based and agentic localization

Issue localization finds the code a developer or agent must read to resolve a
natural-language issue. Recent systems treat the repository as a graph and let
an LLM explore it. LocAgent (Chen et al., 2025) builds a heterogeneous graph
of directories, files, classes and functions with containment, import,
invocation and inheritance edges, and gives an agent tools to search and
traverse it. CoSIL (Jiang et al., 2025) searches call and module graphs
without an index, pruning as the LLM goes. OrcaLoca (Yu et al., 2025) adds
priority scheduling of search actions and distance-aware pruning.
AutoCodeRover (Zhang et al., 2024) combines LLM reasoning with AST-level code
search and optional spectrum-based fault localization, and Agentless (Xia et
al., 2024) narrows hierarchically from files to elements to lines.
RepoGraph (Ouyang et al., 2025) and CodexGraph (Liu et al., 2024) provide
repository-graph construction and querying as infrastructure for such agents.

In all of these, the decision to follow an edge, and when to stop, is made
turn by turn inside the model's reasoning. We reuse LocAgent's graph builder
(adapted; it uses Python's `ast` module) but take the decision out of the
model: we measure, per issue, what expanding along one edge type is worth, and
then make the decision with an explicit rule before any expansion happens.

## Retrieval for code

Classic IR-based bug localization ranks source files against bug reports with
lexical similarity (BugLocator; Zhou et al., 2012), and BM25 (Robertson and
Zaragoza, 2009) remains the standard lexical baseline. Semantic code search
(CodeSearchNet; Husain et al., 2019) moved to dense bi-encoders that embed
natural language and code into one vector space. We use CodeRankEmbed
(Suresh et al., 2025), a 137M-parameter bi-encoder contrastively trained on
the CoRNStack corpus of 21 million (query, code) pairs; it is small enough to
embed every chunk of every snapshot on one consumer GPU. Retrievers and
rerankers trained specifically for issue localization, such as SWERank
(Reddy et al., 2026) and CoRet (Fehr et al., 2025) (proposal), are
complementary: we only use frozen, off-the-shelf models, so any trained
retriever could be added as one more action. To combine lexical and dense
rankings we use reciprocal-rank fusion (Cormack et al., 2009), which needs no
score calibration between the two retrievers.

## Selective expansion and query performance prediction

The question "should this query be expanded?" is old in IR. Pseudo-relevance
feedback expands a query with terms from its top results; it helps on average
but hurts a sizeable share of queries, so Amati et al. (2004) and
Cronen-Townsend et al. (2004) proposed applying it selectively, predicting
from the query and its first-pass results whether expansion will help.
Those predictors grew into query performance prediction (QPP): clarity
(Cronen-Townsend et al., 2002) compares the language model of the top results
with the collection, and NQC (Shtok et al., 2012) uses the spread of the top
retrieval scores. Our post-seed features (BM25 score concentration, number of
distinct files in the seed, agreement between BM25 and dense results) are
post-retrieval QPP signals of this kind. Our setting differs in three ways:
the expansion is structural (along call, import or inheritance edges) rather
than lexical, the candidate actions are typed, so the router picks among
several mechanisms rather than making a yes/no call, and the evaluation holds
a line budget fixed, so every addition displaces something.

## Adaptive routing

Adaptive-RAG (Jeong et al., 2024) routes questions to no retrieval, single-step
or multi-step retrieval using a classifier of question complexity. In code,
AIRCoder (Shi et al., 2026) (proposal) adaptively combines retrieval
dimensions for repository-level completion, where the location to complete is
already known. SWE-Router (Son et al., 2026) (proposal) routes a partial agent
trajectory between weak and strong models rather than among localization
mechanisms. Repository Memory (Wang et al., 2026) (proposal) adds
commit-history retrieval tools to LocAgent but leaves their use to the agent.
To our knowledge no prior work routes among localization mechanisms per issue
with an explicit, inspectable policy.

## Evaluation of exploration and localization

SWE-bench (Jimenez et al., 2024) and its human-validated subset SWE-bench
Verified (OpenAI, 2024) score end-to-end issue resolution; localization is
usually evaluated against the files and functions the developer's patch
edits. This misses code that must be read but is not edited. ContextBench (Li
et al., 2026) scores context retrieval against human-annotated gold contexts.
SWE-Explore (Zhang et al., 2026), our primary benchmark, derives line-level
"core" regions (read by every successful trajectory) and "optional" regions
(read by some) from independent successful agent runs, and scores ranked
region lists under a line budget with recall, nDCG (Järvelin and
Kekäläinen, 2002) adapted to line budgets, first useful hit and noise rates.
Two properties of its released scorer matter for us (docs/stage2.md): its
nDCG normalises against the best reordering of the regions a system returned,
so it does not reward finding more of the core, and it keeps line ranges that
run past the end of a file in the recall denominator. We therefore report the
official numbers for comparability and use the paper's ground-truth-based
nDCG, on clipped regions, as our primary gain. SWE-Explore also reports that
one-shot sparse and lightweight dense retrievers score close to random on its
metrics; our study asks a narrower question that this does not answer, namely
when a cheap structural step on top of such a retriever recovers what it
misses.

## References

- Amati, G., Carpineto, C., and Romano, G. 2004. Query difficulty, robustness, and selective application of query expansion. ECIR.
- Chen, Z. et al. 2025. LocAgent: Graph-guided LLM agents for code localization. ACL.
- Cormack, G. V., Clarke, C. L. A., and Buettcher, S. 2009. Reciprocal rank fusion outperforms Condorcet and individual rank learning methods. SIGIR.
- Cronen-Townsend, S., Zhou, Y., and Croft, W. B. 2002. Predicting query performance. SIGIR.
- Cronen-Townsend, S., Zhou, Y., and Croft, W. B. 2004. A framework for selective query expansion. CIKM.
- Fehr, F. et al. 2025. CoRet: Improved retriever for code editing. ACL (short). (proposal)
- Husain, H., Wu, H.-H., Gazit, T., Allamanis, M., and Brockschmidt, M. 2019. CodeSearchNet challenge: Evaluating the state of semantic code search. arXiv:1909.09436.
- Järvelin, K. and Kekäläinen, J. 2002. Cumulated gain-based evaluation of IR techniques. ACM TOIS.
- Jeong, S., Baek, J., Cho, S., Hwang, S. J., and Park, J. C. 2024. Adaptive-RAG: Learning to adapt retrieval-augmented large language models through question complexity. NAACL.
- Jiang, Z. et al. 2025. CoSIL: Software issue localization via LLM-driven code repository graph searching. arXiv:2503.22424.
- Jimenez, C. E. et al. 2024. SWE-bench: Can language models resolve real-world GitHub issues? ICLR.
- Li, H. et al. 2026. ContextBench: A benchmark for context retrieval in coding agents. CoRR.
- Liu, X. et al. 2024. CodexGraph: Bridging large language models and code repositories via code graph databases. arXiv:2408.03910.
- OpenAI. 2024. Introducing SWE-bench Verified.
- Ouyang, S. et al. 2025. RepoGraph: Enhancing AI software engineering with repository-level code graph. ICLR.
- Reddy, R. G. et al. 2026. SWERank: Software issue localization with code ranking. ICLR. (proposal)
- Robertson, S. and Zaragoza, H. 2009. The probabilistic relevance framework: BM25 and beyond. Foundations and Trends in IR.
- Shi, C., Gao, M., and Gao, Z. 2026. AIRCoder: Adaptive integration of multi-dimensional retrieval for repository-level code completion. ACL. (proposal)
- Shtok, A., Kurland, O., Carmel, D., Raiber, F., and Markovits, G. 2012. Predicting query performance by query-drift estimation. ACM TOIS.
- Son, S. et al. 2026. SWE-Router: Routing in multi-turn agentic software engineering tasks. DL4C Workshop at ICML. (proposal)
- Suresh, T., Reddy, R. G., Xu, Y., Nussbaum, Z., Mulyar, A., Duderstadt, B., and Ji, H. 2025. CoRNStack: High-quality contrastive data for better code retrieval and reranking. arXiv:2412.01007.
- Wang, B. et al. 2026. Improving code localization with repository memory. ICLR. (proposal)
- Xia, C. S., Deng, Y., Dunn, S., and Zhang, L. 2024. Agentless: Demystifying LLM-based software engineering agents. arXiv:2407.01489.
- Yu, Z. et al. 2025. OrcaLoca: An LLM agent framework for software issue localization. ICML.
- Zhang, S. et al. 2026. SWE-Explore: Benchmarking how coding agents explore repositories. arXiv:2606.07297.
- Zhang, Y., Ruan, H., Fan, Z., and Roychoudhury, A. 2024. AutoCodeRover: Autonomous program improvement. arXiv:2404.05427.
- Zhou, J., Zhang, H., and Lo, D. 2012. Where should the bugs be fixed? More accurate information retrieval-based bug localization based on bug reports. ICSE.
