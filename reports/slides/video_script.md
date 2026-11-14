# Video script: When Does Graph Expansion Help?

Naman and Yash, about 10 minutes (about 8.5 minutes of speech plus slide
changes), one pass through `slides.pdf` (17 slides). Naman takes the
introduction, Phase A and the conclusion. Yash takes the setup, Phase B and the
close. Times are approximate (about 140 words a minute). **[Next slide]** marks
a slide change.

---

## Slide 1: Title (Naman, 0:20)

Hi, I am Naman. With me is Yash. Our team, with Aaditya Bhatia, asked a
simple question about coding agents: when does it help to follow the links in
a codebase, like function calls and imports, to find the code that fixes a bug?

**[Next slide]**

## Slide 2: The problem (Naman, 0:35)

Before an agent can fix an issue, it has to find the code that matters, within
a limited context window. Every piece of code it reads pushes another out.

Keyword or embedding search is cheap, but it misses code that is only connected
to the issue through calls, imports or inheritance. Agents like LocAgent walk
the code graph to find it. But the decision of when to expand is made inside the
language model's reasoning. Nobody measures it directly.

**[Next slide]**

## Slide 3: Our question (Naman, 0:35)

So we asked: when does graph expansion help? And can we predict it in advance?

We expected it to depend on the issue. A traceback should make following calls
useful. An issue that names the file should not need expansion. A hub as the
top hit could flood the budget.

Phase A measures what each action is worth, issue by issue. Phase B tries to
turn that into a router. Over to Yash for the setup.

**[Next slide]**

## Slide 4: Overview (Yash, 0:35)

Thanks. For every issue we first run BM25 over the code. That is our seed.
Right after it we compute a few cheap features. A router uses them to pick one
of six actions. The result is cut to the same line budget.

In Phase A we skip the router and run all six actions on every issue. In
Phase B the router has to choose in advance.

**[Next slide]**

## Slide 5: Data, budget and splits (Yash, 0:40)

We use the SWE-Explore benchmark, whose ground truth is the code successful
agents actually read. The Python part has 500 issues from 15 repositories.
Django alone is 41 percent of them.

Every system gets 300 lines. The ground truth is mostly whole files, so even a
perfect ranker reaches only about 40 percent recall. That is why the scores look
low. 56 issues are kept for tuning. The router is always tested on
repositories it has never seen.

**[Next slide]**

## Slide 6: Six retrieval actions (Yash, 0:40)

Stop keeps the BM25 seed. Dense uses a frozen code embedding model. Fusion
merges the two rankings.

The other three expand along calls, imports or inheritance: the graph
neighbours of the top ten hits move up, just below their parent. On the tuning
set, following links in both directions worked best. The graph comes from
LocAgent's builder, which we made about five times faster.

**[Next slide]**

## Slide 7: Scoring problems (Yash, 0:40)

We also found two problems in the benchmark. First, agents often printed line
ranges past the end of a file. The official recall counts those missing
lines. We clip them.

Second, the released nDCG code only compares a system to the best ordering of
its own output, so one early hit can score perfectly. 75 of the 500 BM25 seeds
get a perfect 1.0 while covering only 14 percent of the right code. We use the
paper's definition and report both. Back to Naman for the results.

**[Next slide]**

## Slide 8: Phase A averages (Naman, 0:35)

On average, very little changes. At 300 lines only fusion reliably beats the
BM25 seed, by only about 0.01. Dense on its own is reliably worse. Graph
expansion barely moves the average.

But look at the last row. If we could pick the best action for each issue, the
score would jump from 0.18 to 0.26.

**[Next slide]**

## Slide 9: Per issue (Naman, 0:35)

This is why. Graph expansion helps about 40 percent of issues and hurts about
40 percent. The average hides both.

Picking the best action per issue would be 40 percent better than fusion, the
best fixed action. That best action is spread evenly across five of the six
actions, which is the case for a router.

**[Next slide]**

## Slide 10: Which signals predict the gain? (Naman, 0:35)

So which signals predict the gain? Four of the five expectations from our
proposal did not hold. A traceback does not make call expansion help.
Naming the file does not make expansion useless.

The hub idea holds only as a contrast. Across all our features there is about
four times more signal than chance, so there is something to learn, but it is
weak.

**[Next slide]**

## Slide 11: Strongest signal (Naman, 0:35)

The strongest signal was one we did not predict: how many graph links the top
BM25 hit has. When it is a hub, dense loses 0.06 and fusion loses too. When it
is weakly connected, fusion gains 0.04.

Our reading is that BM25 has landed on a central definition. Then any
change tends to hurt. Yash will show whether a router can use this.

**[Next slide]**

## Slide 12: Tier 1 rules (Yash, 0:40)

Our first router is a short list of if-then rules, picked by a fixed procedure
that we froze before seeing any held-out result. It starts from fusion and adds
up to five rules, each needing a reliable gain on at least 20 issues.

One rule came up in all eight folds. If the top hit has more than seven call
links and BM25 and dense mostly disagree, keep the seed and expand along calls.
On those 64 issues dense is wrong and fusion inherits its mistakes.

**[Next slide]**

## Slide 13: Held-out results (Yash, 0:35)

On repositories it never saw, the full rule list beats fusion by 0.004 and
closes about 5 percent of the gap to the per-issue best. The single stable rule
does a little better, about 7 percent.

But with only 15 repositories the uncertainty is wider than the gain. So
neither router clearly beats fusion.

**[Next slide]**

## Slide 14: Tier 2 and SWE-bench Verified (Yash, 0:35)

We also tried a small classifier instead of rules. It overfit the 56 tuning
issues and did reliably worse than the single rule.

On SWE-bench Verified, which checks whether the developer's edited files come
back, fusion stays best, because graph expansion covers fewer distinct files.
Naman will wrap up.

**[Next slide]**

## Slide 15: What we learned (Naman, 0:35)

To sum up: the right action really does depend on the issue. Picking per
issue would gain 40 percent. But cheap signals predict little of that. One rule
transfers to new repositories and closes about 7 percent of the gap, while a
learned classifier does worse. We also found that the benchmark's released
nDCG rewards a single early hit.

This is the partly negative result our proposal allowed for.

**[Next slide]**

## Slide 16: Limitations (Naman, 0:30)

The main limitation is size: 15 repositories, one of them 41 percent of the
data, so every interval is wide. The ground truth is what agents read, which
may favour broad methods like fusion. And we used only Python, one dense model
and one hop of expansion.

**[Next slide]**

## Slide 17: Thank you (Yash, 0:10)

Thank you for watching. The code and the final report are in our repository.
