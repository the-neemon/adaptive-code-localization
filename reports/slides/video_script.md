# Video script: When Does Graph Expansion Help?

Two speakers, about 11 to 12 minutes, one pass through `slides.pdf` (17 slides).
Speaker 1 takes the introduction, Phase A and the conclusion. Speaker 2 takes
the setup, Phase B and the close. Times are approximate (about 140 words a
minute). **[Next slide]** marks a slide change.

---

## Slide 1: Title (Speaker 1, 0:20)

Hi, we are Naman Singhal, Yash More and Aaditya Bhatia. Our project asks a
simple question about coding agents: when does it help to follow the links
in a codebase, like function calls and imports, to find the code that fixes
a bug?

**[Next slide]**

## Slide 2: The problem (Speaker 1, 0:45)

Before an agent can fix an issue, it has to find the code that matters. It
works under a budget: its context window is limited, so every piece of code
it reads pushes another piece out.

Keyword search like BM25, or embedding search, is cheap. But it misses code
that shares no words with the issue and is only connected to it through
calls, imports or inheritance.

Agents like LocAgent, CoSIL and OrcaLoca walk the code graph to find that
code. But the decision of when to expand is made inside the language model's reasoning. Nobody measures it directly.

**[Next slide]**

## Slide 3: Our question (Speaker 1, 0:45)

So we asked: when does graph expansion help? And can we predict it before we do it?

We expected the answer to depend on the issue. If the issue has a traceback,
following calls from the named function should help. If the issue already
names the file, keyword search finds it and expansion just adds noise. And if
the top hit is a hub with many links, expanding could flood the budget.

We split the work in two. Phase A measures what each retrieval action is
worth, issue by issue. Phase B tries to turn that into a router that picks
one action per issue. Over to my teammate for the setup.

**[Next slide]**

## Slide 4: Overview (Speaker 2, 0:40)

Thanks. Here is the pipeline. For every issue we first run BM25 over the code.
That is our seed. Right after the seed we compute a few cheap features. A router uses them to pick one of six actions. Whatever the action returns is cut
to the same line budget.

In Phase A we skip the router and run all six actions on every issue, so we
can see how much each one gains over the seed. In Phase B the router has to
choose in advance.

**[Next slide]**

## Slide 5: Data, budget and splits (Speaker 2, 0:50)

We use the SWE-Explore benchmark. Its ground truth is the code that successful
agents actually read while fixing each issue. The Python part has 500 issues from 15 repositories. Django alone is 41 percent of them.

Every system returns ranked code cut to 300 lines. The ground truth is mostly whole files, about 1,160 lines for the median
issue. So even a perfect ranker can only reach about 40 percent recall at 300
lines. That is why all the scores you will see look low.

For a fair test, 56 issues from three repositories are used only for tuning.
The other 444 are split so that every test repository is one the router has
never seen.

**[Next slide]**

## Slide 6: Six retrieval actions (Speaker 2, 0:50)

The six actions. Stop keeps the BM25 seed. Dense uses a frozen code embedding
model. Fusion merges the BM25 and dense rankings.

The other three are graph expansion along calls, imports or inheritance. We
take the top ten seed hits and move their graph neighbours up the ranking,
just below their parent. On the tuning set, following links in both
directions worked best, so a strong hit's callers and callees go right after
it.

The code graph comes from LocAgent's open-source builder. We adapted it to
run about five times faster and checked that it gives the same edges.

**[Next slide]**

## Slide 7: Scoring problems (Speaker 2, 0:50)

While building the evaluation we found two problems in the benchmark itself.

First, agents often printed a range like lines 1 to 240 of a file that was
shorter than that. The benchmark keeps the requested range, so its recall
counts lines that do not exist. We clip them.

Second, the released code computes nDCG differently from the paper. It
compares a system only to the best ordering of its own output, so one early
hit can give a perfect score. 75 of the 500 BM25 seeds score a perfect 1.0
while covering only 14 percent of the right code. We use the paper's definition, which tracks recall better. We report both. Back to my teammate
for the results.

**[Next slide]**

## Slide 8: Phase A averages (Speaker 1, 0:45)

On average, very little changes. At 300 lines only fusion reliably beats the BM25 seed, by only about 0.01. Dense on its own is reliably worse. Graph
expansion barely moves the average, although call expansion is the best single
action at the tightest budget of 100 lines.

But look at the last row. If we could pick the best action for each issue,
the score would jump from 0.18 to 0.26.

**[Next slide]**

## Slide 9: Per issue (Speaker 1, 0:45)

This is why. Graph expansion helps about 40 percent of issues and hurts about
40 percent. The average hides both.

Picking the best action per issue would be 40 percent better than fusion,
the best fixed action. And that best action is spread evenly across five of
the six actions. No action wins everywhere, which is the case for a router.

**[Next slide]**

## Slide 10: Which signals predict the gain? (Speaker 1, 0:45)

So which signals predict the gain? We checked the five expectations from our
proposal. Four did not hold. A traceback does not make call expansion help.
Naming the file does not make expansion useless. BM25 and dense disagreeing
does not mean fusion wins.

The hub idea holds only as a contrast between hubs and weakly connected hits.
Across every feature and pair of features, there is about four times more
signal than chance would give. So there is something to learn, but it is weak.

**[Next slide]**

## Slide 11: Strongest signal (Speaker 1, 0:40)

The strongest signal was one we did not predict: how many graph links the
top BM25 hit has. When the top hit is a hub, dense loses 0.06 and fusion
loses too. When it is weakly connected, fusion gains 0.04.

Our reading is that when the issue names the right code, BM25 lands on a central definition. Then any change tends to hurt. My teammate will
show whether a router can use this.

**[Next slide]**

## Slide 12: Tier 1 rules (Speaker 2, 0:50)

Our first router is a short list of if-then rules. We did not write them by
hand. A fixed procedure picks them: start from the best single action, which
was fusion, then add up to five rules. Each rule must cover at least 20 issues
from three repositories and show a reliable gain. We fixed this procedure
before looking at any held-out result.

One rule came up in all eight folds. If the top hit has more than seven call
links and BM25 and dense mostly disagree, keep the seed and expand along
calls. On those 64 issues, dense is wrong and fusion inherits its mistakes.

**[Next slide]**

## Slide 13: Held-out results (Speaker 2, 0:45)

Here is how the routers do on repositories they never saw. The full rule list
beats fusion by 0.004 and closes about 5 percent of the gap to the per-issue
best. The single stable rule does a little better, about 7 percent.

But neither gain is reliable. With only 15 repositories, the uncertainty is
wider than the gain. So neither router clearly beats fusion.

**[Next slide]**

## Slide 14: Tier 2 and SWE-bench Verified (Speaker 2, 0:45)

We also tried a small classifier instead of rules. It overfit the 56 tuning
issues, moved most issues away from fusion. It did reliably worse than the single rule.

Finally, on SWE-bench Verified, which checks whether the files the developer
edited are returned, fusion stays best. The routers lose a little there,
because graph expansion covers fewer distinct files. My teammate will wrap up.

**[Next slide]**

## Slide 15: What we learned (Speaker 1, 0:50)

To sum up. The right action really does depend on the issue: expansion helps and hurts about equally often. Picking per issue would gain 40 percent.

But cheap signals predict little of that. The strongest is how connected the top hit is. Four of our five expectations did not hold. One rule
transfers to new repositories and closes about 7 percent of the gap, while a
learned classifier does worse.

We also found that the benchmark's released nDCG rewards a single early hit.
Overall this is the partly negative result our proposal allowed for: the measurements stand but the router gain is small.

**[Next slide]**

## Slide 16: Limitations (Speaker 1, 0:40)

A few limitations. We have only 15 repositories, one of them 41 percent of
the data, so every interval is wide. The ground truth is what agents read,
often whole files, which may favour broad methods like fusion. We only used
Python, one dense model and one hop of expansion. And our classifier was
designed after we had seen the rule-based results.

**[Next slide]**

## Slide 17: Thank you (Speaker 2, 0:10)

Thank you for watching. The code, the data pipeline and the final report are
all in our repository.
