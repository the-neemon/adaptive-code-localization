# Tier 2 router (classifier) on held-out repositories

### nDCG@100 (444 held-out issues)

| system | mean |
|---|---|
| stop | 0.2724 |
| fixed | 0.2858 |
| router | 0.2780 |
| router_1 | 0.2856 |
| tier2 | 0.2727 |
| oracle | 0.3965 |

- router: gain vs fixed -0.0078 [-0.0245, +0.0159]; GapClosed -0.070 [-0.223, +0.121]; helps 17%, hurts 17%; routes 50% away from the default
- router_1: gain vs fixed -0.0001 [-0.0068, +0.0094]; GapClosed -0.001 [-0.063, +0.074]; helps 9%, hurts 7%; routes 25% away from the default
- tier2: gain vs fixed -0.0130 [-0.0253, +0.0068]; GapClosed -0.118 [-0.237, +0.052]; helps 29%, hurts 28%; routes 89% away from the default
- tier2 vs router_1: -0.0129 [-0.0303, +0.0086]
- tier2 picks the oracle's action on 25% of issues; actions: stop 47%, expand_imports 13%, expand_calls 11%, dense 15%, fusion 14%
- settings chosen on dev: ansible: gbt max_depth=2, n_estimators=50; django: gbt max_depth=3, n_estimators=50; matplotlib: gbt max_depth=3, n_estimators=100; qutebrowser: gbt max_depth=1, n_estimators=100; scikit-learn: gbt max_depth=1, n_estimators=50; small: gbt max_depth=1, n_estimators=50; sphinx: gbt max_depth=2, n_estimators=100; sympy: gbt max_depth=1, n_estimators=100

### nDCG@300 (444 held-out issues)

| system | mean |
|---|---|
| stop | 0.1786 |
| fixed | 0.1872 |
| router | 0.1908 |
| router_1 | 0.1927 |
| tier2 | 0.1798 |
| oracle | 0.2618 |

- router: gain vs fixed +0.0035 [-0.0041, +0.0086]; GapClosed +0.047 [-0.047, +0.114]; helps 17%, hurts 18%; routes 37% away from the default
- router_1: gain vs fixed +0.0055 [-0.0005, +0.0077]; GapClosed +0.073 [-0.005, +0.107]; helps 7%, hurts 6%; routes 15% away from the default
- tier2: gain vs fixed -0.0074 [-0.0255, +0.0006]; GapClosed -0.099 [-0.303, +0.009]; helps 32%, hurts 39%; routes 78% away from the default
- tier2 vs router_1: -0.0129 [-0.0261, -0.0058]
- tier2 picks the oracle's action on 24% of issues; actions: stop 15%, expand_inherit 0%, expand_imports 20%, expand_calls 19%, dense 23%, fusion 22%
- settings chosen on dev: ansible: gbt max_depth=2, n_estimators=50; django: gbt max_depth=3, n_estimators=100; matplotlib: gbt max_depth=3, n_estimators=100; qutebrowser: gbt max_depth=3, n_estimators=50; scikit-learn: gbt max_depth=3, n_estimators=100; small: gbt max_depth=1, n_estimators=100; sphinx: gbt max_depth=3, n_estimators=50; sympy: gbt max_depth=2, n_estimators=50

### recall@300 (444 held-out issues)

| system | mean |
|---|---|
| stop | 0.1219 |
| fixed | 0.1368 |
| router | 0.1360 |
| router_1 | 0.1391 |
| oracle | 0.1878 |

- router: gain vs fixed -0.0008 [-0.0075, +0.0034], vs stop +0.0141; GapClosed -0.016 [-0.130, +0.071]; helps 16%, hurts 18% of issues vs fixed; routes 37% away from the default

- router_1: gain vs fixed +0.0023 [+0.0002, +0.0064], vs stop +0.0173; GapClosed +0.045 [+0.003, +0.131]; helps 7%, hurts 6% of issues vs fixed; routes 15% away from the default

- tier2: mean 0.1332, gain vs fixed -0.0036 [-0.0132, +0.0023], GapClosed -0.071

### file_recall_patch@300 (SWE-bench Verified only) (374 held-out issues)

| system | mean |
|---|---|
| stop | 0.6339 |
| fixed | 0.7149 |
| router | 0.7023 |
| router_1 | 0.7128 |
| oracle | 0.8222 |

- router: gain vs fixed -0.0125 [-0.0360, -0.0036], vs stop +0.0684; GapClosed -0.117 [-0.408, -0.029]; helps 2%, hurts 3% of issues vs fixed; routes 38% away from the default

- router_1: gain vs fixed -0.0021 [-0.0053, +0.0048], vs stop +0.0789; GapClosed -0.019 [-0.049, +0.051]; helps 1%, hurts 1% of issues vs fixed; routes 16% away from the default

- tier2: mean 0.6605, gain vs fixed -0.0544 [-0.1025, -0.0430], GapClosed -0.507

### nDCG@500 (444 held-out issues)

| system | mean |
|---|---|
| stop | 0.1485 |
| fixed | 0.1585 |
| router | 0.1554 |
| router_1 | 0.1595 |
| tier2 | 0.1482 |
| oracle | 0.2142 |

- router: gain vs fixed -0.0031 [-0.0191, +0.0046]; GapClosed -0.056 [-0.312, +0.081]; helps 20%, hurts 26%; routes 49% away from the default
- router_1: gain vs fixed +0.0010 [-0.0099, +0.0046]; GapClosed +0.018 [-0.157, +0.089]; helps 7%, hurts 7%; routes 16% away from the default
- tier2: gain vs fixed -0.0103 [-0.0264, -0.0022]; GapClosed -0.185 [-0.404, -0.044]; helps 20%, hurts 30%; routes 52% away from the default
- tier2 vs router_1: -0.0113 [-0.0283, +0.0014]
- tier2 picks the oracle's action on 22% of issues; actions: stop 9%, expand_imports 14%, expand_calls 12%, dense 16%, fusion 48%
- settings chosen on dev: ansible: logreg C=1; django: logreg C=0.01; matplotlib: logreg C=0.01; qutebrowser: logreg C=1; scikit-learn: logreg C=10; small: gbt max_depth=2, n_estimators=100; sphinx: logreg C=0.01; sympy: logreg C=0.1
