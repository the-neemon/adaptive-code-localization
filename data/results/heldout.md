# Stage 5: Tier 1 router on held-out repositories

### nDCG@100 (444 held-out issues)

| system | mean |
|---|---|
| stop | 0.2724 |
| fixed | 0.2858 |
| router | 0.2780 |
| router_1 | 0.2856 |
| oracle | 0.3965 |

- router: gain vs fixed -0.0078 [-0.0245, +0.0159], vs stop +0.0056; GapClosed -0.070 [-0.223, +0.121]; helps 17%, hurts 17% of issues vs fixed; routes 50% away from the default

- router_1: gain vs fixed -0.0001 [-0.0068, +0.0094], vs stop +0.0132; GapClosed -0.001 [-0.063, +0.074]; helps 9%, hurts 7% of issues vs fixed; routes 25% away from the default

### nDCG@300 (444 held-out issues)

| system | mean |
|---|---|
| stop | 0.1786 |
| fixed | 0.1872 |
| router | 0.1908 |
| router_1 | 0.1927 |
| oracle | 0.2618 |

- router: gain vs fixed +0.0035 [-0.0041, +0.0086], vs stop +0.0121; GapClosed +0.047 [-0.047, +0.114]; helps 17%, hurts 18% of issues vs fixed; routes 37% away from the default

- router_1: gain vs fixed +0.0055 [-0.0005, +0.0077], vs stop +0.0141; GapClosed +0.073 [-0.005, +0.107]; helps 7%, hurts 6% of issues vs fixed; routes 15% away from the default

### By held-out fold (nDCG@300)

| group | n | stop | fixed | router | router_1 | oracle | GapClosed |
|---|---|---|---|---|---|---|---|
| ansible | 40 | 0.2857 | 0.3067 | 0.3218 | 0.3121 | 0.3996 | +0.16 |
| django | 207 | 0.1107 | 0.1161 | 0.1229 | 0.1233 | 0.1771 | +0.11 |
| matplotlib | 29 | 0.1422 | 0.2003 | 0.1851 | 0.1886 | 0.2960 | -0.16 |
| qutebrowser | 30 | 0.1796 | 0.2118 | 0.2079 | 0.2158 | 0.2703 | -0.07 |
| scikit-learn | 31 | 0.3925 | 0.3886 | 0.3876 | 0.3926 | 0.4959 | -0.01 |
| small | 29 | 0.2048 | 0.1821 | 0.1947 | 0.1859 | 0.2666 | +0.15 |
| sphinx | 12 | 0.1333 | 0.1504 | 0.1521 | 0.1504 | 0.1912 | +0.04 |
| sympy | 66 | 0.2387 | 0.2355 | 0.2319 | 0.2460 | 0.3257 | -0.04 |

### By whether the issue names a target file (nDCG@300)

| group | n | stop | fixed | router | router_1 | oracle | GapClosed |
|---|---|---|---|---|---|---|---|
| names a repo file | 81 | 0.1927 | 0.1943 | 0.2058 | 0.2123 | 0.2826 | +0.13 |
| does not | 363 | 0.1755 | 0.1857 | 0.1874 | 0.1883 | 0.2572 | +0.02 |

### Cost per issue (B = 300)

| system | extra retrieval calls | graph nodes visited | ms |
|---|---|---|---|
| stop | 0.00 | 0.0 | 4.1 |
| fixed | 1.00 | 0.0 | 7.5 |
| router | 0.63 | 30.9 | 10.9 |
| router_1 | 0.85 | 19.5 | 9.6 |

### Rules across the 8 per-fold routers

| rule | action | folds |
|---|---|---|
| top_degree_calls > 7 AND bm25_dense_overlap <= 0.20 | expand_calls | 7 |
| seed_files <= 5 AND 1 < top_degree_calls <= 7 | expand_calls | 4 |
| not is_pro AND top_degree_inherit > 0 | expand_imports | 3 |
| is_pro AND seed_files <= 5 | expand_imports | 2 |
| names_py_path AND 4 < top_degree <= 10 | expand_imports | 2 |
| seed_files <= 5 AND top_degree > 10 | expand_inherit | 2 |
| bm25_topk_margin <= 0.15 AND topk_degree_mean <= 6.90 | expand_calls | 1 |
| 0.04 < bm25_margin <= 0.10 AND top_degree > 9 | expand_inherit | 1 |
| names_py_path AND 4 < top_degree <= 11 | expand_calls | 1 |
| 0.04 < bm25_margin <= 0.11 AND top_degree > 11 | stop | 1 |
| issue_tokens > 178 AND top_degree_calls > 7 | stop | 1 |
| bm25_topk_margin > 0.25 AND bm25_dense_overlap > 0.40 | expand_calls | 1 |
| top_degree_calls > 7 AND top_degree_imports <= 1 | dense | 1 |
| seed_files <= 5 AND top_degree_calls > 7 | expand_inherit | 1 |
| issue_tokens <= 89 AND seed_files <= 5 | expand_calls | 1 |
| bm25_topk_margin <= 0.15 AND topk_degree_mean <= 6.80 | expand_calls | 1 |
| 0.15 < bm25_topk_margin <= 0.25 AND top_degree_imports > 1 | expand_imports | 1 |
| bm25_topk_margin <= 0.15 AND topk_degree_mean <= 7.00 | expand_calls | 1 |
| bm25_topk_margin <= 0.15 AND 0.20 < bm25_dense_overlap <= 0.40 | expand_calls | 1 |
| top_degree_calls > 6 AND bm25_dense_overlap <= 0.20 | expand_calls | 1 |
| 0.04 < bm25_margin <= 0.10 AND top_degree > 9 | expand_imports | 1 |
| 2 < n_known_identifiers <= 6 AND bm25_margin <= 0.04 | expand_imports | 1 |
| has_traceback AND n_known_identifiers > 6 | expand_calls | 1 |

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

### file_recall_patch@300, extra Verified issues, router fit on all 500 study issues (90 issues)

| system | mean |
|---|---|
| stop | 0.6059 |
| fixed | 0.6800 |
| router | 0.6378 |
| router_1 | 0.6522 |
| oracle | 0.7822 |

- router: gain vs fixed -0.0422 [-0.1081, +0.0417], vs stop +0.0319; GapClosed -0.413 [-2.323, +0.194]; helps 3%, hurts 8% of issues vs fixed; routes 44% away from the default

- router_1: gain vs fixed -0.0278 [-0.0932, +0.0179], vs stop +0.0463; GapClosed -0.272 [-1.667, +0.090]; helps 1%, hurts 4% of issues vs fixed; routes 14% away from the default

### SWE-bench Verified, file recall@300 of the developer patch (464 issues: 374 held-out study issues + 90 extra)

| system | mean |
|---|---|
| stop | 0.6285 |
| fixed | 0.7081 |
| router | 0.6898 |
| router_1 | 0.7011 |
| fixed: stop | 0.6285 |
| fixed: expand_inherit | 0.6371 |
| fixed: expand_imports | 0.5821 |
| fixed: expand_calls | 0.6386 |
| fixed: dense | 0.6793 |
| fixed: fusion | 0.7081 |

(dev issues are excluded: 36 Verified issues tune the actions and are not held out)

### nDCG@500 (444 held-out issues)

| system | mean |
|---|---|
| stop | 0.1485 |
| fixed | 0.1585 |
| router | 0.1554 |
| router_1 | 0.1595 |
| oracle | 0.2142 |

- router: gain vs fixed -0.0031 [-0.0191, +0.0046], vs stop +0.0069; GapClosed -0.056 [-0.312, +0.081]; helps 20%, hurts 26% of issues vs fixed; routes 49% away from the default

- router_1: gain vs fixed +0.0010 [-0.0099, +0.0046], vs stop +0.0110; GapClosed +0.018 [-0.157, +0.089]; helps 7%, hurts 7% of issues vs fixed; routes 16% away from the default
