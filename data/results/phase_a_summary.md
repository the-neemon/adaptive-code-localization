## Phase A, ndcg@300, 500 issues

| action | mean | gain vs stop | 95% CI (repo bootstrap) | helps | hurts |
|---|---|---|---|---|---|
| stop | 0.1720 | +0.0000 | [+0.0000, +0.0000] | 0% | 0% |
| expand_inherit | 0.1721 | +0.0001 | [-0.0001, +0.0002] | 3% | 2% |
| expand_imports | 0.1698 | -0.0022 | [-0.0163, +0.0074] | 38% | 42% |
| expand_calls | 0.1775 | +0.0055 | [-0.0086, +0.0133] | 41% | 37% |
| dense | 0.1523 | -0.0197 | [-0.0470, -0.0057] | 45% | 48% |
| fusion | 0.1844 | +0.0124 | [+0.0021, +0.0306] | 52% | 35% |

Best fixed action: **fusion** (0.1844). Per-issue oracle over 6 actions: 0.2582 [0.2022, 0.3503], +40% over the best fixed action.

Oracle choice (cheapest action reaching the per-issue best): stop 17%, expand_inherit 0%, expand_imports 20%, expand_calls 19%, dense 22%, fusion 21%

(Outside dev only, since lambda and k were tuned on dev.)
## Phase A, ndcg@300, 444 issues

| action | mean | gain vs stop | 95% CI (repo bootstrap) | helps | hurts |
|---|---|---|---|---|---|
| stop | 0.1786 | +0.0000 | [+0.0000, +0.0000] | 0% | 0% |
| expand_inherit | 0.1787 | +0.0001 | [-0.0002, +0.0002] | 4% | 2% |
| expand_imports | 0.1743 | -0.0043 | [-0.0213, +0.0055] | 37% | 44% |
| expand_calls | 0.1812 | +0.0026 | [-0.0183, +0.0104] | 40% | 39% |
| dense | 0.1579 | -0.0207 | [-0.0541, -0.0052] | 44% | 49% |
| fusion | 0.1872 | +0.0086 | [-0.0010, +0.0244] | 52% | 36% |

Best fixed action: **fusion** (0.1872). Per-issue oracle over 6 actions: 0.2618 [0.2027, 0.3762], +40% over the best fixed action.

Oracle choice (cheapest action reaching the per-issue best): stop 18%, expand_inherit 0%, expand_imports 20%, expand_calls 19%, dense 23%, fusion 20%

### Gain vs stop by repository

| group | n | stop | expand_inherit | expand_imports | expand_calls | dense | fusion |
|---|---|---|---|---|---|---|---|
| django/django | 207 | 0.1107 | +0.0003 | +0.0025 | +0.0103 | -0.0070 | +0.0054 |
| sympy/sympy | 66 | 0.2387 | -0.0002 | -0.0312 | +0.0035 | -0.0394 | -0.0032 |
| ansible/ansible | 40 | 0.2857 | +0.0001 | +0.0053 | -0.0042 | -0.0235 | +0.0211 |
| scikit-learn/scikit-learn | 31 | 0.3925 | -0.0003 | -0.0312 | -0.0457 | -0.1090 | -0.0039 |
| qutebrowser/qutebrowser | 30 | 0.1796 | +0.0003 | +0.0292 | +0.0113 | +0.0225 | +0.0322 |
| matplotlib/matplotlib | 29 | 0.1422 | +0.0000 | -0.0070 | +0.0321 | +0.0199 | +0.0581 |
| internetarchive/openlibrary | 20 | 0.1368 | +0.0000 | +0.0090 | +0.0272 | -0.0143 | +0.0374 |
| pytest-dev/pytest | 18 | 0.0858 | +0.0000 | +0.0164 | +0.0317 | -0.0418 | -0.0219 |
| pydata/xarray | 18 | 0.1335 | +0.0000 | +0.0193 | +0.0281 | +0.0229 | +0.1121 |
| sphinx-doc/sphinx | 12 | 0.1333 | -0.0005 | -0.0084 | -0.0071 | -0.0208 | +0.0171 |
| pylint-dev/pylint | 10 | 0.2209 | +0.0000 | +0.0348 | -0.0410 | -0.0784 | -0.0463 |
| astropy/astropy | 9 | 0.2762 | +0.0000 | -0.0271 | -0.0546 | -0.0671 | -0.0269 |
| psf/requests | 7 | 0.1469 | +0.0000 | -0.0271 | +0.0230 | -0.0560 | -0.0279 |
| mwaskom/seaborn | 2 | 0.0850 | +0.0000 | -0.0364 | -0.0795 | -0.0333 | +0.0494 |
| pallets/flask | 1 | 0.0476 | +0.0000 | -0.0455 | +0.0801 | +0.0082 | +0.1422 |

### Gain vs stop by source

| group | n | stop | expand_inherit | expand_imports | expand_calls | dense | fusion |
|---|---|---|---|---|---|---|---|
| Pro | 90 | 0.2172 | +0.0001 | +0.0141 | +0.0080 | -0.0061 | +0.0284 |
| Verified | 410 | 0.1621 | +0.0001 | -0.0058 | +0.0050 | -0.0226 | +0.0089 |

### Gain vs stop by whether the issue names a target file

| group | n | stop | expand_inherit | expand_imports | expand_calls | dense | fusion |
|---|---|---|---|---|---|---|---|
| names a repo file | 89 | 0.1836 | -0.0002 | +0.0125 | +0.0113 | -0.0181 | -0.0016 |
| does not | 411 | 0.1695 | +0.0002 | -0.0054 | +0.0043 | -0.0200 | +0.0154 |
