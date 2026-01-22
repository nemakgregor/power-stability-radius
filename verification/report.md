# Verification report

## How to reproduce (quick)

Generate report (auto-generates missing verification/results/*.json by default):

```bash
poetry run python src/power_stability_radius.py report \
  --results-dir verification/results --out verification/report.md
```

To disable auto-generation (only aggregate existing JSONs):

```bash
poetry run python src/power_stability_radius.py report \
  --generate-missing-results 0 \
  --results-dir verification/results --out verification/report.md
```

---

## Notes on metrics

- **MC coverage** is reported as `coverage = 100 * feasible_in_ball / total_feasible_in_box`.
- If `total_feasible_in_box = 0`, then coverage is **n/a** with `mc_status=no_feasible_samples` (not a crash).

---

Таблица (критерии usability по задаче: coverage > 70%, match > 70%, time < 10 sec):

| case | status | coverage % | top risky match % | N-1 critical match % | time sec |
|---|---|---:|---:|---:|---:|
| case30 | no_feasible_samples | n/a | 0.000 | 70.588 | 3.528 |
| case118 | no_feasible_samples | n/a | 0.000 | 83.908 | 5.517 |
| case300 | no_feasible_samples | n/a | n/a | 66.667 | 1.577 |
| case1354_pegase | no_feasible_samples | n/a | n/a | 83.543 | 78.200 |
| case9241_pegase | no_feasible_samples_n1_match_undefined | n/a | n/a | n/a | 50.450 |

---

## case30

- status: **no_feasible_samples**
- coverage % (MC): n/a (mc_status=no_feasible_samples; feasible_in_box=0/50000)
- top risky match % (common/10): **0%** (common=0, top_k=10, known_k=3)
- top risky recall % (common/known): **0%** (= 0 / 3)
- N-1 critical match %: **70.588%** (= 12 / 17; selected=17; status=ok)
- time sec (demo): **3.528**

### Monte Carlo coverage (details)

- status: **no_feasible_samples**
- n_samples=50000, seed=0, chunk_size=256
- sampling box: [box_lo, box_hi] = [-991.23, 991.23] (computed as [-2*max_r, 2*max_r], max_r=495.615)
- min_r (guaranteed L2 ball radius) = 0
- total_feasible_in_box = 0 / 50000 = 0%
- feasible_in_ball = 0
- coverage = feasible_in_ball / total_feasible_in_box * 100 = n/a (denominator is 0, i.e. no feasible samples in box)

### Known congested corridors (literature/manual)

- known pairs: 1-2, 2-4, 4-6
- common in top-10: **0**; match% (common/10) = **0.000%**; recall% (common/known) = **0.000%**
- Calculation: match = 100 * common / 10 = 100 * 0 / 10; recall = 100 * common / known = 100 * 0 / 3.
- Примечание: при малом числе известных пар метрика (common/10) имеет низкий максимум (например, для 3 известных линий максимум = 30.0%).

### Top-10 risky lines (min radius_l2)

| rank | line_idx | from_bus | to_bus | radius_l2 |
|---:|---:|---:|---:|---:|
| 1 | 0 | 0 | 1 | 0 |
| 2 | 11 | 11 | 14 | 12.0483 |
| 3 | 26 | 23 | 24 | 13.2125 |
| 4 | 20 | 9 | 20 | 13.5629 |
| 5 | 28 | 24 | 26 | 14.5661 |
| 6 | 24 | 21 | 23 | 14.7728 |
| 7 | 18 | 9 | 19 | 18.9595 |
| 8 | 27 | 24 | 25 | 21.5 |
| 9 | 17 | 18 | 19 | 23.5297 |
| 10 | 19 | 9 | 16 | 24.7507 |

### Adequacy check (top risky)

- Criterion (interpretable): recall% > 70% => **FAIL**

### Adequacy check (N-1 criticality)

- Criterion: N-1 critical match % > 80% => **FAIL**

### Scalability check

- Criterion: time < 10 sec => **PASS**

## case118

- status: **no_feasible_samples**
- coverage % (MC): n/a (mc_status=no_feasible_samples; feasible_in_box=0/50000)
- top risky match % (common/10): **0%** (common=0, top_k=10, known_k=2)
- top risky recall % (common/known): **0%** (= 0 / 2)
- N-1 critical match %: **83.908%** (= 73 / 87; selected=87; status=ok)
- time sec (demo): **5.517**

### Monte Carlo coverage (details)

- status: **no_feasible_samples**
- n_samples=50000, seed=0, chunk_size=256
- sampling box: [box_lo, box_hi] = [-1585.64, 1585.64] (computed as [-2*max_r, 2*max_r], max_r=792.818)
- min_r (guaranteed L2 ball radius) = 0
- total_feasible_in_box = 0 / 50000 = 0%
- feasible_in_ball = 0
- coverage = feasible_in_ball / total_feasible_in_box * 100 = n/a (denominator is 0, i.e. no feasible samples in box)

### Known congested corridors (literature/manual)

- known pairs: 38-65, 30-38
- common in top-10: **0**; match% (common/10) = **0.000%**; recall% (common/known) = **0.000%**
- Calculation: match = 100 * common / 10 = 100 * 0 / 10; recall = 100 * common / known = 100 * 0 / 2.
- Примечание: при малом числе известных пар метрика (common/10) имеет низкий максимум (например, для 2 известных линий максимум = 20.0%).

### Top-10 risky lines (min radius_l2)

| rank | line_idx | from_bus | to_bus | radius_l2 |
|---:|---:|---:|---:|---:|
| 1 | 89 | 37 | 64 | 0 |
| 2 | 97 | 46 | 68 | 0 |
| 3 | 98 | 48 | 68 | 0 |
| 4 | 99 | 68 | 69 | 0 |
| 5 | 107 | 68 | 74 | 0 |
| 6 | 110 | 68 | 76 | 0 |
| 7 | 100 | 23 | 69 | 0.117281 |
| 8 | 61 | 41 | 48 | 1.99557 |
| 9 | 62 | 41 | 48 | 1.99557 |
| 10 | 28 | 22 | 23 | 9.40243 |

### Adequacy check (top risky)

- Criterion (interpretable): recall% > 70% => **FAIL**

### Adequacy check (N-1 criticality)

- Criterion: N-1 critical match % > 80% => **PASS**

### Scalability check

- Criterion: time < 10 sec => **PASS**

## case300

- status: **no_feasible_samples**
- coverage % (MC): n/a (mc_status=no_feasible_samples; feasible_in_box=0/50000)
- top risky match %: n/a
- N-1 critical match %: **66.667%** (= 94 / 141; selected=141; status=ok)
- time sec (demo): **1.577**

### Monte Carlo coverage (details)

- status: **no_feasible_samples**
- n_samples=50000, seed=0, chunk_size=256
- sampling box: [box_lo, box_hi] = [-16903.5, 16903.5] (computed as [-2*max_r, 2*max_r], max_r=8451.73)
- min_r (guaranteed L2 ball radius) = 0
- total_feasible_in_box = 0 / 50000 = 0%
- feasible_in_ball = 0
- coverage = feasible_in_ball / total_feasible_in_box * 100 = n/a (denominator is 0, i.e. no feasible samples in box)

### Scalability check

- Criterion: time < 10 sec => **PASS**

## case1354_pegase

- status: **no_feasible_samples**
- coverage % (MC): n/a (mc_status=no_feasible_samples; feasible_in_box=0/50000)
- top risky match %: n/a
- N-1 critical match %: **83.543%** (= 731 / 875; selected=875; status=ok)
- time sec (demo): **78.200**

### Monte Carlo coverage (details)

- status: **no_feasible_samples**
- n_samples=50000, seed=0, chunk_size=256
- sampling box: [box_lo, box_hi] = [-627600, 627600] (computed as [-2*max_r, 2*max_r], max_r=313800)
- min_r (guaranteed L2 ball radius) = 0
- total_feasible_in_box = 0 / 50000 = 0%
- feasible_in_ball = 0
- coverage = feasible_in_ball / total_feasible_in_box * 100 = n/a (denominator is 0, i.e. no feasible samples in box)

### Adequacy check (N-1 criticality)

- Criterion: N-1 critical match % > 80% => **PASS**

### Literature comparison (Nguyen 2018)
Nguyen (2018) отмечает, что полито́пные convex inner approximations покрывают существенные доли true region порядка 50–90% на 1354 buses (arXiv:1708.06845v3). Здесь coverage: n/a (MC status: no_feasible_samples).
Критерий адекватности по задаче: >70%. Условие не может быть проверено (coverage = n/a).

### Scalability check

- Criterion: time < 10 sec => **FAIL**

## case9241_pegase

- status: **no_feasible_samples_n1_match_undefined**
- coverage % (MC): n/a (mc_status=no_feasible_samples; feasible_in_box=0/50000)
- top risky match %: n/a
- N-1 critical match %: n/a (status=no_finite_radius_nminus1; selected=0)
- time sec (demo): **50.450**

### Monte Carlo coverage (details)

- status: **no_feasible_samples**
- n_samples=50000, seed=0, chunk_size=256
- sampling box: [box_lo, box_hi] = [-1.42295e+06, 1.42295e+06] (computed as [-2*max_r, 2*max_r], max_r=711477)
- min_r (guaranteed L2 ball radius) = 0
- total_feasible_in_box = 0 / 50000 = 0%
- feasible_in_ball = 0
- coverage = feasible_in_ball / total_feasible_in_box * 100 = n/a (denominator is 0, i.e. no feasible samples in box)

### Literature comparison (Nguyen 2018, Lee 2019)
Nguyen (2018) на large-scale тестах также демонстрирует substantial fractions для внутренних аппроксимаций. Lee (2019, IEEE TPS) указывает, что convex quadratic restriction может быть достаточно большой для практической эксплуатации. Здесь coverage: n/a (MC status: no_feasible_samples).
Критерий адекватности по задаче: >70%. Условие не может быть проверено (coverage = n/a).

### Scalability check

- Criterion: time < 10 sec => **FAIL**

