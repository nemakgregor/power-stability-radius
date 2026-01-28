# Verification report

## Notes on workflow

- Base point is produced by **opf_pypsa** (PyPSA DC OPF + HiGHS).
- Input cases are auto-downloaded deterministically when missing (supported filenames only).
- No PF/AC-OPF modes are supported.

---

Таблица (критерии usability по задаче: coverage > 70%, match > 70%, time < 10 sec):

| case | status | coverage % | top risky match % | N-1 critical match % | time sec |
|---|---|---:|---:|---:|---:|
| case30 | config_mismatch_generated_n1_match_undefined | 0.000 | 10.000 | n/a | 31.908 |
| case118 | config_mismatch_generated_n1_match_undefined | 0.000 | 0.000 | n/a | 6.568 |
| case300 | config_mismatch_generated_n1_match_undefined | 0.000 | n/a | n/a | 11.149 |
| case1354_pegase | config_mismatch_generated_n1_match_undefined | 0.000 | n/a | n/a | 53.417 |
| case9241_pegase | config_mismatch_generated_n1_match_undefined | 0.000 | n/a | n/a | 763.901 |

---

## case30

- status: **config_mismatch_generated_n1_match_undefined**
- coverage % (MC): **0.000%** (= 0 / 50000; feasible_in_box=50000/50000; mc_status=ok)
- base feasibility (w.r.t. stored limits): **feasible**
- top risky match % (common/10): **10%** (common=1, top_k=10, known_k=3)
- top risky recall % (common/known): **33.3333%** (= 1 / 3)
- N-1 critical match %: n/a (status=no_finite_radius_nminus1; selected=0)
- time sec (demo): **31.908**

### Top-10 risky lines (min radius_l2)

| rank | line_idx | from_bus | to_bus | radius_l2 |
|---:|---:|---:|---:|---:|
| 1 | 0 | 1 | 2 | 5.8316 |
| 2 | 11 | 12 | 15 | 14.2413 |
| 3 | 20 | 10 | 21 | 16.0656 |
| 4 | 18 | 10 | 20 | 19.1961 |
| 5 | 28 | 25 | 27 | 20.5844 |
| 6 | 27 | 25 | 26 | 21.5 |
| 7 | 24 | 22 | 24 | 22.0601 |
| 8 | 26 | 24 | 25 | 22.219 |
| 9 | 17 | 19 | 20 | 24.4067 |
| 10 | 29 | 27 | 29 | 26.5431 |

### Scalability check

- Criterion: time < 10 sec => **FAIL**

## case118

- status: **config_mismatch_generated_n1_match_undefined**
- coverage % (MC): **0.000%** (= 0 / 4084; feasible_in_box=4084/50000; mc_status=ok)
- base feasibility (w.r.t. stored limits): **feasible**
- top risky match % (common/10): **0%** (common=0, top_k=10, known_k=2)
- top risky recall % (common/known): **0%** (= 0 / 2)
- N-1 critical match %: n/a (status=no_finite_radius_nminus1; selected=0)
- time sec (demo): **6.568**

### Top-10 risky lines (min radius_l2)

| rank | line_idx | from_bus | to_bus | radius_l2 |
|---:|---:|---:|---:|---:|
| 1 | 34 | 26 | 30 | 0 |
| 2 | 96 | 65 | 68 | 0 |
| 3 | 130 | 89 | 92 | 0 |
| 4 | 152 | 100 | 103 | 0 |
| 5 | 1 | 1 | 3 | 16.9018 |
| 6 | 114 | 77 | 80 | 25.3567 |
| 7 | 3 | 3 | 5 | 25.7022 |
| 8 | 144 | 94 | 100 | 28.5361 |
| 9 | 11 | 2 | 12 | 28.5806 |
| 10 | 46 | 34 | 37 | 29.3833 |

### Scalability check

- Criterion: time < 10 sec => **PASS**

## case300

- status: **config_mismatch_generated_n1_match_undefined**
- coverage % (MC): **0.000%** (= 0 / 2066; feasible_in_box=2066/50000; mc_status=ok)
- base feasibility (w.r.t. stored limits): **feasible**
- top risky match %: n/a
- N-1 critical match %: n/a (status=no_finite_radius_nminus1; selected=0)
- time sec (demo): **11.149**

### Scalability check

- Criterion: time < 10 sec => **FAIL**

## case1354_pegase

- status: **config_mismatch_generated_n1_match_undefined**
- coverage % (MC): **0.000%** (= 0 / 50000; feasible_in_box=50000/50000; mc_status=ok)
- base feasibility (w.r.t. stored limits): **feasible**
- top risky match %: n/a
- N-1 critical match %: n/a (status=no_finite_radius_nminus1; selected=0)
- time sec (demo): **53.417**

### Scalability check

- Criterion: time < 10 sec => **FAIL**

### Literature comparison (Nguyen 2018)
Nguyen (2018) отмечает, что полито́пные convex inner approximations покрывают существенные доли true region порядка 50–90% на 1354 buses (arXiv:1708.06845v3). Здесь получено coverage ≈ 0.000%.
Критерий адекватности по задаче: >70%. Условие не выполнено.

## case9241_pegase

- status: **config_mismatch_generated_n1_match_undefined**
- coverage % (MC): **0.000%** (= 0 / 50000; feasible_in_box=50000/50000; mc_status=ok)
- base feasibility (w.r.t. stored limits): **feasible**
- top risky match %: n/a
- N-1 critical match %: n/a (status=no_finite_radius_nminus1; selected=0)
- time sec (demo): **763.901**

### Scalability check

- Criterion: time < 10 sec => **FAIL**

### Literature comparison (Nguyen 2018, Lee 2019)
Nguyen (2018) на large-scale тестах также демонстрирует substantial fractions для внутренних аппроксимаций. Lee (2019, IEEE TPS) указывает, что convex quadratic restriction может быть достаточно большой для практической эксплуатации. Здесь получено coverage ≈ 0.000%.
Критерий адекватности по задаче: >70%. Условие не выполнено.

