# UNITS_CONTRACT — контракты единиц измерения, моделей и схем данных

Этот файл — **главная спецификация** проекта:
- что именно вычисляется (математические модели),
- какие сделаны допущения (DC/AC),
- какие единицы измерения и соглашения о знаках используются,
- как устроены входы/выходы (`results.json`, `__meta__`, per-line поля),
- какие ограничения и fail-fast правила являются частью “контракта”,
- какие изменения предлагаются (proposed contracts), не ломая текущую совместимость.

> Важно: тесты в `tests/` следует рассматривать как исполняемую часть этого контракта.

---

## 0. Область применения и “что такое stability radius”

### Цель
Для заданного базового режима сети и ограничений на линии оценить **робастность** режима к возмущениям узловых инъекций.

Типичный вопрос:
- “Какой максимальный размер возмущения (в смысле L2-нормы в пространстве узлов) гарантированно не приведёт к перегрузке какой-либо линии?”

### Определение (в духе сертификата)
Пусть для линии `ℓ` ограничение имеет вид:

- DC: `|f_ℓ(Δp)| ≤ c_ℓ`
- AC: `max(|S_from(ΔP,ΔQ)|, |S_to(ΔP,ΔQ)|) ≤ c_ℓ` (на практике сертификат строится отдельно по каждому концу)

Пусть `Δu` — вектор возмущений (DC: `Δu = Δp`, AC: `Δu = [ΔP; ΔQ]`).

**Радиус устойчивости/робастности** (в смысле сертификата) — это число `r ≥ 0`, такое что:
> для всех `Δu`, удовлетворяющих `||Δu||₂ ≤ r`, ограничения линий не нарушаются (в рамках используемой модели/линейризации).

Проект вычисляет **сертификаты** (нижние оценки). Они гарантируют безопасность внутри шара, но не претендуют на точное вычисление “истинного” максимального радиуса для нелинейной AC системы.

---

## 1. Глоссарий

- **Base point / базовая точка** — режим, вокруг которого строится линейная модель (и сертификат).
- **Slack bus** — опорный узел, задающий ссылку для углов; в DC/AC редукции удаляется одна степень свободы.
- **Balanced disturbances (балансированные возмущения)** — возмущения, суммарная активная (и в AC также реактивная) мощность которых равна нулю:
  - DC: `1ᵀ Δp = 0`
  - AC: `1ᵀ ΔP = 0` и `1ᵀ ΔQ = 0`
- **H_full** — матрица чувствительности потоков по линиям к узловым инъекциям в DC модели: `Δf = H_full Δp`.
- **DCOperator** — операторная реализация DC модели без обязательной материализации `H_full`.
- **LODF** — line outage distribution factors, быстрая аппроксимация перераспределения потоков при отключении линии.
- **ACOperator** — оператор Якобиана (разреженный) и LU для решения адъюнктных систем при AC сертификате.
- **Certificate soundness** — корректность сертификата: “внутри шара нарушений нет”.
- **Usefulness** — полезность: сертификат может быть “логически верным, но тривиальным” (например, `r* = 0`).
- **AC FPF (Feasible Power Flow)** — OPF-задача, минимизирующая отклонение генераторов от начальной диспетчеризации при соблюдении AC ограничений. Решается `pandapower.runopp`.
- **Metric radius** — L2-радиус в пространстве с метрикой M: `||Δu||_M = sqrt(Δuᵀ M Δu)`. При M = I совпадает с обычным L2-радиусом.
- **Sigma radius (σ-radius)** — безразмерное расстояние от базового потока до лимита, нормированное по σ потока: `r_σ = margin / σ_flow`.
- **Worst-case verification** — проверка направления worst-case при нелинейном AC PF: применяется возмущение `Δu = r · d*` и решается полный PF для проверки реального |S|.

---

## 2. Детерминизм и правила “без скрытых эффектов” (CURRENT)

### 2.1. Нет неявных скачиваний входов
- Если входной `.m` файл отсутствует:
  - `allow_download=false` → **FileNotFoundError**
  - `allow_download=true` → детерминированная попытка скачать кейс по имени файла
- Это важно для воспроизводимости и CI.

### 2.2. Стабильный порядок индексов
Во всех вычислениях используются **стабильные упорядочивания**:
- `bus_ids = sorted(net.bus.index)`
- `line_ids = sorted(net.line.index)`
- Этот порядок является “оси координат” для массивов в `results.json` и внутренних операторов.

### 2.3. Fail-fast вместо эвристик
Если режим/конфигурация несовместимы с алгоритмом, проект завершает работу с явной ошибкой, например:
- AC Monte Carlo требует `pandapower` как per-sample PF движок.
- `ac.lossless=false` не реализован (и для сертификата, и для AC MC).

---

## 3. Единицы измерения и соглашения о знаках

### 3.1. Источники данных
Вход: MATPOWER/PGLib `.m` → конвертация в `pandapower net`.

**Ключевой момент:** MATPOWER/PGLib rating `rateA` обычно в **MVA**.

### 3.2. DC (линейная модель)
Единицы:
- узловые активные инъекции: **MW**
- потоки по линиям: **MW**
- лимиты линий: исходно **MVA**, но в DC трактуются как **MW** при предположении `PF=1` (lossless DC соглашение проекта)

Соглашение о знаке потока:
- `flow0_mw` для линии — **знаковый** поток в направлении “bus0 → bus1” (ориентация берётся из pandapower `from_bus`, `to_bus` и/или из построения в PyPSA).

### 3.3. AC (сертификат вокруг AC PF)
Единицы:
- `ΔP`: **MW**
- `ΔQ`: **MVAr**
- базовые потоки на концах линии:
  - `p_from_mw`, `q_from_mvar`, `p_to_mw`, `q_to_mvar`
- ограничения линии: **MVA**
- проверка ограничений в AC MC: по величине полной мощности:
  - `|S| = sqrt(P² + Q²)` в **MVA**

---

## 4. Базовые точки (base point) и диспетч (dispatch)

Проект различает:
- **источник диспетча активной мощности** (case vs DC OPF),
- **базовую точку для AC сертификата** (всегда AC PF решение).

### 4.1. DC base point (CURRENT)

#### A) `base_dispatch=case`
- Инъекции берутся из `pandapower net` (load/shunt/gen/sgen).
- Далее вектор инъекций **балансируется**:
  - дисбаланс (сумма инъекций) добавляется/вычитается в slack bus так, чтобы сумма стала ровно 0.
- Потоки базовой точки восстанавливаются через DCOperator:
  - `f0 = DCOperator.flows_from_bus_injections_mw(p)`

#### B) `base_dispatch=dc_opf`
- Решается DC OPF (PyPSA + HiGHS).
- Результат содержит:
  - `line_flows_mw` (базовые потоки по линиям)
  - `bus_injections_mw` (балансированные инъекции)
  - `gen_dispatch_mw_by_name` (активная мощность генераторов для воспроизводимости режима)
- Выполняется проверка согласованности:
  - потоки OPF должны совпадать с восстановлением через `DCOperator.flows_from_bus_injections_mw(...)` в пределах tolerances.

**Контракт:** DC OPF используется как источник диспетча и базовых потоков в DC части и как источник `P` для последующего AC PF (если AC включён).

### 4.2. AC base point (CURRENT)
- Решается **AC PF**, а не OPF.
- Solver: `pandapower` или `pypsa`.
- При `base_dispatch=dc_opf` активная мощность генераторов из OPF применяется к `pandapower net` (только `net.gen.p_mw`), затем решается AC PF.

**Контракт:** AC сертификат вычисляется вокруг найденного AC PF режима (Vm, Va), и эти значения сохраняются в `__meta__.base_point_ac` для воспроизводимости и проверок.

### 4.3. AC FPF base point (CURRENT)
Альтернативный метод получения AC базовой точки через **AC OPF feasibility** (`pandapower.runopp`).

#### Когда используется
Когда обычный AC PF (runpp) не сходится на диспетчеризации из DC OPF, или когда нужна базовая точка, гарантированно удовлетворяющая тепловые ограничения линий.

#### Математическая формулировка
```
min  Σ_i (P_{g,i} − P_{g,i}⁰)²
s.t. AC power flow equations           (equality)
     P_g^min ≤ P_g ≤ P_g^max          (generator limits)
     Q_g^min ≤ Q_g ≤ Q_g^max          (reactive limits)
     V^min   ≤ V   ≤ V^max            (voltage bounds)
     |S_ij|  ≤ S_ij^max               (thermal limits)
```

Цель — найти ближайший к P⁰ допустимый режим (минимальное отклонение от начального диспетча).

#### Конфигурация (`ACFPFConfig`)
| Параметр               | По умолчанию | Описание |
|------------------------|:------:|-----|
| `pg0_source`           | `"case"` | Источник P⁰: `"case"` (net.gen.p_mw) или `"midpoint"` ((min+max)/2) |
| `vm_min_pu / vm_max_pu`| 0.9 / 1.1 | Допустимые границы |V| для OPF |
| `max_iteration`        | 300    | Максимальное число итераций PDIPM |
| `max_loading_percent`  | 99.0   | Лимит загрузки линий (99% компенсирует допуски PIPS) |
| `max_attempts`         | 1      | Число попыток с расширением границ (до 3) |
| `per_attempt_timeout`  | 0      | Тайм-аут на одну попытку runopp (0 = без ограничения) |

#### Post-OPP PF validation
После OPP (interior-point) выполняется повторный PF (`runpp`) с найденной диспетчеризацией, чтобы Якобиан AC сертификата линейризовался в точке Newton-Raphson, а не PIPS. Это критично, т.к. верификация (MC, worst-case) также использует `runpp`.

#### Реализация
- Модуль: `stability_radius.base_point.pandapower_opp`
- Entrypoint: `solve_ac_fpf()`
- Обёртка: `stability_radius.base_point.ac.solve_ac_fpf_base_point()`

**Контракт:** AC FPF возвращает `PyPSAAPFResult` (тот же тип, что AC PF), поэтому downstream-код (AC сертификат, DC base point from ACPF) работает без изменений.

---

## 5. DC модель (математика и реализация)

### 5.1. Допущения DC модели (CURRENT)
- Потери пренебрежимы (lossless).
- Углы малы, напряжения по модулю считаются фиксированными.
- Основная динамика потоков задаётся реактивностью ветвей.

### 5.2. DCOperator (CURRENT)
DCOperator строит разреженную систему для углов (с удалённым slack):

- Строится ориентированная инцидентность `A` для набора ветвей, участвующих в B-матрице.
- Коэффициенты ветвей `b` в **MW/rad** примерно:
  - `b ≈ V_kV^2 / X_ohm`
- Редуцированная матрица:
  - `B_red = A_redᵀ diag(b) A_red`
- Для возмущений:
  - `Δf_lines = W * (B_red^{-1} * Δp_red)`
  - где `W = diag(b_lines) * A_lines_red`

#### Какие элементы входят в B-матрицу
Для избегания вырожденности и радиальных эффектов:
- линии (`net.line`)
- трансформаторы (`net.trafo`) с учётом tap ratio (DC аппроксимация) и phase shift
- элементы `net.impedance`

#### Фазосдвигающие трансформаторы
Если `shift_degree != 0`, появляется постоянный член в уравнениях, который хранится как `shift_inj_red` и учитывается при восстановлении **абсолютных** потоков базовой точки.

### 5.3. Operator vs Materialize режимы (CURRENT)

#### Режим `dc.mode=operator`
- `H_full` **не материализуется**.
- Нормы строк `||Proj(g_ℓ)||₂` вычисляются пачками через адъюнкт/решение (через LU `B_red`).
- Память: низкая.
- N-1: **недоступен**.

#### Режим `dc.mode=materialize`
- Строится плотная `H_full` размера `(n_line, n_bus)` (dtype configurable).
- Память: высокая (O(n_line*n_bus)).
- Позволяет:
  - N-1 эффективные радиусы через LODF
  - более “прямые” пост‑обработки

---

## 6. DC радиусы и вероятностные метрики

### 6.1. DC L2 радиус (балансированные возмущения)
Проект сертифицирует устойчивость к `Δp`, удовлетворяющим `sum(Δp)=0`, и измеряет размер возмущения в полной норме `||Δp||₂` в пространстве всех узлов.

Проблема: чувствительность строки `g` определена с точностью до добавления константы `α·1` (из-за выбора slack).
Решение: использовать проектированную норму:

- `Proj(g) = g - mean(g)·1`
- `||Proj(g)||₂² = ||g||₂² - (sum(g))² / n_bus`

Тогда для линии ℓ:
- `margin_ℓ = c_ℓ - |f0_ℓ|`
- `r_ℓ = margin_ℓ / ||Proj(g_ℓ)||₂`

Глобальный сертификат:
- `r* = min_ℓ r_ℓ`

### 6.2. Sigma radius и overload probability
Если `Δp ~ N(0, σ² I)` в балансированном подпространстве, то поток по линии имеет стандартное отклонение:
- `σ_flow_ℓ = σ_inj * ||Proj(g_ℓ)||₂`

Тогда:
- `r_sigma = margin / σ_flow`
- вероятность перегруза при ненулевом базовом потоке (симметричный лимит ±c):
  - `P(|f0 + X| > c) = Q((c - |f0|)/σ_flow) + Q((c + |f0|)/σ_flow)`

---

## 7. N-1 эффективные радиусы (DC)

### 7.1. Идея
Хотим оценить “робастность с учётом одиночного отключения линии”.

Используется аппроксимация через LODF:
- базовые потоки после отключения `k`:
  - `f^(k) = f + LODF[:,k] * f_k`, `f_k^(k)=0`
- чувствительность строки под отключением:
  - `g_m^(k) ≈ g_m + LODF[m,k] * g_k` (если включён `update_sensitivities`)

Эффективный радиус линии m:
- `r_m^(N-1) = min_{k != m} margin_m^(k) / ||Proj(g_m^(k))||₂`

### 7.2. Islanding
Если `1 - PTDF[k,k] ≈ 0`, LODF не определён (островная ситуация / радиальный разрез).
Контракт:
- `islanding=skip` → колонка считается NaN и контингентность пропускается (логируется WARNING)
- `islanding=raise` → ошибка

---

## 8. AC сертификат (AC L2 вокруг AC PF)

### 8.1. Допущения (CURRENT)
- Сертификат строится вокруг **AC PF решения**.
- В текущей версии реализован только режим `lossless=true`:
  - сопротивления линий принудительно `r=0` (для согласования модели сертификата и MC режима)
- PV/PQ switching и жёсткие Q‑лимиты генераторов не моделируются в сертификате “как в дискретной нелинейной задаче”; сертификат — это линейная оценка вокруг найденной точки.

### 8.2. ACOperator: Якобиан и адъюнктные решения
Строится разреженный Якобиан PF в редуцированной форме (без slack) для уравнений P/Q по не‑slack узлам и переменных углов/модулей по не‑slack узлам:

- `J * dx = du`, где:
  - `x = [θ_non_slack; V_non_slack]`
  - `u = [P_non_slack; Q_non_slack]`

Для вычисления чувствительности скаляра ограничения (например, `|S_from|`) по инъекциям используется адъюнкт:
- решается `Jᵀ y = b`
- затем `y` интерпретируется как коэффициенты линейной формы по `ΔP/ΔQ`.

### 8.3. Ограничение линии и “двухконечность”
Для каждой линии существуют **две** потенциально связывающие стороны:
- from-end
- to-end

Проект вычисляет радиус отдельно для каждого конца, затем по линии берёт минимум.

Unified поля per line (используются в таблицах и AC MC):
- `binding_end`: `"from"` или `"to"`
- `margin_ac_mva`: запас по связывающему концу
- `||h||2`: норма дуальной чувствительности по связывающему концу
- `radius_ac_l2`: итоговый радиус по линии (min(from,to))

### 8.4. Балансирование в AC
Если `compute.ac.balance=true`, проект сертифицирует возмущения в подпространстве:
- `1ᵀ ΔP = 0`
- `1ᵀ ΔQ = 0`

Норма чувствительности учитывает проектирование отдельно для P и Q блоков.

### 8.5. AC Metric Radius (CURRENT)
Обобщение L2-радиуса на пространство с метрикой M (SPD):

```
r_M = margin / ||h||_M⁻¹
```

где: `||h||_M⁻¹ = sqrt(hᵀ M⁻¹ h)`.

При `M = I` это обычный L2-радиус. При `M = diag(1/σ²)` это совпадает с σ-радиусом (cross-check).

#### Реализация
- Модуль: `stability_radius.radii.ac_metric_radius`
- Функция: `compute_ac_metric_radius()`
- Поддерживаемые режимы M:
  - `diag`: одномерный массив `d`, M = diag(d), быстрый путь
  - `dense`: произвольная SPD матрица, факторизация Холецкого
- Валидация: M должна быть SPD (positive definite), иначе — `ValueError`

### 8.6. AC Sigma Radius (CURRENT)
Безразмерная метрика "сколько σ от базового потока до лимита":

```
r_σ = margin_mva / σ_flow_mva
```

где:
- `margin_mva = s_limit − s_binding` — запас по связывающему концу (MVA)
- `σ_flow_mva = ||h||_{diag(σ²)} = sqrt(hᵀ diag(σ²) h)` — standard deviation потока при гауссовых возмущениях (Σ = diag(σ²))

#### Вероятность перегрузки (Gaussian)
```
P(|S| ≥ limit) ≈ Φ(−r_σ)
```

где Φ — функция стандартного нормального распределения. Точна при σ потока достаточно мал (линеаризация точна).

#### Источники σ
| Значение `sigma_p_mw_source` | Описание |
|------|------|
| `"uniform"` | Одно скалярное значение σ на все шины (broadcast) |
| `"uc_jl"` | Per-bus массив из UnitCommitment.jl instance (population std) |
| `""` (пусто) | σ-radius не вычисляется |

#### Реализация
- Модуль: `stability_radius.radii.ac_sigma_radius`
- Функция: `compute_ac_sigma_radius()`
- Worst-case direction: `d* = Σ h / ||Σ h||₂ · r_σ` (масштабированная проекция)

### 8.7. Worst-Case Verification (CURRENT)
Проверка сертификата путём применения worst-case возмущения к нелинейному AC PF:

1. Берётся направление `d*` (worst-case из линейного сертификата)
2. Масштабируется до `scale · r* · d*` (scale ∈ {0.5, 0.8, 0.9, 1.0, 1.05, 1.1, 1.2})
3. Добавляются `sgen` с `(ΔP, ΔQ)` на каждую шину
4. Решается полный AC PF (`pandapower.runpp`)
5. Измеряется фактический `|S|` на связывающем конце
6. Сравнивается с лимитом: violation = `|S_actual| > s_limit`

#### Ожидаемый результат
- При `scale ≤ 1.0`: violation не ожидается (сертификат гарантирует)
- При `scale > 1.0`: violation может быть (тест tightness сертификата)
- При PF non-convergence: результат — NaN (diverged)

#### Реализация
- Модуль: `stability_radius.verification.verify_worst_case`
- Функция: `verify_worst_case()`
- Тип результата: `WorstCaseVerificationResult`

### 8.8. AC Sigma Monte Carlo (CURRENT)
Monte Carlo верификация σ-радиуса: сэмплирование гауссовых возмущений и проверка нелинейным PF.

- Сэмплы: `Δu ~ N(0, Σ)`, масштабированные до шара радиуса `r_σ`
- Для каждого сэмпла: AC PF → фактический `|S|` → violation yes/no
- Результат:
  - `empirical_feasible_fraction`: доля безопасных сэмплов
  - `per_line_overload_fractions`: per-line эмпирическая вероятность перегрузки
  - `pf_failure_fraction`: доля несходимостей PF

#### Реализация
- Модуль: `stability_radius.verification.ac_monte_carlo_sigma`
- Функция: `run_ac_monte_carlo_sigma()`
- Тип результата: `ACSigmaMCResult`

---

## 9. Monte Carlo верификация (DC/AC)

### 9.1. DC MC (CURRENT)
- Генерируются балансированные гауссовы возмущения `Δp`.
- Потоки считаются линейно:
  - `f = f0 + DCOperator.flows_from_delta_injections(Δp)`
- Проверяется:
  - доля безопасных (`|f| ≤ c + feas_tol`)
  - soundness внутри шара радиуса `r*` (равномерные точки в L2 шаре, балансированные)
  - вероятность массы шара под гауссом (аналитически через χ² CDF и эмпирически)

### 9.2. AC MC (CURRENT)
- Per-sample PF выполняется **только `pandapower.runpp`**.
- Для каждого сэмпла добавляются `sgen` по всем шинам, задающие (ΔP, ΔQ), затем решается PF.
- Контракт корректности режима:
  - AC MC выполняет базовый PF без возмущений и сравнивает `|S|` на концах линий с полями `ac_s0_from_mva/ac_s0_to_mva` из `results.json` в пределах `ac.basepoint_s_tol_mva`.
- Если PF не сходится:
  - сэмпл считается неуспешным (в отчёте есть `pf_failures_gaussian`).

---

## 10. Схема данных `results.json` (CURRENT, schema_version=3)

### 10.1. Верхний уровень
- `__meta__`: объект с метаданными режима и схемы
- `line_<idx>`: объект на линию (pandapower `net.line.index`)

### 10.2. `__meta__` (ключевые поля)
Минимально значимые:
- `schema_version: 3`
- `input_path: <abs or resolved path>`
- `slack_bus: int` (как передан пользователем)
- `base_dispatch: "case" | "dc_opf"`
- `compute_dc: bool`, `compute_ac: bool`
- `dc: { mode, dtype, chunk_size, inj_std_mw, nminus1_computed }`
- `ac: { pf_solver, pf_init, lossless(true), chunk_size, balance, pf_status }`
- `ac.sigma_source: "uniform" | "uc_jl" | null` — source of injection σ arrays
- `ac.sigma_p_mw: list[float] | float | null` — per-bus σ_P or scalar (uniform), null if not computed
- `ac.sigma_q_mvar: list[float] | float | null` — per-bus σ_Q or scalar (uniform), null if not computed
- `ac.sigma_n_timesteps: int | null` — number of timesteps (from UC.jl), null if not applicable
- `ac.sigma_computed: bool`
- `ac.metric_enabled: bool`, `ac.metric_computed: bool`
- `ac.save_h_vectors: bool`
- `base_point_dc`: либо `null`, либо JSON-friendly структура
- `base_point_ac`: либо `null`, либо JSON-friendly структура
- `compute_time_sec: float`
- (опционально) результаты проверок согласованности OPF/DC:
  - `opf_bus_balance_abs_mw`, `opf_dc_flow_max_abs_diff_mw`, ...

### 10.3. Per-line поля (DC)
Минимальный набор (если DC считался):
- `flow0_mw` — signed MW
- `p0_mw` — abs(flow0_mw)
- `p_limit_mw_est` — лимит, трактуемый как MW (источник: MVA rating)
- `margin_mw` — `p_limit_mw_est - p0_mw`, клипнутый в [0, +inf] в compute
- `norm_g` — `||Proj(g)||₂` (балансированная проекция)
- `radius_l2` — L2 радиус (MW)
- `sigma_flow`, `radius_sigma`, `overload_probability` — вероятностные метрики (если вычислены)
- `radius_nminus1`, `worst_contingency_line_idx` — если включён N-1

### 10.4. Per-line поля (AC)
Минимальный набор (если AC считался):
- `ac_s_limit_mva`
- `ac_s0_from_mva`, `ac_s0_to_mva`
- `margin_ac_mva`
- `||h||2`
- `binding_end`
- `radius_ac_l2`

Sigma-radius поля (если `ac.sigma_computed=true`):
- `sigma_flow_mva` — flow standard deviation at binding end (MVA)
- `radius_ac_sigma` — dimensionless radius in σ units
- `overload_probability_ac` — Gaussian overload probability
- `worst_case_dp_mw` — list[float] per-bus worst-case ΔP perturbation (MW), or null
- `worst_case_dq_mvar` — list[float] per-bus worst-case ΔQ perturbation (MVAr), or null
- `worst_case_s_predicted_mva` — predicted |S| at binding end under worst-case perturbation (MVA)

---

## 11. Failure modes (CURRENT) — как интерпретировать ошибки

### 11.1. Нулевой радиус `r* = 0`
Это может быть:
- **binding constraint**: базовая точка на границе лимита (сертификат тривиален, но логически верен)
- **bad limits**: лимиты некорректны/нулевые, либо базовая точка не соответствует лимитам

Проект различает эти случаи на уровне статусов верификации.

### 11.2. Несходимость AC PF в MC
- PF failure → sample считается нарушающим (в статистике отдельно).
- Это нормальная ситуация для агрессивных σ или плохих базовых режимов.
- Но если PF не сходится даже на базовой точке — это ошибка режима/конфигурации.

### 11.3. Несогласованность OPF и DCOperator
Если `base_dispatch=dc_opf`, проект проверяет:
- баланс инъекций по узлам (сумма близка к 0)
- совпадение потоков OPF с восстановлением DCOperator

Несоответствие — ошибка, потому что иначе сертификат и верификация будут вычисляться для разных физических режимов.

---

# PROPOSED CONTRACTS (предложения) — без ломания текущей схемы

Ниже — предложения по улучшению контрактов и читаемости, которые можно внедрять постепенно.

## P1. Явная секция единиц в `__meta__`
Добавить:
```json
"units": {
  "dc_flow": "MW",
  "dc_limit": "MW_assumed_from_MVA_pf1",
  "ac_flow_p": "MW",
  "ac_flow_q": "MVAr",
  "ac_limit": "MVA"
}
```
Плюсы:
- меньше путаницы вокруг `p_limit_mw_est`.

## P2. Переименование `p_limit_mw_est` → `dc_limit_mw_pf1`
Сохранить старое поле как алиас на 1–2 версии.
Плюсы:
- название становится честным (мы реально используем MVA как MW при PF=1).

## P3. Декларация “distribution contract” для verification
Сейчас DC MC может брать σ из:
- `__meta__.dc.inj_std_mw` или override

Предложение:
- добавить в `__meta__` отдельный блок:
  - `verification_defaults: { dc_sigma_mw, ac_sigma_p_mw, ac_sigma_q_mvar }`
и использовать его для report/MC, чтобы меньше параметров передавать руками.

## P4. Стабильный список полей для таблиц
Зафиксировать “официальный” список колонок в одном месте и документировать в README/UNITS_CONTRACT.

## P5. Поддержка `ac.lossless=false` (долгосрочно)
Это серьёзное изменение модели:
- PF и сертификат должны использовать согласованную (r,x) модель,
- AC MC должен проверять тот же режим,
- появятся дополнительные эффекты (потери, изменение распределения потоков).

Предложение: реализовывать отдельной веткой, не смешивая с рефакторингом структуры.

---

## 12. Как читать тесты как спецификацию

Ключевые тестовые контракты:
- `test_certificate_concept.py`:
  - tightness на границе для worst-line направления
  - инвариантность к выбору slack при балансированных возмущениях
- `test_config_extends.py`:
  - корректная композиция `extends:` с относительными путями
- `test_verification_report_and_monte_carlo.py`:
  - отчёт не должен печатать `nan%` (только `n/a`)
- `test_opf_dc_consistency.py`:
  - OPF и DCOperator должны давать согласованные потоки (в пределах tol)
- `test_ac_metric_radius.py`:
  - диагональная M, плотная M, M=I → L2, валидация SPD, нулевой знаменатель → inf
- `test_ac_sigma_radius.py`:
  - формула margin/σ_flow, баланс, worst-case point, overload probability
- `test_verify_worst_case.py`:
  - violation при boundary-scale, отсутствие violation при half-scale, PF divergence → NaN
- `test_ac_mc_sigma.py`:
  - soundness внутри σ-шара, per-line overload probabilities, валидация входов
- `test_pp_helpers.py`:
  - `is_in_service` (dict/Series/fallback), `bus_vn_kv` (5 ветвей), `resolve_slack_pos` (id/position/error)
- `test_verification_status.py`:
  - `summarize_status`: 6 статусов (OK, TRIVIAL, INFEASIBLE, UNSOUND, INCONCLUSIVE, NOT_COMPUTED)
- `test_statistics_table.py`:
  - ASCII/CSV форматирование, column inference (DC/AC/both/neither), radius summary
- `test_metrics_analysis.py`:
  - unified DataFrame, rank correlations (negation для radii), precision-at-k
- `test_workflows_helpers.py`:
  - `_merge_line_results` (merging, overwrites, numeric sort), `_build_sigma_arrays` (uniform/uc_jl/validation)

---

## 13. Итог: “истина” текущей версии

**CURRENT contracts**:
- DC сертификаты и вероятностные метрики основаны на балансированной L2‑геометрии и DCOperator.
- AC сертификат строится вокруг AC PF (или AC FPF) базовой точки и использует адъюнктные решения Якобиана.
- AC metric radius обобщает L2-радиус на метрику M (SPD); при M = diag(1/σ²) совпадает с σ-radius.
- AC σ-radius и worst-case verification документированы в секциях 8.5–8.8.
- Верификация Monte Carlo (DC/AC/AC-sigma) проверяет именно те режимы и поля, которые записаны в `results.json`.
- Проект придерживается детерминизма, стабильного порядка и fail-fast поведения.

Если вы меняете:
- единицы,
- порядок индексов,
- состав `__meta__`,
- семантику base point,
то это должно сопровождаться:
1) обновлением этого файла,
2) обновлением тестов, которые выражают контракт.