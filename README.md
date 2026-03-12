
Инструмент для вычисления **радиусов устойчивости / робастности** по **ограничениям загрузки линий** в электроэнергетических сетях.

Проект решает практическую задачу:

> “Насколько можно (в терминах нормы возмущений инъекций по узлам) отклониться от базового режима, прежде чем какая‑либо линия нарушит термическое ограничение?”

Результат выдаётся **по каждой линии** (и агрегировано через минимум по линиям) и может быть:
- **DC (линейная модель)** — быстро, масштабируемо, подходит для больших сетей.
- **AC L2-сертификат вокруг AC PF базовой точки** — нелинейная физика учтена через линейное приближение (Якобиан/адъюнкт).

Дополнительно:
- **Monte Carlo верификация** (DC и AC).
- **Markdown отчёт по нескольким кейсам**.
- **Табличный вывод** (ASCII/CSV).

---

## Быстрый старт

Установка (Poetry):

```bash
poetry install
```

Запуск CLI:

```bash
poetry run python src/power_stability_radius.py --config conf/config.yaml <command> [options...]
```

Команды:
- `compute` (алиас: `demo`) — посчитать радиусы, сохранить `results.json` и таблицы.
- `monte-carlo` — проверить сертификат методом Монте‑Карло.
- `report` — прогнать верификацию по нескольким кейсам и собрать Markdown отчёт.
- `table` — утилита форматирования `results.json`.

---

## Важные свойства проекта (контракты верхнего уровня)

### 1) AC сертификат всегда строится вокруг **AC PF базовой точки**
- AC часть опирается на решение **AC Power Flow (AC PF)**.
- **DC OPF никогда не является AC базовой точкой**.
- Если нужен “OPF‑диспетч для режима”, это делается так:
  1) решаем **DC OPF (PyPSA + HiGHS)** для получения активной мощности генераторов,
  2) затем решаем **AC PF** уже на этой активной мощности,
  3) и строим AC сертификат вокруг найденного AC PF режима.

Управляется параметром:
- `compute.base_dispatch: case | dc_opf`

### 2) Детерминизм и отсутствие скрытых сайд‑эффектов
- **Нет неявных скачиваний** входных `.m` кейсов.
- Скачивание возможно только явно:
  - CLI: `--allow-download 1`
  - YAML: `io.allow_download: true`
- Везде используется **стабильная сортировка**:
  - buses: `sorted(net.bus.index)`
  - lines: `sorted(net.line.index)`

### 3) Явные ограничения / fail-fast политика
- `ac.lossless=false` **не поддержан** (явная ошибка).
- AC Monte‑Carlo **поддерживает только `pandapower`** как per-sample PF движок:
  - если радиус AC считали с `ac.pf_solver=pypsa`, то AC MC завершится с явной ошибкой.
- N-1 эффективные радиусы для DC требуют материализации `H_full`:
  - `--compute-nminus1 1` допускается только при `--dc-mode materialize`.

---

## Что такое “радиус устойчивости” в этом проекте

В линейной форме (DC и линейризованный AC) ограничения линии имеют вид:

- Пусть `f0` — базовый поток по линии (MW для DC, MVA для AC по модулю).
- Пусть `c` — симметричный лимит (в DC — “MVA, трактуемые как MW при PF=1”; в AC — лимит в MVA).
- Пусть возмущение инъекций по узлам `Δp` (и в AC также `Δq`).
- Пусть линейная чувствительность потока к инъекциям:
  - DC: `Δf = H Δp`
  - AC (сертификат): `Δ|S| ≈ hᵀ [ΔP; ΔQ]` (через адъюнкт-решение системы Якобиана)

Тогда по неравенству Коши–Буняковского для каждой линии возникает безопасный радиус:
- `margin = c - |f0|`
- `r = margin / ||g||` (DC) или `r = margin / ||h||` (AC)

Глобальный сертификат для режима:
- `r* = min_over_lines r_line`

Это **сертификат** (нижняя оценка) — он гарантирует безопасность внутри шара, но не обещает, что это точный максимум.

Подробная математика, единицы, допущения и схемы данных описаны в **`UNITS_CONTRACT.md`**.

---

## Архитектура репозитория

- `src/power_stability_radius.py` — тонкий entrypoint.
- `src/stability_radius/cli.py` — argparse CLI, компоновка YAML, запуск workflow.
- `src/stability_radius/workflows.py` — основной детерминированный пайплайн.
- `src/stability_radius/parsers/matpower.py` — детерминированный парсер MATPOWER/PGLib `.m` → pandapower net.
- `src/stability_radius/base_point/*` — генераторы базовых точек (DC case / DC OPF / AC PF).
- `src/stability_radius/dc/dc_model.py` — DCOperator (разреженная факторизация, PTDF‑подобные операции).
- `src/stability_radius/ac/ac_model.py` — ACOperator (Ybus, Якобиан, LU; используется в AC L2).
- `src/stability_radius/radii/*` — расчёт радиусов (DC L2 / metric / sigma / N-1, AC L2).
- `src/stability_radius/verification/*` — Monte Carlo верификация и отчёт.
- `tests/` — контрактные тесты (детерминизм, единицы, инварианты, smoke).

---

## Конфигурация (YAML с `extends`)

Входная точка: `conf/config.yaml`:

```yaml
extends:
  - ./config_shared.yaml
  - ./config_compute.yaml
  - ./config_monte_carlo.yaml
  - ./config_report.yaml
```

- `extends` реализован внутри проекта (через OmegaConf), с:
  - проверкой циклов,
  - разрешением путей относительно файла, который делает extends.

CLI‑флаги имеют приоритет над YAML.

---

## Артефакты запуска (run directory)

Каждая команда создаёт директорию в `runs/<module>/` (см. `logging.run_dir_mode`):
- `runs/<module>/<timestamp>/` (по умолчанию) или
- `runs/<module>/<run_name>/` (overwrite)

Типичные файлы:
- `run.log`
- `argv.txt`
- `config_source.yaml` (копия входного YAML)
- `config.json`, `config.yaml` (эффективная конфигурация)
- `results.json` + `results_table*.txt/csv` (для `compute`)
- `monte_carlo_stats.json` (для `monte-carlo`)
- `verification_report.md` (для `report`)

---

## Команда `compute` (основной пайплайн)

Семантика:
1) Загрузить `.m` → `pandapower net`.
2) (Опционально) DC OPF для диспетча (`base_dispatch=dc_opf`).
3) DC часть (если включена):
   - собрать `DCOperator` и/или `H_full`,
   - посчитать DC радиусы (L2 / sigma / probability / N-1).
4) AC часть (если включена):
   - решить AC PF базовую точку,
   - посчитать AC L2 сертификат на концах линий, агрегировать по линии.
5) Слить результаты в один `results.json`:
   - per-line ключи: `line_<idx>`
   - метаданные: `__meta__` (schema_version=2)

Пример (AC+DC, скачивание разрешено явно):

```bash
poetry run python src/power_stability_radius.py \
  --config conf/config.yaml \
  --run-tests 0 \
  --allow-download 1 \
  compute \
  --input data/input/pglib_opf_case30_ieee.m \
  --slack-bus 0 \
  --base-dispatch case
```

---

## Команда `monte-carlo` (верификация)

`monte-carlo` берёт:
- исходный `.m` кейс,
- `results.json` от `compute`,
и проверяет:
- DC: линейно, быстро, много сэмплов.
- AC: нелинейно, PF на каждый сэмпл (дорого), `pandapower` only.

Пример (DC):

```bash
poetry run python src/power_stability_radius.py \
  --config conf/config.yaml \
  --run-tests 0 \
  monte-carlo \
  --mode dc \
  --results verification/results/case30.json \
  --input data/input/pglib_opf_case30_ieee.m \
  --n-samples 50000 \
  --seed 42
```

---

## Команда `report` (multi-case Markdown)

- Читает список кейсов из YAML: `report.cases`.
- Ничего не скачивает и не генерирует “на лету”.
- В strict режиме требует DC/AC секции при наличии кейса.

Пример:

```bash
poetry run python src/power_stability_radius.py \
  --config conf/config.yaml \
  --run-tests 0 \
  report \
  --results-dir verification/results \
  --out verification/report.md
```

---

## Результаты: формат `results.json` (кратко)

- `__meta__`: версия схемы, входной файл, режим, настройки DC/AC, базовые точки.
- Для каждой линии: `line_<pandapower_line_index>`:
  - DC поля: `flow0_mw`, `p_limit_mw_est`, `margin_mw`, `norm_g`, `radius_l2`, ...
  - AC поля: `ac_s_limit_mva`, `ac_s0_from_mva`, `ac_s0_to_mva`, `||h||2`, `radius_ac_l2`, ...

Полный контракт схемы и единиц: **`UNITS_CONTRACT.md`**.

---

## Логи и трассируемость

Проект использует `logging`:
- консольный уровень по умолчанию: `INFO`
- файл: `DEBUG`
- системные этапы оборачиваются `log_stage(...)`, чтобы в логе были границы этапов и время.

---

## Минимальный план рефакторинга (без изменения математики)

Этот репозиторий уже следует принципу “минимальные зависимости и явные контракты”, но дальнейшие улучшения возможны.

### Минимально необходимые (MVP) улучшения
1) **Документировать** единицы/соглашения и схему результатов как “source of truth” (сделано в `UNITS_CONTRACT.md`).
2) Упорядочить и зафиксировать **schema_version** и миграции:
   - официально описать v2 (текущая),
   - наметить v3 (предложение без поломки совместимости).

### Опционально (после стабилизации)
- Добавить поддержку `ac.lossless=false` (потребует согласования PF, Якобиана и MC).
- Сделать AC Monte‑Carlo поддерживающим PyPSA per-sample PF (если появится детерминизм/устойчивость).
- Вынести генерацию таблиц/CSV в отдельный “export layer” (с минимальным количеством “магических” колонок).
