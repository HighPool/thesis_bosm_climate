# BOSM Climate

Projekt k diplomovej práci zameranej na optimalizáciu nákladných black-box úloh s využitím náhradných modelov v klimaticky orientovaných aplikáciách.

Aktuálna implementácia pokrýva dve úlohy:

* **LAQN** — rozmiestnenie environmentálnych senzorov odvodené z dát London Air Quality Network
* **WindWake** — rozmiestnenie veterných turbín nad simulovanou veternou farmou

Cieľom projektu je porovnať vybrané optimalizačné algoritmy v **jednotnom experimentálnom rámci**, zbierať porovnateľné metriky a vytvárať výstupy vhodné na ďalšiu analýzu a vizualizáciu.

---

## Implementované algoritmy

### Spoločné algoritmy

* Random Search
* PyBADS
* TuRBO

### pySOT kombinácie

* RBF + DYCORS
* RBF + SRBF
* RBF + SOP
* GP + EI
* GP + LCB
* Poly + DYCORS
* Poly + SRBF
* Poly + SOP

---

## Experimentálny rámec

Všetky algoritmy sú implementované tak, aby vracali rovnaké základné metriky:

* `X_hist`          — história všetkých navštívených bodov v poradí, v akom ich algoritmus vyhodnotil.
* `y_hist`          — história hodnôt cieľovej funkcie prislúchajúcich bodom z X_hist.
* `best_so_far`     — konvergenčná krivka, ktorá po každej evaluácii uchováva najlepšiu dovtedy nájdenú hodnotu.
* `best_x`          — najlepšie nájdené riešenie, teda bod rozhodovacieho priestoru s najlepšou dosiahnutou hodnotou cieľovej funkcie.
* `best_y`          — najlepšia dosiahnutá hodnota cieľovej funkcie počas daného behového spustenia.
* `budget`          — maximálny počet povolených evaluácií cieľovej funkcie v jednom behovom spustení.
* `call_count`      — skutočný počet vykonaných volaní cieľovej funkcie počas behového spustenia.
* `evals_to_f_best` — počet evaluácií potrebných na prvé dosiahnutie finálne najlepšieho riešenia.
* `total_time`      — celkový čas trvania jedného behového spustenia algoritmu.

To umožňuje:

* priame porovnanie algoritmov medzi sebou,
* agregáciu výsledkov po behoch a po problémoch,
* export priebehu optimalizácie do CSV pre **IOHanalyzer**.

---

## Štruktúra projektu

### `data/`

Vstupné dáta a problémové inštancie pre obe úlohy.

* `data/laqn/` — predspracované problémové inštancie LAQN
* `data/wind/` — vstupné súbory pre úlohu rozmiestnenia veterných turbín

### `optimizers/`

Implementačné jadro algoritmov.

* `optimizers/laqn/` — implementácie algoritmov pre LAQN
* `optimizers/wind/` — implementácie algoritmov pre WindWake

### `experiments/`

Skripty pre spúšťanie experimentov.

* `experiments/laqn/singlerun/`
* `experiments/laqn/multirun/`
* `experiments/wind/singlerun/`
* `experiments/wind/multirun/`

### `results/`

Výstupy experimentov vo forme:

* summary JSON
* detailné per-problem JSON
* CSV pre IOHanalyzer
* pomocné porovnávacie súbory

---

## Odporúčaná príprava virutálneho prostredia

Projekt používa **oddelené virtuálne prostredia** pre úlohy `LAQN` a `WindWake`.

---

## Prostredie pre LAQN

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-laqn.txt
```

---

## Prostredie pre WindWake

```bash
python3 -m venv .venv_wind
source .venv_wind/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-wind.txt
python scripts/apply_pysot_patches.py
```

### Poznámka k `pySOT`

Pre wind úlohu sa používa knižnica `pySOT`, pri ktorej bolo potrebné opraviť viacero kompatibilitných problémov voči novším verziám `NumPy` a pri výpočte niektorých akvizičných / merit funkcií.

Tieto opravy sú automatizované skriptom:

```bash
python utils/apply_pysot_patches.py
```

Tento krok je potrebné vykonať po inštalácii `requirements-wind.txt` vo virtuálnom prostredí `.venv_wind`.

---

## Spúšťanie testovacích behov

Testovacie skripty slúžia na rýchle overenie správnosti implementácie jedného algoritmu na jednej úlohe.

### LAQN

```bash
python3 -m experiments.laqn.singlerun.run_random_search_laqn_one
```
```bash
python3 -m experiments.laqn.singlerun.run_turbo_laqn_one
```
```bash
python3 -m experiments.laqn.singlerun.run_pybads_laqn_one
```

### WindWake

```bash
python3 -m experiments.wind.singlerun.run_random_search_wind_one
```
```bash
python3 -m experiments.wind.singlerun.run_turbo_wind_one
```
```bash
python3 -m experiments.wind.singlerun.run_pybads_wind_one
```

Konkrétna kombinácia náhradného modelu a stratégie pre selekciu ďalších bodov algoritmu `pySOT` sa nastavuje priamo v `run_pysot_wind_one.py` pomocou:

* `surrogate_type`
* `strategy_type`

```bash
python3 -m experiments.wind.singlerun.run_pysot_wind_one
```
---

## Multirun experimentov

Finálne experimenty sa vykonávajú pomocou batch skriptov. Výstupný priečinok je pomenovaný hlavne podľa budget a n_runs, ale samotný priebeh experimentu závisí aj od ďalších parametrov. Tie možno rozdeliť na parametre batch experimentu, parametre algoritmu a parametre úlohy.

### Parametre batch experimentu

Tieto parametre riadia rozsah experimentu bez ohľadu na použitý algoritmus.

* `budget` — určuje maximálny počet evaluácií cieľovej funkcie v jednom behu.
* `n_runs` — určuje počet nezávislých behov pre rovnakú konfiguráciu.
* `seed`   — určuje inicializáciu náhodnosti naprieč behovými spusteniami.

Nastavujú sa v batch skriptoch:

* `experiments/laqn/multirun/`
* `experiments/wind/multirun/`

### Parametre algoritmu

Tieto parametre ovplyvňujú spôsob, akým algoritmus navrhuje a hodnotí nové body.

* `n_init`           — určuje veľkosť počiatočnej vzorky.
* `batch_size`       — určuje počet bodov navrhovaných naraz.
* `use_ard`          — určuje, či sa pri GP modeluje samostatná citlivosť každého rozmeru.
* `n_training_steps` — určuje počet krokov tréningu surrogate modelu.
* `surrogate_type`   — pri pySOT určuje typ náhradného modelu.
* `strategy_type`    — pri pySOT určuje stratégiu výberu ďalšieho bodu.
* `num_cand`         — určuje počet kandidátskych bodov pri kandidátových stratégiách.
* `use_restarts`     — určuje, či sa stratégia môže znovu inicializovať.
* `asynchronous`     — určuje, či stratégia pracuje asynchrónne.

Kde sa nastavujú:

* `optimizers/laqn/`
* `optimizers/wind/`

čiastočne aj vo volajúcich skriptoch v `experiments/...`

* `TuRBO`         → turbo_laqn.py, turbo_wind.py
* `PyBADS`        → pybads_laqn.py, pybads_wind.py
* `pySOT`         → pysot_wind.py
* `Random Search` → random_search_laqn.py, random_search_wind.py

### Parametre úlohy

Tieto parametre ovplyvňujú samotnú definíciu riešeného problému.

`LAQN`

výber problémových inštancií — určuje, nad akými úlohami sa experiment vykonáva
`include_initial`            — určuje, či sa použijú počiatočné body zo samotnej inštancie

Kde sa nastavujú:

výber inštancií v `experiments/laqn/singlerun/` a `experiments/laqn/multirun/`
logika počiatočných bodov najmä v `optimizers/laqn/`

Spúšťanie skritpov:

```bash
python3 -m experiments.laqn.multirun.run_random_search_laqn_batch
```
```bash
python3 -m experiments.laqn.multirun.run_turbo_laqn_batch
```
```bash
python3 -m experiments.laqn.multirun.run_pybads_laqn_batch
```

`WindWake`

* `n_turbines` — určuje počet turbín a tým aj dimenziu problému
* `wind_seed`  — určuje generovanie veterných podmienok
* `n_samples`  — určuje počet opakovaných vzoriek pri evaluácii

Kde sa nastavujú:

v `optimizers/wind/`
a konkrétne hodnoty vo volajúcich skriptoch v `experiments/wind/singlerun/` a `experiments/wind/multirun/`


Spúšťanie skritpov:

```bash
python3 -m experiments.wind.multirun.run_random_search_wind_batch
```
```bash
python3 -m experiments.wind.multirun.run_turbo_wind_batch
```
```bash
python3 -m experiments.wind.multirun.run_pybads_wind_batch
```

Konkrétna kombinácia náhradného modelu a stratégie pre selekciu ďalších bodov algoritmu `pySOT` sa nastavuje priamo v `run_pysot_wind_batch.py` pomocou:

* `surrogate_type`
* `strategy_type`

```bash
python3 -m experiments.wind.multirun.run_pysot_wind_batch
```

---

## Výstupy experimentov

Výstupy sa ukladajú podľa experimentálneho nastavenia:

results/<task>/final/budget<budget>_runs<n_runs>/

Výsledky jednotlivých pySOT kombinácií sa ukladajú do samostatných priečinkov, napríklad:

* `pysot_rbf_dycors`
* `pysot_gp_ei`
* `pysot_gp_lcb`

Každý algoritmus vytvára:

### 1. Summary JSON

Agregované výsledky metódy cez všetky behy alebo problémy.

### 2. Per-problem JSON

Detailné výsledky jednotlivých behov.

### 3. IOHanalyzer CSV

CSV výstup s priebehom evaluácií pre vizualizáciu a porovnanie.

---

## IOHanalyzer

CSV výstupy je možné nahrať do webového rozhrania **IOHanalyzer**.

Mapovanie stĺpcov:

* `evaluation`    → Evaluation counter
* `best_so_far_y` → Function values
* `problem_id`    → Function ID
* `algorithm_id`  → Algorithm ID
* `dimension`     → Problem dimension
* `run_id`        → Run ID

---

## Porovnanie výsledkov

Pre LAQN sú v projekte aj pomocné skripty na tvorbu porovnávacích tabuliek medzi algoritmami:

```bash
python3 results/laqn/build_comparison_summary.py <budget> <n_runs>
python3 results/laqn/build_per_problem_comparison.py <budget> <n_runs>
```

Tieto skripty vytvárajú:

* globálne porovnanie algoritmov,
* porovnanie po jednotlivých problémových inštanciách.

---

## Použité externé repozitáre a frameworky

* **PyBADS**  — Bayesian Adaptive Direct Search
* **TuRBO**   — trust-region Bayesian optimization
* **pySOT**   — Surrogate Optimization Toolbox
* **LAQN-BO** — rámec na generovanie a spracovanie LAQN úloh
* **FLORIS**  — simulácia pre úlohu rozmiestnenia veterných turbín

---

## Stav projektu

Projekt obsahuje:

* implementáciu algoritmov pre obe úlohy,
* jednotný experimentálny rámec,
* export výstupov do JSON a CSV,
* základ pre ďalšie porovnanie algoritmov v diplomovej práci.