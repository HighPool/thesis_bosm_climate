# BOSM Climate

Projekt k diplomovej práci zameranej na optimalizáciu nákladných black-box úloh s využitím náhradných modelov v klimaticky orientovaných aplikáciách. Aktuálna implementácia je zameraná na úlohu rozmiestnenia environmentálnych senzorov odvodenú z dát LAQN a na porovnanie viacerých optimalizačných metód v jednotnom experimentálnom rámci.

## Obsah projektu

Projekt obsahuje implementácie a experimentálne skripty pre tieto algoritmy:

* TuRBO
* PyBADS
* Random Search

Výstupy experimentov sú ukladané tak, aby bolo možné:

* porovnávať každý algoritmus medzi sebou,
* analyzovať výsledky po problémoch aj globálne,
* a exportovať priebehy behu každého algoritmu do CSV formátu podľa požadovaného formátu pre IOHanalyzer.

Použitý algoritmus PyBADS predstavuje Python implementáciu Bayesian Adaptive Direct Search a repozitár TuRBO poskytuje implementáciu trust-region Bayesian optimization. Úlohy LAQN sú odvodené z rámca LAQN-BO, ktorý slúži na generovanie a spracovanie problémových inštancií z dát London Air Quality Network.

## Štruktúra adresára

### `optimizers/`

Obsahuje implementačné jadro jednotlivých optimalizačných algoritmov pre úlohy LAQN.

`turbo_laqn.py`
`pybads_laqn.py`
`random_search_laqn.py`

### `experiments/singlerun/`

Obsahuje testovacie skripty pre jeden beh algoritmu na jednej problémovej inštancii.
Tieto skripty slúžia len na rýchle overenie správnosti implementácie a vypisujú metriky priamo do terminálu.

### `experiments/multirun/`

Obsahuje hlavné experimentálne skripty pre dávkové vyhodnotenie algoritmov na celej sade LAQN úloh.

Tieto skripty:

* prechádzajú všetky problémové inštancie,
* vykonávajú viac behov každého algoritmu,
* agregujú metriky po jednotlivých úlohách aj globálne,
* ukladajú summary JSON,
* ukladajú detailné run-level JSON súbory po problémoch,
* a vytvárajú CSV výstup pre IOHanalyzer.

Každý multirun skript používa experimentálne nastavenie definované hodnotami `budget` a `n_runs`. Tieto dve hodnoty určujú nielen priebeh experimentu, ale aj výstupnú štruktúru výsledkov. Výsledky sa ukladajú do priečinka:

`results/laqn/final/budget<budget>_runs<n_runs>/`

V rámci tohto experimentálneho priečinka má každý algoritmus vlastný podpriečinok:

* `random_search/`
* `turbo/`
* `pybads/`

Príklad pre `budget = 10` a `n_runs = 20`:

`results/laqn/final/budget10_runs20/random_search/`
`results/laqn/final/budget10_runs20/turbo/`
`results/laqn/final/budget10_runs20/pybads/`

### `results/`

Obsahuje výsledky experimentov a pomocné skripty pre porovnanie výsledkov.

Hlavná logika ukladania je založená na experimentálnom nastavení:

* `budget`
* `n_runs`

Pre každú kombináciu týchto hodnôt sa v priečinku `results/laqn/final/` vytvorí samostatný experimentálny priečinok v tvare:

`budget<budget>_runs<n_runs>`

Príklad:

`results/laqn/final/budget10_runs20/`
`results/laqn/final/budget500_runs20/`
`results/laqn/final/budget1000_runs30/`

V rámci každého experimentálneho priečinka sa nachádzajú:

* podpriečinky jednotlivých algoritmov (`random_search`, `turbo`, `pybads`),
* priečinok `comparison_summary/` pre globálne porovnanie algoritmov,
* priečinok `per_problem_comparison/` pre porovnanie po jednotlivých problémových inštanciách.

Typická štruktúra vyzerá takto:

```text
results/laqn/final/budget10_runs20/
├── random_search/
│   ├── per_problem/
│   ├── random_search_summary_2015_budget10_runs20.json
│   └── random_search_ioh_2015_budget10_runs20.csv
├── turbo/
│   ├── per_problem/
│   ├── turbo_summary_2015_budget10_runs20.json
│   └── turbo_ioh_2015_budget10_runs20.csv
├── pybads/
│   ├── per_problem/
│   ├── pybads_summary_2015_budget10_runs20.json
│   └── pybads_ioh_2015_budget10_runs20.csv
├── comparison_summary/
│   ├── comparison_summary.json
│   └── comparison_summary.csv
└── per_problem_comparison/
    ├── per_problem_comparison.json
    └── per_problem_comparison.csv
```

Pomocné skripty:

* `build_comparison_summary.py` — vytvorenie globálneho porovnania algoritmov pre zvolené experimentálne nastavenie
* `build_per_problem_comparison.py` — vytvorenie porovnania po jednotlivých úlohách pre zvolené experimentálne nastavenie

### `data/laqn/2015/preprocessed/`

Obsahuje predspracované problémové inštancie LAQN vo formáte `.p`.

## Príprava prostredia

V koreňovom adresári projektu spustite:

```bash
python3 -m venv .venv && source .venv/bin/activate && python3 -m pip install --upgrade pip && python3 -m pip install -r requirements.txt
```

Tento príkaz pripraví prostredie, ale aktivácia `.venv` platí len pre daný shell.

## Spúšťanie testovacích skriptov

Tieto skripty vykonajú jeden beh a vypíšu metriky do terminálu.

### Random Search

```bash
python3 -m experiments.singlerun.run_random_search_laqn_one
```

### TuRBO

```bash
python3 -m experiments.singlerun.run_turbo_laqn_one
```

### pyBADS

```bash
python3 -m experiments.singlerun.run_pybads_laqn_one
```

## Spúšťanie multirun experimentov

Tieto skripty vykonávajú finálne experimenty nad všetkými problémovými inštanciami.

Každý skript má vo svojom vnútri definované experimentálne nastavenie `budget` a `n_runs`. Tieto hodnoty určujú priebeh experimentu aj názvy a umiestnenie výstupných súborov.

### Random Search

```bash
python3 -m experiments.multirun.run_random_search_laqn_batch
```

### TuRBO

```bash
python3 -m experiments.multirun.run_turbo_laqn_batch
```

### pyBADS

```bash
python3 -m experiments.multirun.run_pybads_laqn_batch
```

## Výstupy experimentov

Každý multirun skript vytvára výstupy, ktorých názvy aj umiestnenie zodpovedajú konkrétnemu experimentálnemu nastaveniu `budget` a `n_runs`.

Pre algoritmus Random Search a nastavenie `budget = 10`, `n_runs = 20` sa napríklad vytvoria súbory:

* `random_search_summary_2015_budget10_runs20.json`
* `random_search_ioh_2015_budget10_runs20.csv`

Analogicky sa pomenúvajú aj výstupy metód TuRBO a pyBADS.

Tieto súbory sa ukladajú do algoritmického podpriečinka patriaceho danému experimentu, napríklad:

`results/laqn/final/budget10_runs20/random_search/`

Každý multirun skript vytvára tri úrovne výstupov:

### 1. Summary JSON

Agregované výsledky celej metódy cez všetky problémy.

Obsahuje napríklad:

* priemernú odchýlku od optima,
* medián odchýlky,
* smerodajnú odchýlku,
* úspešnosť,
* priemerný počet evaluácií do najlepšieho riešenia,
* priemerný čas jedného behového spustenia,
* priemerný počet unikátne navštívených lokalít,
* celkový čas batch experimentu.

### 2. Per-problem run-level JSON

Detailné výsledky po jednotlivých problémoch a behoch.

Tieto súbory sa ukladajú do podpriečinka:

`per_problem/`

v rámci adresára konkrétneho algoritmu.

### 3. IOHanalyzer CSV

Run-level CSV súbor s priebehom evaluácií, vhodný na nahratie do IOHanalyzer.

## Porovnanie výsledkov medzi algoritmami

Po dobehnutí multirun experimentov je možné vytvoriť dve porovnávacie tabuľky pre konkrétne experimentálne nastavenie.

### Globálne porovnanie algoritmov

```bash
python3 results/laqn/build_comparison_summary.py <budget> <n_runs>
```

Príklad:

```bash
python3 results/laqn/build_comparison_summary.py 10 20
```

Výstupy sa uložia do:

* `results/laqn/final/budget10_runs20/comparison_summary/comparison_summary.json`
* `results/laqn/final/budget10_runs20/comparison_summary/comparison_summary.csv`

### Porovnanie po jednotlivých problémoch

```bash
python3 results/laqn/build_per_problem_comparison.py <budget> <n_runs>
```

Príklad:

```bash
python3 results/laqn/build_per_problem_comparison.py 10 20
```

Výstupy sa uložia do:

* `results/laqn/final/budget10_runs20/per_problem_comparison/per_problem_comparison.json`
* `results/laqn/final/budget10_runs20/per_problem_comparison/per_problem_comparison.csv`

## IOHanalyzer

CSV výstupy je možné nahrať do webového rozhrania IOHanalyzer namapovaním nasledujúcich stĺpcov:

* `evaluation`    → Evaluation counter
* `best_so_far_y` → Function values
* `problem_id`    → Function ID
* `algorithm_id`  → Algorithm ID
* `dimension`     → Problem dimension
* `run_id`        → Run ID

## Použité externé repozitáre

Pri implementácii a príprave experimentálneho rámca sa vychádzalo najmä z týchto externých repozitárov:

* **PyBADS** — Python implementácia algoritmu Bayesian Adaptive Direct Search
  Repozitár: `https://github.com/acerbilab/pybads`

* **TuRBO** — trust-region Bayesian optimization framework použitý ako základ implementácie metódy TuRBO
  Repozitár: `https://github.com/uber-research/TuRBO`

* **LAQN-BO** — rámec na spracovanie a generovanie problémových inštancií odvodených z dát London Air Quality Network
  Repozitár: `https://github.com/sighellan/LAQN-BO`