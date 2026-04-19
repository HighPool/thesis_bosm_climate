# BOSM Climate

Projekt k diplomovej práci zameranej na optimalizáciu nákladných black-box úloh s využitím náhradných modelov v klimaticky orientovaných aplikáciách. Aktuálna implementácia je zameraná na úlohu rozmiestnenia environmentálnych senzorov odvodenú z dát LAQN a na porovnanie viacerých optimalizačných metód v jednotnom experimentálnom rámci. Pri implementácii sa využívajú externé repozitáre PyBADS, TuRBO a LAQN-BO. :contentReference[oaicite:1]{index=1}

## Obsah projektu

Projekt obsahuje implementácie a experimentálne skripty pre tieto algoritmy:

- TuRBO
- PyBADS
- Random Search

Výstupy experimentov sú ukladané tak, aby bolo možné:
- porovnávať každý algoritmus medzi sebou,
- analyzovať výsledky po problémoch aj globálne,
- a exportovať priebehy behu každého algoritmu do CSV formátu podľa požadovaného formátu pre IOHanalyzer.

Použitý algoritmus PyBADS predstavuje Python implementáciu Bayesian Adaptive Direct Search a repozitár TuRBO poskytuje implementáciu trust-region Bayesian optimization. Úlohy LAQN sú odvodené z rámca LAQN-BO, ktorý slúži na generovanie a spracovanie problémových inštancií z dát London Air Quality Network. :contentReference[oaicite:2]{index=2}

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
Obsahuje hlavné experimentálne skripty. Tieto skripty:
- prechádzajú všetky problémové inštancie,
- vykonávajú viac behov každého algoritmu,
- agregujú metriky,
- ukladajú summary v JSON,
- ukladajú detail pre každý beh algoritmu v JSON,
- a vytvárajú CSV pre IOHanalyzer.

### `results/`
Obsahuje výsledky experimentov a pomocné skripty pre porovnanie výsledkov.

- `results/final/turbo/` — finálne výstupy TuRBO
- `results/final/random_search/` — finálne výstupy Random Search
- `results/final/pybads/` — finálne výstupy pyBADS
- `build_comparison_summary.py` — vytvorenie globálneho porovnania metód
- `build_per_problem_comparison.py` — vytvorenie porovnania po jednotlivých problémoch

### `data/laqn/2015/preprocessed/`
Obsahuje predspracované problémové inštancie LAQN vo formáte `.p`.


## Príprava prostredia
V koreňovom adresári projektu spustite:

```bash
python3 -m venv .venv && source .venv/bin/activate && python3 -m pip install --upgrade pip && python3 -m pip install -r requirements.txt
```
Tento príkaz pripraví prostredie, ale aktivácia .venv platí len pre daný shell.

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
Každý multirun skript vytvára tri úrovne výstupov:

1. Summary JSON
Agregované výsledky celej metódy cez všetky problémy.
Obsahuje:
- priemernú odchýlku od optima,
- medián odchýlky,
- úspešnosť,
- priemerný počet evaluácií do najlepšieho riešenia,
- priemerný čas jedného behového spustenia,
- priemerný počet unikátne navštívených lokalít,
- celkový čas batch experimentu.

2. Per-problem run-level JSON
- Detailné výsledky po jednotlivých problémoch a behoch.

3. IOHanalyzer CSV
- Run-level CSV súbor s priebehom evaluácií, vhodný na nahratie do IOHanalyzer.

## IOHanalyzer
CSV výstupy je možné nahrať do webového rozhrania IOHanalyzer namapovaním následujúcich stĺpcov:
evaluation    → Evaluation counter
best_so_far_y → Function values
problem_id    → Function ID
algorithm_id  → Algorithm ID
dimension     → Problem dimension
run_id        → Run ID