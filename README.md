### Cieľ

Implementovať **baseline metódu (random search)** pre úlohy LAQN-BO a získať:

* finálne metriky (regret, success rate),
* konvergenčné krivky,
* reprodukovateľné experimenty.

---

## Štruktúra skriptov

### `batch_random_search_laqn.py`

* Implementuje **random search pre všetky vygenerované LAQN úlohy (multi run)**.
* Náhodne vyberá body z `problem.domain` a hodnoty z `problem.labels`.

* Vracia:
  * históriu (`X_hist`, `y_hist`),
  * `best_hist` (best-so-far),
  * finálne riešenie (`best_x`, `best_y`).
* Pre každý problém:
  * vypočíta odchylku,
  * zistí, či bolo nájdené optimum.

* Ukladá výsledky do JSON.

---

### `batch_random_search_laqn_multirun.py`

* Rozšírenie na **viac behov (n_runs)** → stabilný experiment.
* Pre každý problém počíta:
  * mean / median / std odchylku,
  * success rate,
  * priemernú konvergenčnú krivku.

* Vytvára:

  * `results_random_search_2015_multirun.json`
  * baseline výsledok.

---

### `plot_random_search_convergence.py`

* Načíta JSON výsledky.
* Vypočíta:
  * globálnu mean a median krivku.

* Ukladá graf do:
  * `results/random_search_convergence.png`
  * `results/random_search_convergence.svg`

---

### `run_random_search_pipeline.py`

* **Pipeline**:

  1. spustí experiment (multi-run),
  2. následne vykreslí graf.
* Umožňuje spustiť celý workflow jedným príkazom.

---

## Dátová časť

* Použité sú **vygenerované LAQN problémy** (`.p` súbory).
* Pre správne načítanie je potrebný súbor:

  * `setup_helper.py`

---

## Spustenie

```bash
python3 -m experiments.run_random_search_pipeline
```

---

## Výstupy

* JSON výsledky (metriky)
* PNG/SVG graf konvergencie
