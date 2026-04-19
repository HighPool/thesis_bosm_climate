from __future__ import annotations
from pathlib import Path
from optimizers.laqn.random_search_laqn import load_problem, run_random_search_laqn

"""
Testovacie spustenie metódy Random Search na jednej LAQN úlohe
- načíta jednu problémovú inštanciu
- spustí jeden beh algoritmu
- vypíše do terminálu tieto metriky:
  - algorithm_name:     názov použitej optimalizačnej metódy
  - problem_id:         identifikátor riešenej problémovej inštancie
  - dimension:          počet rozmerov rozhodovacieho priestoru
  - run_id:             identifikátor konkrétneho behového opakovania
  - budget:             počet volaní algoritmu povolených v tomto behu
  - seed:               inicializačné semeno generátora náhodných čísel
  - include_initial:    informácia, či boli použité počiatočné body z problémovej inštancie
  - best_y:             najlepšia nájdená hodnota cieľovej funkcie
  - optimum:            skutočná optimálna hodnota cieľovej funkcie v danej úlohe
  - deviation:          rozdiel medzi optimom a najlepšou nájdenou hodnotou
  - success:            informácia, či algoritmus našiel optimum
  - call_count: celkový počet volaní algoritmu počas behového spustenia
  - unique_eval_count:  počet unikátne navštívených lokalít
  - evals_to_f_best:    počet volaní potrebných na prvé dosiahnutie finálne najlepšieho riešenia
  - curve_length:       dĺžka best-so-far krivky
  - total_time_seconds: celkový čas trvania jedného behového spustenia
  - best_x:             súradnice najlepšieho nájdeného riešenia
  - optimum_x:          súradnice skutočne optimálneho riešenia
  - len(X_hist):        počet uložených navštívených bodov
  - len(y_hist):        počet uložených hodnôt cieľovej funkcie
  - len(best_so_far):   počet bodov v konvergenčnej krivke

Slúži na rýchle overenie správnosti implementácie
"""

def main():
    # Načíta sa prvá dostupná inštancia problému z priečinka
    problem_file = next(Path("data/laqn/2015/preprocessed").glob("*.p"))
    problem = load_problem(problem_file)

    # Parametre testovacieho behového spustenia
    budget = 500
    seed = 0
    include_initial = True
    run_id = 1

    result = run_random_search_laqn(
        problem=problem,
        budget=budget,
        seed=seed,
        include_initial=include_initial,
        run_id=run_id,
    )

    print("\n===== RANDOM SEARCH | SINGLE-RUN TEST =====")
    print(f"algorithm_name:        {result.algorithm_name}")
    print(f"problem_id:            {result.problem_id}")
    print(f"dimension:             {result.dimension}")
    print(f"run_id:                {result.run_id}")
    print(f"budget:                {result.budget}")
    print(f"seed:                  {result.seed}")
    print(f"include_initial:       {include_initial}")

    print("\n----- RESULT QUALITY -----")
    print(f"best_y:                {result.best_y}")
    print(f"optimum:               {result.optimum}")
    print(f"deviation:             {result.deviation_from_optimum}")
    print(f"success:               {result.success}")

    print("\n----- SEARCH PROCESS -----")
    print(f"call_count:            {result.call_count}")
    print(f"unique_eval_count:     {result.unique_eval_count}")
    print(f"evals_to_f_best:       {result.evals_to_f_best}")
    print(f"curve_length:          {len(result.best_so_far)}")

    print("\n----- TIME -----")
    print(f"total_time_seconds:    {result.total_time:.6f}")

    print("\n----- BEST SOLUTION -----")
    print(f"best_x:                {result.best_x}")
    print(f"optimum_x:             {result.optimum_x}")

    print("\n----- HISTORY CHECK -----")
    print(f"len(X_hist):           {len(result.X_hist)}")
    print(f"len(y_hist):           {len(result.y_hist)}")
    print(f"len(best_so_far):      {len(result.best_so_far)}")

if __name__ == "__main__":
    main()