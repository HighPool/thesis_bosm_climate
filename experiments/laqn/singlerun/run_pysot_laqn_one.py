from __future__ import annotations

import json
from pathlib import Path

from optimizers.laqn.pysot_laqn import (
    load_problem,
    run_pysot_laqn,
)

print("run_pysot_laqn_one.py loaded", flush=True)

def main():
    print("main() entered", flush=True)
    print("Vyhľadávam prvú dostupnú LAQN inštanciu...", flush=True)
    problem_path = next(Path("data/laqn/2015/preprocessed").glob("*.p"))
    print(f"Načítavam problém: {problem_path}", flush=True)

    problem = load_problem(problem_path)

    budget = 20
    seed = 0
    run_id = 1

    surrogate_type = "rbf"
    strategy_type = "dycors"

    n_init = 10
    batch_size = 1
    num_cand = 1000
    use_restarts = True
    asynchronous = True
    verbose = True
    extra_config = None

    algo_tag = f"pysot_{surrogate_type}_{strategy_type}"

    out_dir = Path("results/laqn/singlerun") / algo_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== PYSOT LAQN | SINGLE RUN | START ===", flush=True)
    print(f"problem_id:             {problem.identifier}", flush=True)
    print(f"dimension:              {problem.domain.shape[1]}", flush=True)
    print(f"n_domain_points:        {problem.domain.shape[0]}", flush=True)
    print(f"budget:                 {budget}", flush=True)
    print(f"seed:                   {seed}", flush=True)
    print(f"surrogate_type:         {surrogate_type}", flush=True)
    print(f"strategy_type:          {strategy_type}", flush=True)
    print(f"n_init:                 {n_init}", flush=True)
    print(f"batch_size:             {batch_size}", flush=True)
    print(f"num_cand:               {num_cand}", flush=True)
    print("\nSpúšťam optimalizáciu...", flush=True)

    result = run_pysot_laqn(
        problem=problem,
        surrogate_type=surrogate_type,
        strategy_type=strategy_type,
        budget=budget,
        seed=seed,
        run_id=run_id,
        n_init=n_init,
        batch_size=batch_size,
        num_cand=num_cand,
        use_restarts=use_restarts,
        asynchronous=asynchronous,
        verbose=verbose,
        extra_config=extra_config,
    )

    payload = {
        "config": {
            "algorithm_name": result.algorithm_name,
            "problem_path": str(problem_path),
            "budget": budget,
            "seed": seed,
            "counting_mode": "algorithm_calls",
            "surrogate_type": surrogate_type,
            "strategy_type": strategy_type,
            "n_init": n_init,
            "batch_size": batch_size,
            "num_cand": num_cand,
            "use_restarts": use_restarts,
            "asynchronous": asynchronous,
        },
        "result": result.to_dict(),
    }

    out_path = out_dir / (
        f"run_pysot_laqn_one_{surrogate_type}_{strategy_type}"
        f"_{problem.identifier}_budget{budget}_seed{seed}.json"
    )

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("\n=== PYSOT LAQN | SINGLE RUN | DONE ===")
    print(f"algorithm_name:         {result.algorithm_name}")
    print(f"problem_id:             {result.problem_id}")
    print(f"dimension:              {result.dimension}")
    print(f"run_id:                 {result.run_id}")
    print(f"budget:                 {result.budget}")
    print(f"seed:                   {seed}")

    print("\n----- RESULT QUALITY -----")
    print(f"best_y:                 {result.best_y}")
    print(f"optimum:                {result.optimum}")
    print(f"deviation:              {result.deviation_from_optimum}")
    print(f"success:                {result.success}")

    print("\n----- SEARCH PROCESS -----")
    print(f"call_count:             {result.call_count}")
    print(f"unique_eval_count:      {result.unique_eval_count}")
    print(f"evals_to_f_best:        {result.evals_to_f_best}")
    print(f"curve_length:           {len(result.best_so_far)}")

    print("\n----- PYSOT CONFIG -----")
    print(f"surrogate_type:         {result.surrogate_type}")
    print(f"strategy_type:          {result.strategy_type}")
    print(f"n_init:                 {result.n_init}")
    print(f"batch_size:             {result.batch_size}")
    print(f"num_cand:               {result.num_cand}")

    print("\n----- TIME -----")
    print(f"total_time_seconds:     {result.total_time:.6f}")

    print("\n----- BEST SOLUTION -----")
    print(f"best_x:                 {result.best_x}")
    print(f"optimum_x:              {result.optimum_x}")

    print("\n----- HISTORY CHECK -----")
    print(f"len(x_hist):            {len(result.x_hist)}")
    print(f"len(y_hist):            {len(result.y_hist)}")
    print(f"len(best_so_far):       {len(result.best_so_far)}")

    print(f"\nsaved_to:               {out_path}")


if __name__ == "__main__":
    main()