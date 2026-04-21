from __future__ import annotations

import json
from pathlib import Path

from optimizers.wind.pysot_wind import (
    make_windwake_problem,
    run_pysot_wind,
)


def main():
    file_path = Path("data/wind/windwake_input.json")

    n_turbines = 3
    wind_seed = 0
    n_samples = 5

    budget = 20
    seed = 0
    run_id = 1

    surrogate_type = "gp"
    strategy_type = "lcb"

    n_init = 10
    batch_size = 1
    num_cand = 1000
    use_restarts = True
    asynchronous = True
    verbose = True
    extra_config = None

    algo_tag = f"pysot_{surrogate_type}_{strategy_type}"

    out_dir = Path("results/wind/singlerun") / algo_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    problem = make_windwake_problem(
        file_path=file_path,
        n_turbines=n_turbines,
        wind_seed=wind_seed,
        n_samples=n_samples,
    )

    result = run_pysot_wind(
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
            "file": str(file_path),
            "n_turbines": n_turbines,
            "wind_seed": wind_seed,
            "n_samples": n_samples,
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
        f"run_pysot_wind_one_{surrogate_type}_{strategy_type}"
        f"_n{n_turbines}_budget{budget}_seed{seed}.json"
    )

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\n=== PYSOT WIND | SINGLE RUN ===")
    print(f"algorithm_name:   {result.algorithm_name}")
    print(f"problem_id:       {result.problem_id}")
    print(f"dimension:        {result.dimension}")
    print(f"budget:           {result.budget}")
    print(f"call_count:       {result.call_count}")
    print(f"best_y:           {result.best_y:.6f}")
    print(f"evals_to_f_best:  {result.evals_to_f_best}")
    print(f"surrogate_type:   {result.surrogate_type}")
    print(f"strategy_type:    {result.strategy_type}")
    print(f"n_init:           {result.n_init}")
    print(f"batch_size:       {result.batch_size}")
    print(f"num_cand:         {result.num_cand}")
    print(f"total_time [s]:   {result.total_time:.6f}")
    print(f"saved_to:         {out_path}")


if __name__ == "__main__":
    main()