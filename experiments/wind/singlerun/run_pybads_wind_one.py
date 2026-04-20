from __future__ import annotations

import json
from pathlib import Path

from optimizers.wind.pybads_wind import (
    make_windwake_problem,
    run_pybads_wind,
)


def main():
    file_path = Path("data/wind/windwake_input.json")

    n_turbines = 3
    wind_seed = 0
    n_samples = 5

    budget = 20
    seed = 0
    run_id = 1

    out_dir = Path("results/wind/singlerun/pybads")
    out_dir.mkdir(parents=True, exist_ok=True)

    problem = make_windwake_problem(
        file_path=file_path,
        n_turbines=n_turbines,
        wind_seed=wind_seed,
        n_samples=n_samples,
    )

    result = run_pybads_wind(
        problem=problem,
        budget=budget,
        seed=seed,
        run_id=run_id,
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
        },
        "result": result.to_dict(),
    }

    out_path = out_dir / f"run_pybads_wind_one_n{n_turbines}_budget{budget}_seed{seed}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("\n=== PYBADS WIND | SINGLE RUN ===")
    print(f"problem_id:       {result.problem_id}")
    print(f"dimension:        {result.dimension}")
    print(f"budget:           {result.budget}")
    print(f"call_count:       {result.call_count}")
    print(f"best_y:           {result.best_y:.6f}")
    print(f"evals_to_f_best:  {result.evals_to_f_best}")
    print(f"iterations:       {result.iterations}")
    print(f"success:          {result.success}")
    print(f"message:          {result.message}")
    print(f"total_time [s]:   {result.total_time:.6f}")
    print(f"saved_to:         {out_path}")


if __name__ == "__main__":
    main()