from __future__ import annotations

from pathlib import Path

from optimizers.wind.pysot_wind import (
    make_windwake_problem,
    run_pysot_wind,
)


def main():
    file_path = Path("data/wind/windwake_input.json")

    problem = make_windwake_problem(
        file_path=file_path,
        n_turbines=3,
        wind_seed=0,
        n_samples=5,
    )

    result = run_pysot_wind(
        problem=problem,
        surrogate_type="rbf",
        strategy_type="dycors",
        budget=20,
        seed=0,
        run_id=1,
        n_init=10,
        batch_size=1,
        num_cand=1000,
        use_restarts=True,
        asynchronous=True,
        verbose=True,
        extra_config=None,
    )

    print("\n=== PYSOT WIND TEST ===")
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
    print(f"len(X_hist):      {len(result.X_hist)}")
    print(f"len(y_hist):      {len(result.y_hist)}")
    print(f"len(best_so_far): {len(result.best_so_far)}")


if __name__ == "__main__":
    main()