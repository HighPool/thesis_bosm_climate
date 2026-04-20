from __future__ import annotations

import time
import numpy as np

from data.wind.problems.windwake import WindWakeLayout
from pybads import BADS


def main():
    problem = WindWakeLayout(
        file="data/wind/windwake_input.json",
        n_turbines=3,
        wind_seed=0,
        n_samples=5,
    )

    lb = np.asarray(problem.lbs(), dtype=float)
    ub = np.asarray(problem.ubs(), dtype=float)
    dim = problem.dims()

    # počiatočný bod v strede priestoru
    x0 = 0.5 * (lb + ub)

    call_counter = {"n": 0}

    def objective(x):
        call_counter["n"] += 1
        y = float(problem.evaluate(x))
        print(f"eval {call_counter['n']:02d}: y = {y:.6f}")
        return y

    options = {
        "max_fun_evals": 20,
        "display": "iter",
    }

    start = time.perf_counter()

    bads = BADS(
        fun=objective,
        x0=x0,
        lower_bounds=lb,
        upper_bounds=ub,
        options=options,
    )

    result = bads.optimize()

    total_time = time.perf_counter() - start

    print("\n=== PYBADS WIND TEST ===")
    print("dimension:", dim)
    print("calls:", call_counter["n"])
    print("best_x:", result.x)
    print("best_y:", result.fval)
    print("total_time [s]:", total_time)
    print("result keys:", result.keys() if hasattr(result, "keys") else type(result))


if __name__ == "__main__":
    main()