from __future__ import annotations

import time
import numpy as np
from turbo import Turbo1

from data.wind.problems.windwake import WindWakeLayout


def main():
    problem = WindWakeLayout(
        file="data/wind/windwake_input.json",
        n_turbines=3,
        wind_seed=0,
        n_samples=5,
    )

    lb = np.asarray(problem.lbs(), dtype=float)
    ub = np.asarray(problem.ubs(), dtype=float)
    dim = int(problem.dims())

    x_hist: list[list[float]] = []
    y_hist: list[float] = []
    best_so_far: list[float] = []

    best_y = np.inf
    best_x = None

    def objective(x):
        nonlocal best_y, best_x

        x = np.asarray(x, dtype=float).reshape(-1)
        y = float(problem.evaluate(x))

        x_hist.append(x.tolist())
        y_hist.append(y)

        if y < best_y:
            best_y = y
            best_x = x.copy()

        best_so_far.append(float(best_y))
        print(f"eval {len(y_hist):02d}: y = {y:.6f}")
        return y

    start = time.perf_counter()

    turbo = Turbo1(
        f=objective,
        lb=lb,
        ub=ub,
        n_init=5,
        max_evals=20,
        batch_size=1,
        verbose=True,
        use_ard=True,
        max_cholesky_size=2000,
        n_training_steps=50,
        min_cuda=1024,
        device="cpu",
        dtype="float64",
    )

    turbo.optimize()
    total_time = time.perf_counter() - start

    print("\n=== TURBO WIND TEST ===")
    print("dimension:", dim)
    print("calls:", len(y_hist))
    print("best_x:", best_x)
    print("best_y:", best_y)
    print("curve_length:", len(best_so_far))
    print("total_time [s]:", total_time)


if __name__ == "__main__":
    main()