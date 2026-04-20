from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import time
import numpy as np

from data.wind.problems.windwake import WindWakeLayout


@dataclass
class RandomSearchWindResult:
    algorithm_name: str
    problem_id: str
    dimension: int
    run_id: int | None

    X_hist: list[list[float]]
    y_hist: list[float]
    best_so_far: list[float]
    best_x: list[float]
    best_y: float

    budget: int
    call_count: int
    evals_to_f_best: int
    total_time: float
    seed: int

    file: str
    n_turbines: int
    wind_seed: int
    n_samples: int | None
    lower_bounds: list[float]
    upper_bounds: list[float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_windwake_problem(
    file_path: str | Path,
    n_turbines: int = 3,
    wind_seed: int = 0,
    n_samples: int | None = 5,
):
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Vstupný JSON súbor neexistuje: {file_path}")

    problem = WindWakeLayout(
        file=file_path,
        n_turbines=n_turbines,
        wind_seed=wind_seed,
        n_samples=n_samples,
    )
    return problem


def run_random_search_wind(
    problem,
    budget: int = 20,
    seed: int = 0,
    run_id: int | None = None,
) -> RandomSearchWindResult:
    start_total = time.perf_counter()

    if budget <= 0:
        raise ValueError("budget musí byť kladný")

    rng = np.random.default_rng(seed)

    lb = np.asarray(problem.lbs(), dtype=float)
    ub = np.asarray(problem.ubs(), dtype=float)
    dim = int(problem.dims())

    if lb.shape != ub.shape or lb.shape[0] != dim:
        raise ValueError(
            f"Nekonzistentné rozmery bounds. lb={lb.shape}, ub={ub.shape}, dim={dim}"
        )

    x_hist: list[list[float]] = []
    y_hist: list[float] = []
    best_so_far: list[float] = []

    best_x_arr: np.ndarray | None = None
    best_y = np.inf  # minimalizujeme

    for _ in range(budget):
        x = rng.uniform(lb, ub, size=dim)
        y = float(problem.evaluate(x))

        x_hist.append(x.astype(float).tolist())
        y_hist.append(y)

        if y < best_y:
            best_y = y
            best_x_arr = x.copy()

        best_so_far.append(float(best_y))

    final_best = best_so_far[-1]
    evals_to_f_best = next(
        i + 1 for i, v in enumerate(best_so_far)
        if np.isclose(v, final_best)
    )

    total_time = time.perf_counter() - start_total

    if best_x_arr is None:
        raise RuntimeError("Nepodarilo sa určiť best_x")

    problem_id = f"windwake_n{problem.n_turbines}_wseed{problem.wind_seed}"

    return RandomSearchWindResult(
        algorithm_name="RandomSearch",
        problem_id=problem_id,
        dimension=dim,
        run_id=run_id,
        X_hist=x_hist,
        y_hist=y_hist,
        best_so_far=best_so_far,
        best_x=best_x_arr.astype(float).tolist(),
        best_y=float(best_y),
        budget=int(budget),
        call_count=int(budget),
        evals_to_f_best=int(evals_to_f_best),
        total_time=float(total_time),
        seed=int(seed),
        file=str(problem.file),
        n_turbines=int(problem.n_turbines),
        wind_seed=int(problem.wind_seed),
        n_samples=None if problem.n_samples is None else int(problem.n_samples),
        lower_bounds=lb.astype(float).tolist(),
        upper_bounds=ub.astype(float).tolist(),
    )