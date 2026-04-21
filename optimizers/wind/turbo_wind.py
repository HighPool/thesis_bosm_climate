from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import time
import numpy as np

from turbo import Turbo1
from data.wind.problems.windwake import WindWakeLayout


@dataclass
class TurboWindResult:
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

    n_init: int
    batch_size: int
    use_ard: bool
    n_training_steps: int
    device: str
    dtype: str

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

    return WindWakeLayout(
        file=file_path,
        n_turbines=n_turbines,
        wind_seed=wind_seed,
        n_samples=n_samples,
    )


def run_turbo_wind(
    problem,
    budget: int = 20,
    seed: int = 0,
    run_id: int | None = None,
    n_init: int = 5,
    batch_size: int = 1,
    use_ard: bool = True,
    n_training_steps: int = 50,
    device: str = "cpu",
    dtype: str = "float64",
    verbose: bool = True,
) -> TurboWindResult:
    if budget <= 0:
        raise ValueError("budget musí byť kladný")
    if n_init <= 0:
        raise ValueError("n_init musí byť kladné")
    if n_init > budget:
        raise ValueError("n_init nesmie byť väčšie ako budget")

    start_total = time.perf_counter()
    np.random.seed(seed)

    lb = np.asarray(problem.lbs(), dtype=float)
    ub = np.asarray(problem.ubs(), dtype=float)
    dim = int(problem.dims())

    x_hist: list[list[float]] = []
    y_hist: list[float] = []
    best_so_far: list[float] = []

    best_y = np.inf
    best_x_arr: np.ndarray | None = None

    def objective(x):
        nonlocal best_y, best_x_arr

        x = np.asarray(x, dtype=float).reshape(-1)
        y = float(problem.evaluate(x))

        x_hist.append(x.astype(float).tolist())
        y_hist.append(y)

        if y < best_y:
            best_y = y
            best_x_arr = x.copy()

        best_so_far.append(float(best_y))
        return y

    turbo = Turbo1(
        f=objective,
        lb=lb,
        ub=ub,
        n_init=int(n_init),
        max_evals=int(budget),
        batch_size=int(batch_size),
        verbose=bool(verbose),
        use_ard=bool(use_ard),
        max_cholesky_size=2000,
        n_training_steps=int(n_training_steps),
        min_cuda=1024,
        device=device,
        dtype=dtype,
    )

    turbo.optimize()
    total_time = time.perf_counter() - start_total

    if best_x_arr is None:
        raise RuntimeError("Nepodarilo sa určiť best_x")

    final_best = best_so_far[-1]
    evals_to_f_best = next(
        i + 1 for i, v in enumerate(best_so_far)
        if np.isclose(v, final_best)
    )

    problem_id = f"windwake_n{problem.n_turbines}_wseed{problem.wind_seed}"

    return TurboWindResult(
        algorithm_name="TuRBO",
        problem_id=problem_id,
        dimension=dim,
        run_id=run_id,
        X_hist=x_hist,
        y_hist=y_hist,
        best_so_far=best_so_far,
        best_x=best_x_arr.astype(float).tolist(),
        best_y=float(best_y),
        budget=int(budget),
        call_count=len(y_hist),
        evals_to_f_best=int(evals_to_f_best),
        total_time=float(total_time),
        seed=int(seed),
        file=str(problem.file),
        n_turbines=int(problem.n_turbines),
        wind_seed=int(problem.wind_seed),
        n_samples=None if problem.n_samples is None else int(problem.n_samples),
        lower_bounds=lb.astype(float).tolist(),
        upper_bounds=ub.astype(float).tolist(),
        n_init=int(n_init),
        batch_size=int(batch_size),
        use_ard=bool(use_ard),
        n_training_steps=int(n_training_steps),
        device=str(device),
        dtype=str(dtype),
    )