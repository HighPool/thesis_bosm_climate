from __future__ import annotations
import logging
import io
import os
import pickle
import time
import warnings
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
from pybads import BADS

class BudgetReached(Exception):
    pass

@dataclass
class PyBADSLAQNResult:
    identifier: str
    best_x: list[float]
    best_y: float
    best_so_far: list[float]
    x_hist: list[list[float]]
    y_hist: list[float]
    eval_count: int
    evals_to_f_best: int
    total_time: float
    deviation_from_optimum: float
    optimum: float
    optimum_x: list[float]
    success: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

def load_problem(problem_path: str | Path):
    problem_path = Path(problem_path)
    with problem_path.open("rb") as f:
        return pickle.load(f)

class LAQNPyBADSObjective:
    """
    pyBADS navrhuje spojitý 2D bod.
    My ho priradíme k najbližšiemu bodu z problem.domain.
    Hodnotu zoberieme z problem.labels.
    Keďže pyBADS minimalizuje, vraciame -y.
    """

    def __init__(
        self,
        problem,
        total_budget: int = 1000,
        include_initial_points: bool = True,
    ):
        self.problem = problem
        self.total_budget = int(total_budget)

        self.domain = np.asarray(problem.domain, dtype=float)
        self.labels = np.asarray(problem.labels, dtype=float).reshape(-1)

        if self.domain.ndim != 2:
            raise ValueError(f"problem.domain musí byť 2D pole, shape={self.domain.shape}")
        if self.domain.shape[1] != 2:
            raise ValueError(f"Očakávam 2D problém, ale domain má shape={self.domain.shape}")
        if len(self.domain) != len(self.labels):
            raise ValueError("Počet bodov v domain a labels sa nezhoduje")

        self.tree = cKDTree(self.domain)

        # cache: index bodu v domain -> objective value
        self.cache: dict[int, float] = {}

        self.x_hist: list[np.ndarray] = []
        self.y_hist: list[float] = []
        self.best_so_far: list[float] = []

        if include_initial_points:
            self._load_initial_points()

    def _load_initial_points(self) -> None:
        xx0 = np.asarray(self.problem.xx, dtype=float)
        yy0 = np.asarray(self.problem.yy, dtype=float).reshape(-1)

        if len(xx0) != len(yy0):
            raise ValueError("Počet xx a yy sa nezhoduje")

        for x0, y0 in zip(xx0, yy0):
            idx = self._snap_to_index(x0)

            if idx in self.cache:
                continue

            y0 = float(y0)
            self.cache[idx] = y0
            self.x_hist.append(self.domain[idx].copy())
            self.y_hist.append(y0)

            if not self.best_so_far:
                self.best_so_far.append(y0)
            else:
                self.best_so_far.append(max(self.best_so_far[-1], y0))

    @property
    def eval_count(self) -> int:
        return len(self.cache)

    def _snap_to_index(self, x: np.ndarray) -> int:
        x = np.asarray(x, dtype=float).reshape(-1)
        _, idx = self.tree.query(x, k=1)
        return int(idx)

    def sample_unseen_point(self) -> np.ndarray:
        """
        Náhodne vyberie ešte nevyhodnotený bod z domain.
        Ak už sú všetky body vyhodnotené, vyhodí chybu.
        """
        unseen_indices = [i for i in range(len(self.domain)) if i not in self.cache]
        if not unseen_indices:
            raise BudgetReached("Všetky body z domain už boli vyhodnotené.")

        idx = int(np.random.choice(unseen_indices))
        return self.domain[idx].copy()

    def __call__(self, x: np.ndarray) -> float:
        idx = self._snap_to_index(x)

        if idx in self.cache:
            return -self.cache[idx]

        if self.eval_count >= self.total_budget:
            raise BudgetReached("Dosiahnutý limit unikátnych evaluácií.")

        y = float(self.labels[idx])
        self.cache[idx] = y

        self.x_hist.append(self.domain[idx].copy())
        self.y_hist.append(y)

        if not self.best_so_far:
            self.best_so_far.append(y)
        else:
            self.best_so_far.append(max(self.best_so_far[-1], y))

        return -y

def _build_and_run_bads_silently(
    objective,
    x0,
    lb,
    ub,
    plb,
    pub,
    options,
):
    """
    Vytvorí aj spustí BADS bez zahlcujúcich výpisov do konzoly.
    Umlčí stdout, stderr, warnings aj logging.
    """
    sink = io.StringIO()

    previous_disable = logging.root.manager.disable

    try:
        logging.disable(logging.CRITICAL)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            with redirect_stdout(sink), redirect_stderr(sink):
                bads = BADS(
                    objective,
                    x0=x0,
                    lower_bounds=lb,
                    upper_bounds=ub,
                    plausible_lower_bounds=plb,
                    plausible_upper_bounds=pub,
                    options=options,
                )
                return bads.optimize()

    finally:
        logging.disable(previous_disable)

def run_pybads_on_problem(
    problem,
    total_budget: int = 1000,
    random_seed: int | None = None,
    display: str = "off",
) -> PyBADSLAQNResult:
    start_total = time.perf_counter()

    if random_seed is not None:
        np.random.seed(random_seed)

    domain = np.asarray(problem.domain, dtype=float)
    xx0 = np.asarray(problem.xx, dtype=float)
    yy0 = np.asarray(problem.yy, dtype=float).reshape(-1)

    best_init_idx = int(np.argmax(yy0))
    x0 = xx0[best_init_idx].astype(float)

    lb = domain.min(axis=0)
    ub = domain.max(axis=0)

    plb = np.quantile(domain, 0.10, axis=0)
    pub = np.quantile(domain, 0.90, axis=0)

    objective = LAQNPyBADSObjective(
        problem=problem,
        total_budget=total_budget,
        include_initial_points=True,
    )

    restart_id = 0
    max_restarts = 50

    while objective.eval_count < total_budget and restart_id < max_restarts:
        restart_id += 1

        # display necháme kvôli konzistencii v options, ale výpisy aj tak potlačíme
        local_display = display if restart_id == 1 else "off"

        options = {
            "display": local_display,
            "max_fun_evals": 100000,
        }

        try:
            _build_and_run_bads_silently(
                objective=objective,
                x0=x0,
                lb=lb,
                ub=ub,
                plb=plb,
                pub=pub,
                options=options,
            )
        except BudgetReached:
            break

        try:
            x0 = objective.sample_unseen_point()
        except BudgetReached:
            break

    if not objective.y_hist:
        raise RuntimeError("Nevznikla žiadna história evaluácií.")

    best_idx = int(np.argmax(objective.y_hist))
    best_x = np.asarray(objective.x_hist[best_idx], dtype=float)
    best_y = float(objective.y_hist[best_idx])

    optimum = float(problem.maximum)
    optimum_x = np.asarray(problem.maximiser, dtype=float)
    deviation = float(optimum - best_y)
    success = bool(np.isclose(best_y, optimum))

    final_best = objective.best_so_far[-1]
    evals_to_f_best = next(
        i + 1 for i, v in enumerate(objective.best_so_far)
        if np.isclose(v, final_best)
    )

    total_time = time.perf_counter() - start_total

    return PyBADSLAQNResult(
        identifier=str(problem.identifier),
        best_x=best_x.tolist(),
        best_y=best_y,
        best_so_far=[float(v) for v in objective.best_so_far],
        x_hist=[np.asarray(x, dtype=float).tolist() for x in objective.x_hist],
        y_hist=[float(v) for v in objective.y_hist],
        eval_count=objective.eval_count,
        evals_to_f_best=evals_to_f_best,
        total_time=total_time,
        deviation_from_optimum=deviation,
        optimum=optimum,
        optimum_x=optimum_x.tolist(),
        success=success,
    )