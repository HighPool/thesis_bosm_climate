from __future__ import annotations

"""
Implementácia metódy Random Search pre úlohy LAQN

Skript zabezpečuje:
- načítanie jednej problémovej inštancie LAQN
- výber bodov náhodným vyhľadávaním nad diskrétnou doménou
- sledovanie histórie jedného behového spustenia
- a výpočet metrík jedného experimentu
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import pickle
import time
import numpy as np
from scipy.spatial import cKDTree

@dataclass

# Výsledok jedného behu implementácie Random Search na jednej LAQN úlohe.
class RandomSearchLAQNResult:
    algorithm_name: str
    problem_id: str
    dimension: int
    run_id: int | None

    chosen_indices: list[int]
    X_hist: list[list[float]]
    y_hist: list[float]
    best_so_far: list[float]
    best_x: list[float]
    best_y: float

    budget: int
    call_count: int
    unique_eval_count: int

    evals_to_f_best: int
    total_time: float
    deviation_from_optimum: float
    optimum: float
    optimum_x: list[float]
    success: bool
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

# Načítanie jednej inštancie problému LAQN z .p súboru
def load_problem(path: str | Path):
    path = Path(path)
    with path.open("rb") as f:
        return pickle.load(f)

# Premietne počiatočné body z problem.xx na indexy najbližších bodov v doméne
def _snap_indices_for_initial_points(problem) -> list[int]:
    domain = np.asarray(problem.domain, dtype=float)
    xx = np.asarray(problem.xx, dtype=float)

    tree = cKDTree(domain)
    indices = []
    used = set()

    for x0 in xx:
        _, idx = tree.query(x0, k=1)
        idx = int(idx)
        if idx not in used:
            used.add(idx)
            indices.append(idx)

    return indices


def run_random_search_laqn(
    problem,
    budget: int = 500,
    seed: int = 0,
    include_initial: bool = True,
    run_id: int | None = None,
) -> RandomSearchLAQNResult:
    # Spustí jeden beh metódy Random Search na jednej LAQN úlohe
    
    start_total = time.perf_counter()

    if budget <= 0:
        raise ValueError("budget musí byť kladný")

    rng = np.random.default_rng(seed)

    domain = np.asarray(problem.domain, dtype=float)
    labels = np.asarray(problem.labels, dtype=float).reshape(-1)

    if domain.ndim != 2:
        raise ValueError(f"problem.domain musí byť 2D pole, shape={domain.shape}")
    if len(domain) != len(labels):
        raise ValueError("Počet bodov v domain a labels sa nezhoduje")

    n_points = domain.shape[0]

    initial_indices: list[int] = []
    if include_initial:
        initial_indices = _snap_indices_for_initial_points(problem)

    x_hist: list[list[float]] = []
    y_hist: list[float] = []
    best_so_far: list[float] = []
    chosen_indices: list[int] = []

    visited_unique: set[int] = set()

    for call_idx in range(budget):
        if include_initial and call_idx < len(initial_indices):
            idx = initial_indices[call_idx]
        else:
            idx = int(rng.integers(0, n_points))

        chosen_indices.append(idx)
        visited_unique.add(idx)

        x = domain[idx].astype(float).tolist()
        y = float(labels[idx])

        x_hist.append(x)
        y_hist.append(y)

        if not best_so_far:
            best_so_far.append(y)
        else:
            best_so_far.append(max(best_so_far[-1], y))

    best_idx = int(np.argmax(y_hist))
    best_x = x_hist[best_idx]
    best_y = float(y_hist[best_idx])

    optimum = float(problem.maximum)
    optimum_x = np.asarray(problem.maximiser, dtype=float).tolist()
    deviation = float(optimum - best_y)
    success = bool(np.isclose(best_y, optimum))

    final_best = best_so_far[-1]
    evals_to_f_best = next(
        i + 1 for i, v in enumerate(best_so_far)
        if np.isclose(v, final_best)
    )

    total_time = time.perf_counter() - start_total

    return RandomSearchLAQNResult(
        algorithm_name="RandomSearch",
        problem_id=str(problem.identifier),
        dimension=int(domain.shape[1]),
        run_id=run_id,
        chosen_indices=chosen_indices,
        X_hist=x_hist,
        y_hist=y_hist,
        best_so_far=best_so_far,
        best_x=best_x,
        best_y=best_y,
        budget=int(budget),
        call_count=int(budget),
        unique_eval_count=int(len(visited_unique)),
        evals_to_f_best=int(evals_to_f_best),
        total_time=float(total_time),
        deviation_from_optimum=deviation,
        optimum=optimum,
        optimum_x=optimum_x,
        success=success,
        seed=int(seed),
    )