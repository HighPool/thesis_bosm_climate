from dataclasses import dataclass
import numpy as np
import pickle
from pathlib import Path


@dataclass
class RandomSearchLAQNResult:
    chosen_indices: np.ndarray
    X_hist: np.ndarray
    y_hist: np.ndarray
    best_hist: np.ndarray
    best_x: np.ndarray
    best_y: float
    n_evals: int
    seed: int
    identifier: str


def load_problem(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def run_random_search_laqn(problem, budget=20, seed=0, include_initial=True):
    rng = np.random.default_rng(seed)

    domain = np.asarray(problem.domain, dtype=float)
    labels = np.asarray(problem.labels, dtype=float)

    n_points = domain.shape[0]

    if budget <= 0:
        raise ValueError("budget musí byť kladný")

    # zoznam už pozorovaných indexov
    used = set()

    X_hist = []
    y_hist = []

    # ak chceš zachovať benchmarkový štart z 5 bodov
    if include_initial:
        for x0, y0 in zip(problem.xx, problem.yy):
            # nájdi index x0 v doméne
            matches = np.where(np.all(np.isclose(domain, x0), axis=1))[0]
            if len(matches) == 0:
                raise RuntimeError("Počiatočný bod z .xx sa nenašiel v .domain")
            idx = int(matches[0])
            if idx not in used:
                used.add(idx)
                X_hist.append(domain[idx])
                y_hist.append(labels[idx])

    remaining_budget = budget - len(X_hist)
    if remaining_budget < 0:
        raise ValueError("budget je menší než počet počiatočných bodov")

    available = np.array([i for i in range(n_points) if i not in used], dtype=int)

    chosen_indices = []

    for _ in range(remaining_budget):
        if len(available) == 0:
            break

        pick_pos = rng.integers(0, len(available))
        idx = int(available[pick_pos])

        chosen_indices.append(idx)
        used.add(idx)

        X_hist.append(domain[idx])
        y_hist.append(labels[idx])

        available = np.delete(available, pick_pos)

    X_hist = np.asarray(X_hist, dtype=float)
    y_hist = np.asarray(y_hist, dtype=float)

    # LAQN je maximalizačná úloha
    best_hist = np.maximum.accumulate(y_hist)
    best_idx = int(np.argmax(y_hist))
    best_x = X_hist[best_idx]
    best_y = float(y_hist[best_idx])

    return RandomSearchLAQNResult(
        chosen_indices=np.asarray(chosen_indices, dtype=int),
        X_hist=X_hist,
        y_hist=y_hist,
        best_hist=best_hist,
        best_x=best_x,
        best_y=best_y,
        n_evals=len(y_hist),
        seed=seed,
        identifier=problem.identifier,
    )


if __name__ == "__main__":
    problem_file = next(Path("data_sorted/2015_problems/preprocessed").glob("*.p"))
    problem = load_problem(problem_file)

    result = run_random_search_laqn(problem, budget=20, seed=42, include_initial=True)

    print("Problem:", result.identifier)
    print("n_evals:", result.n_evals)
    print("best_y:", result.best_y)
    print("best_x:", result.best_x)
    print("true maximum:", problem.maximum)
    print("true maximiser:", problem.maximiser)