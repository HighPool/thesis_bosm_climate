from __future__ import annotations
"""
Implementácia metódy TuRBO pre úlohy LAQN

Skript zabezpečuje:
- načítanie jednej problémovej inštancie LAQN
- premietnutie spojitého návrhu algoritmu TuRBO na diskrétnu množinu lokalít
- sledovanie histórie behového spustenia
- výpočet metrík jedného behového experimentu
- a voliteľné uloženie výsledku do JSON
"""

import pickle
import time
import numpy as np
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
from scipy.spatial import cKDTree
from turbo import Turbo1

@dataclass
class TurboLAQNResult:
    """
    Výsledok jedného spustenia implementácie TuRBO algoritmu na jednej LAQN úlohe

    Ukladajú sa:
    - id algoritmu, problému a spustenie behu
    - priebeh evaluácií a best-so-far krivka
    - finálne najlepšie riešenie
    - a základné metriky použité v ďalšom vyhodnotení
    """

    algorithm_name: str
    problem_id: str
    dimension: int
    run_id: int | None

    best_x: list[float]
    best_y: float
    best_so_far: list[float]
    x_hist: list[list[float]]
    y_hist: list[float]

    budget: int
    call_count: int
    unique_eval_count: int

    evals_to_f_best: int
    total_time: float
    deviation_from_optimum: float
    optimum: float
    optimum_x: list[float]
    success: bool

    # Formátovanie python objektu na JSON dáta
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

# Načítanie jednej inštancie problému LAQN z .p súboru
def load_problem(problem_path: str | Path):
    problem_path = Path(problem_path)
    with problem_path.open("rb") as f:
        return pickle.load(f)


class LAQNTurboObjective:
    """
    Wrapper - Most medzi spojitým priestorom TuRBO a diskrétnou doménou úlohy LAQN

    TuRBO pracuje v spojitom priestore a navrhuje 2D bod 
    Tento bod sa pri každom volaní premietne na najbližšiu platnú lokalitu z problem.domain
    Hodnota cieľovej funkcie sa zoberie z problem.labels
    TuRBO rieši minimalizačné úlohy, zatiaľ čo 
    LAQN je formulovaná ako maximalizačná úloha
    vracia sa hodnota -y

    total_budget      počet volaní algoritmu,
    unique_eval_count počet unikátne navštívených lokalít
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

        # Kontrola štruktúry vstupnej domény a hodnôt
        if self.domain.ndim != 2:
            raise ValueError(f"problem.domain musí byť 2D pole, shape={self.domain.shape}")
        if self.domain.shape[1] != 2:
            raise ValueError(f"Očakávam 2D problém, ale domain má shape={self.domain.shape}")
        if len(self.domain) != len(self.labels):
            raise ValueError("Počet bodov v domain a labels sa nezhoduje")

        # KD-strom slúži na rýchle priradenie spojitého bodu k najbližšej lokalite
        self.tree = cKDTree(self.domain)

        # Cache uchováva hodnoty navštívených bodov domény
        self.cache: dict[int, float] = {}

        # História sa vedie po jednotlivých volaniach algoritmu
        self.x_hist: list[np.ndarray] = []
        self.y_hist: list[float] = []
        self.best_so_far: list[float] = []

        self.call_count = 0

        # Počiatočné body sa môžu prevziať zo samotnej inštancie problému
        self.initial_x = (
            np.asarray(problem.xx, dtype=float)
            if include_initial_points else np.empty((0, 2))
        )
        self.initial_y = (
            np.asarray(problem.yy, dtype=float).reshape(-1)
            if include_initial_points else np.empty((0,))
        )

        if len(self.initial_x) != len(self.initial_y):
            raise ValueError("Počet xx a yy sa nezhoduje")

    @property
    # Počet unikátne navštívených lokalít počas spustenia behov
    def unique_eval_count(self) -> int:
        return len(self.cache)

    # Premietne spojitý 2D bod na index najbližšieho bodu z diskrétnej domény
    def _snap_to_index(self, x: np.ndarray) -> int:
        x = np.asarray(x, dtype=float).reshape(-1)
        _, idx = self.tree.query(x, k=1)
        return int(idx)

    def suggest_initial_points(self, n_init: int) -> np.ndarray:
        """
        Navrhne počiatočné body pre TuRBO.
        Prednostne sa použijú počiatočné body z problem.xx
        Ak ich je menej než požaduje TuRBO, zvyšok sa doplní náhodne z diskrétnej domény.
        """
        candidates: list[np.ndarray] = []
        used_init_indices: set[int] = set()

        for x0 in self.initial_x:
            idx = self._snap_to_index(x0)
            if idx in used_init_indices:
                continue
            candidates.append(self.domain[idx].copy())
            used_init_indices.add(idx)
            if len(candidates) >= n_init:
                break

        if len(candidates) < n_init:
            remaining = [i for i in range(len(self.domain)) if i not in used_init_indices]
            if len(remaining) < (n_init - len(candidates)):
                raise ValueError("Nedostatok unikátnych bodov v domain pre inicializáciu.")

            extra = np.random.choice(remaining, size=(n_init - len(candidates)), replace=False)
            for idx in extra:
                candidates.append(self.domain[int(idx)].copy())

        return np.asarray(candidates, dtype=float)

    def __call__(self, x: np.ndarray) -> float:
        """
        Vyhodnotenie jedného bodu navrhnutého algoritmom
        Po dosiahnutí rozpočtu sa ďalšie volania už nezapočítavajú
        """
        if self.call_count >= self.total_budget:
            if self.y_hist:
                return -float(self.y_hist[-1])
            return 0.0

        self.call_count += 1

        idx = self._snap_to_index(x)

        # Nový bod sa vloží do cache len pri prvej návšteve.
        if idx not in self.cache:
            self.cache[idx] = float(self.labels[idx])

        y = float(self.cache[idx])

        # Ukladanie histórie po každom volaní algoritmu.
        self.x_hist.append(self.domain[idx].copy())
        self.y_hist.append(y)

        if not self.best_so_far:
            self.best_so_far.append(y)
        else:
            self.best_so_far.append(max(self.best_so_far[-1], y))

        return -y


def run_turbo_on_problem(
    problem,
    total_budget: int = 500,
    random_seed: int | None = None,
    run_id: int | None = None,
    use_ard: bool = True,
    n_training_steps: int = 50,
    verbose: bool = False,
    batch_size: int = 1,
    n_init: int | None = None,
    device: str = "cpu",
    dtype: str = "float64",
) -> TurboLAQNResult:
    """
    Spustí jeden beh metódy TuRBO na jednej LAQN úlohe.

    Funkcia:
    - pripraví objective wrapper
    - nastaví hranice priestoru
    - inicializuje TuRBO
    - spustí optimalizáciu
    - vypočíta metriky jedného experimentu
    """
    start_total = time.perf_counter()

    if random_seed is not None:
        np.random.seed(random_seed)

    domain = np.asarray(problem.domain, dtype=float)
    if domain.ndim != 2 or domain.shape[1] != 2:
        raise ValueError(f"Očakávam 2D doménu shape (n,2), dostal som {domain.shape}")

    lb = domain.min(axis=0)
    ub = domain.max(axis=0)

    objective = LAQNTurboObjective(
        problem=problem,
        total_budget=total_budget,
        include_initial_points=True,
    )

    # Počet inicializačných bodov
    if n_init is None:
        n_init = max(4, 2 * domain.shape[1])

    X_init = objective.suggest_initial_points(n_init=n_init)

    # Kompatibilita medzi verziami TuRBO algoritmu
    try:
        turbo = Turbo1(
            f=objective,
            lb=lb,
            ub=ub,
            n_init=n_init,
            max_evals=total_budget,
            batch_size=batch_size,
            verbose=verbose,
            use_ard=use_ard,
            n_training_steps=n_training_steps,
            device=device,
            dtype=dtype,
            X_init=X_init,
        )
    except TypeError:
        turbo = Turbo1(
            f=objective,
            lb=lb,
            ub=ub,
            n_init=n_init,
            max_evals=total_budget,
            batch_size=batch_size,
            verbose=verbose,
            use_ard=use_ard,
            n_training_steps=n_training_steps,
            device=device,
            dtype=dtype,
        )

    turbo.optimize()

    if not objective.y_hist:
        raise RuntimeError("Nevznikla žiadna história evaluácií.")

    # Najlepšie riešenie sa určuje podľa maximálnej hodnoty enviromentálneho senzora
    best_idx = int(np.argmax(objective.y_hist))
    best_x = np.asarray(objective.x_hist[best_idx], dtype=float)
    best_y = float(objective.y_hist[best_idx])

    optimum = float(problem.maximum)
    optimum_x = np.asarray(problem.maximiser, dtype=float)
    deviation = float(optimum - best_y)
    success = bool(np.isclose(best_y, optimum))

    # evals_to_f_best udáva, po koľkých TuRBO volaniach bolo prvýkrát dosiahnuté finálne maximum
    final_best = objective.best_so_far[-1]
    evals_to_f_best = next(
        i + 1 for i, v in enumerate(objective.best_so_far)
        if np.isclose(v, final_best)
    )

    total_time = time.perf_counter() - start_total

    return TurboLAQNResult(
        algorithm_name="TuRBO",
        problem_id=str(problem.identifier),
        dimension=int(domain.shape[1]),
        run_id=run_id,
        best_x=best_x.tolist(),
        best_y=best_y,
        best_so_far=[float(v) for v in objective.best_so_far],
        x_hist=[np.asarray(x, dtype=float).tolist() for x in objective.x_hist],
        y_hist=[float(v) for v in objective.y_hist],
        budget=int(total_budget),
        call_count=int(objective.call_count),
        unique_eval_count=int(objective.unique_eval_count),
        evals_to_f_best=int(evals_to_f_best),
        total_time=float(total_time),
        deviation_from_optimum=deviation,
        optimum=optimum,
        optimum_x=optimum_x.tolist(),
        success=success,
    )

# Uloží výsledok jedného behu experimentu do JSON súboru
def save_result_json(result: TurboLAQNResult, out_path: str | Path) -> None:
    import json

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)