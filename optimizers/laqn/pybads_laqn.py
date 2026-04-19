from __future__ import annotations

"""
Implementácia metódy pyBADS pre úlohy LAQN

Skript zabezpečuje:
- načítanie jednej problémovej inštancie LAQN
- premietnutie spojitého návrhu algoritmu pyBADS na diskrétnu množinu lokalít
- sledovanie histórie jedného behového spustenia
- výpočet metrík jedného experimentu
- a prípravu výsledku na ďalšie uloženie alebo agregáciu
"""

import io
import logging
import pickle
import time
import warnings
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
from pybads import BADS
from scipy.spatial import cKDTree


class BudgetReached(Exception):
    """
    Pomocná výnimka používaná na ukončenie spustenia behov po vyčerpaní rozpočtu

    pyBADS si môže počas optimalizácie pýtať ďalšie evaluácie. Keď objective wrapper
    zistí, že bol dosiahnutý limit volaní algoritmu, vyhodí túto výnimku a tým
    korektne preruší ďalšie spracovanie
    """
    pass


@dataclass
class PyBADSLAQNResult:
    """
    Výsledok jednej implementácie pyBADS algoritmu na jednej LAQN úlohe

    Ukladajú sa:
    - identifikátory algoritmu, problému a spustenia behov
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


class LAQNPyBADSObjective:
    """
    Wrapper - Most medzi spojitým priestorom PyBADS a diskrétnou doménou úlohy LAQN

    pyBADS pracuje v spojitom priestore a navrhuje 2D bod
    Tento bod sa pri každom volaní premietne na najbližšiu platnú lokalitu z problem.domain
    Hodnota cieľovej funkcie sa zoberie z problem.labels
    pyBADS rieši minimalizačné úlohy, zatiaľ čo 
    LAQN je formulovaná ako maximalizačná úloha
    vracia sa hodnota -y

    total_budget      počet volaní algoritmu,
    unique_eval_count počet unikátne navštívených lokalít
    """

    def __init__(
        self,
        problem,
        total_budget: int = 500,
        include_initial_points: bool = True,
    ):
        self.problem = problem
        self.total_budget = int(total_budget)

        self.domain = np.asarray(problem.domain, dtype=float)
        self.labels = np.asarray(problem.labels, dtype=float).reshape(-1)

        # Kontrola konzistencie vstupnej domény a hodnôt
        if self.domain.ndim != 2:
            raise ValueError(f"problem.domain musí byť 2D pole, shape={self.domain.shape}")
        if self.domain.shape[1] != 2:
            raise ValueError(f"Očakávam 2D problém, ale domain má shape={self.domain.shape}")
        if len(self.domain) != len(self.labels):
            raise ValueError("Počet bodov v domain a labels sa nezhoduje")

        # KD-strom slúži na rýchle priradenie spojitého bodu k najbližšej lokalite
        self.tree = cKDTree(self.domain)

        # Cache uchováva hodnoty už navštívených lokalít
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

    def sample_restart_point(self) -> np.ndarray:
        """
        Vyberie nový štartovací bod pre ďalší restart pyBADS
        Pri tejto implementácii sa štartovací bod vyberá náhodne z domény
        """
        idx = int(np.random.randint(0, len(self.domain)))
        return self.domain[idx].copy()

    def suggest_initial_x0(self) -> np.ndarray:
        """
        Vyberie počiatočný bod pre prvý štart pyBADS.
        Ak sú k dispozícii počiatočné body z problémovej inštancie, použije sa
        ten s najvyššou počiatočnou hodnotou. Inak sa zvolí náhodný bod z domény.
        """
        if len(self.initial_x) == 0:
            idx = int(np.random.randint(0, len(self.domain)))
            return self.domain[idx].copy()

        best_init_idx = int(np.argmax(self.initial_y))
        return self.initial_x[best_init_idx].astype(float)

    def __call__(self, x: np.ndarray) -> float:
        """
        Vyhodnotenie jedného bodu navrhnutého algoritmom
        Po dosiahnutí rozpočtu sa objective wrapper ukončí výnimkou BudgetReached
        čím sa zastaví ďalšie volanie evaluácií zo strany pyBADS.
        """
        if self.call_count >= self.total_budget:
            raise BudgetReached("Dosiahnutý limit volaní algoritmu.")

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


def _build_and_run_bads_silently(
    objective,
    x0,
    lb,
    ub,
    plb,
    pub,
    options,
):

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
    total_budget: int = 500,
    random_seed: int | None = None,
    run_id: int | None = None,
    display: str = "off",
) -> PyBADSLAQNResult:
    """
    Spustí jeden beh pyBADS algoritmu na jednej LAQN úlohe

    Funkcia:
    - pripraví objective wrapper
    - nastaví hranice priestoru a plausibilné hranice
    - inicializuje pyBADS
    - opakovane ho spúšťa cez restarty, kým sa nevyčerpá rozpočet
    - a po dobehu vypočíta metriky jedného behového experimentu
    """
    start_total = time.perf_counter()

    if random_seed is not None:
        np.random.seed(random_seed)

    domain = np.asarray(problem.domain, dtype=float)

    lb = domain.min(axis=0)
    ub = domain.max(axis=0)

    # plausibilné hranice zúžia "rozumne pravdepodobnú" časť priestoru pre pyBADS
    plb = np.quantile(domain, 0.10, axis=0)
    pub = np.quantile(domain, 0.90, axis=0)

    objective = LAQNPyBADSObjective(
        problem=problem,
        total_budget=total_budget,
        include_initial_points=True,
    )

    x0 = objective.suggest_initial_x0()

    restart_id = 0
    max_restarts = 200

    # pyBADS sa opakovane reštartuje, kým sa nevyčerpá rozpočet alebo počet restartov.
    while objective.call_count < total_budget and restart_id < max_restarts:
        restart_id += 1

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

        if objective.call_count >= total_budget:
            break

        x0 = objective.sample_restart_point()

    if not objective.y_hist:
        raise RuntimeError("Nevznikla žiadna história evaluácií.")

    # Najlepšie riešenie sa určuje podľa maximálnej pozorovanej hodnoty.
    best_idx = int(np.argmax(objective.y_hist))
    best_x = np.asarray(objective.x_hist[best_idx], dtype=float)
    best_y = float(objective.y_hist[best_idx])

    optimum = float(problem.maximum)
    optimum_x = np.asarray(problem.maximiser, dtype=float)
    deviation = float(optimum - best_y)
    success = bool(np.isclose(best_y, optimum))

    # evals_to_f_best udáva, po koľkých volaniach bolo prvýkrát dosiahnuté finálne maximum.
    final_best = objective.best_so_far[-1]
    evals_to_f_best = next(
        i + 1 for i, v in enumerate(objective.best_so_far)
        if np.isclose(v, final_best)
    )

    total_time = time.perf_counter() - start_total

    return PyBADSLAQNResult(
        algorithm_name="PyBADS",
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