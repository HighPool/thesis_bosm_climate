from __future__ import annotations
import pickle
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import numpy as np
from poap.controller import SerialController
from pySOT.experimental_design import SymmetricLatinHypercube
from pySOT.optimization_problems import OptimizationProblem
from pySOT.strategy import (
    DYCORSStrategy,
    EIStrategy,
    LCBStrategy,
    SOPStrategy,
    SRBFStrategy,
)
from pySOT.surrogate import (
    CubicKernel,
    GPRegressor,
    LinearTail,
    PolyRegressor,
    RBFInterpolant,
)
from scipy.spatial import cKDTree
from sklearn.exceptions import ConvergenceWarning

print("pysot_laqn.py imported", flush=True)

@dataclass
class PySOTLAQNResult:
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

    surrogate_type: str
    strategy_type: str
    n_init: int
    batch_size: int
    num_cand: int | None
    use_restarts: bool
    asynchronous: bool
    verbose: bool
    seed: int

    lower_bounds: list[float]
    upper_bounds: list[float]

    extra_config: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

def load_problem(problem_path: str | Path):
    print(f"Loading problem from: {problem_path}", flush=True)
    problem_path = Path(problem_path)
    with problem_path.open("rb") as f:
        return pickle.load(f)

def _validate_pysot_combination(
    surrogate_type: str,
    strategy_type: str,
) -> None:
    surrogate_type = surrogate_type.lower()
    strategy_type = strategy_type.lower()

    allowed_surrogates = {"rbf", "gp", "poly"}
    allowed_strategies = {"dycors", "srbf", "sop", "ei", "lcb"}

    if surrogate_type not in allowed_surrogates:
        raise ValueError(
            f"Neplatný surrogate_type='{surrogate_type}'. "
            f"Povolené: {sorted(allowed_surrogates)}"
        )

    if strategy_type not in allowed_strategies:
        raise ValueError(
            f"Neplatný strategy_type='{strategy_type}'. "
            f"Povolené: {sorted(allowed_strategies)}"
        )

    if strategy_type in {"ei", "lcb"} and surrogate_type != "gp":
        raise ValueError(
            f"Stratégia '{strategy_type}' vyžaduje surrogate_type='gp'."
        )

class LAQNPySOTObjective:
    """
    Wrapper medzi spojitým priestorom pySOT a diskrétnou doménou LAQN.
    pySOT minimalizuje, LAQN je vecne maximalizačná úloha.
    Preto objective vracia -y.
    """

    def __init__(self, problem, total_budget: int, verbose: bool = False):
        self.problem = problem
        self.total_budget = int(total_budget)
        self.verbose = bool(verbose)

        self.domain = np.asarray(problem.domain, dtype=float)
        self.labels = np.asarray(problem.labels, dtype=float).reshape(-1)

        if self.domain.ndim != 2:
            raise ValueError(f"problem.domain musí byť 2D pole, shape={self.domain.shape}")
        if self.domain.shape[1] != 2:
            raise ValueError(f"Očakávam 2D problém, ale domain má shape={self.domain.shape}")
        if len(self.domain) != len(self.labels):
            raise ValueError("Počet bodov v domain a labels sa nezhoduje")

        self.tree = cKDTree(self.domain)

        # cache podľa indexu diskrétnej lokality
        self.cache: dict[int, float] = {}

        self.x_hist: list[list[float]] = []
        self.y_hist: list[float] = []
        self.best_so_far: list[float] = []

        self.best_y = -np.inf
        self.best_x_arr: np.ndarray | None = None
        self.call_count = 0

    @property
    def unique_eval_count(self) -> int:
        return len(self.cache)

    def _snap_to_index(self, x: np.ndarray) -> int:
        x = np.asarray(x, dtype=float).reshape(-1)
        _, idx = self.tree.query(x, k=1)
        return int(idx)

    def __call__(self, x):
        x = np.asarray(x, dtype=float).reshape(-1)

        if self.call_count >= self.total_budget:
            if self.y_hist:
                return -float(self.y_hist[-1])
            return 0.0

        self.call_count += 1

        idx = self._snap_to_index(x)

        if idx not in self.cache:
            self.cache[idx] = float(self.labels[idx])

        y = float(self.cache[idx])
        snapped_x = self.domain[idx].copy()

        self.x_hist.append(snapped_x.astype(float).tolist())
        self.y_hist.append(y)

        if y > self.best_y:
            self.best_y = y
            self.best_x_arr = snapped_x.copy()

        if not self.best_so_far:
            self.best_so_far.append(y)
        else:
            self.best_so_far.append(max(self.best_so_far[-1], y))

        if self.verbose:
            print(
                f"eval {self.call_count:03d}/{self.total_budget}: "
                f"y = {y:.6f} | best = {self.best_so_far[-1]:.6f}",
                flush=True,
            )

        return -y

class LAQNPySOTProblem(OptimizationProblem):
    """
    pySOT adapter pre LAQN objective.
    """

    def __init__(self, objective, dim: int, lb: np.ndarray, ub: np.ndarray):
        self.objective = objective
        self.dim = int(dim)
        self.lb = np.asarray(lb, dtype=float)
        self.ub = np.asarray(ub, dtype=float)

        self.int_var = np.array([], dtype=int)
        self.cont_var = np.arange(self.dim, dtype=int)

    def eval(self, x):
        x = np.asarray(x, dtype=float).reshape(-1)
        return self.objective(x)

def _build_surrogate(
    surrogate_type: str,
    dim: int,
    lb: np.ndarray,
    ub: np.ndarray,
):
    surrogate_type = surrogate_type.lower()

    if surrogate_type == "rbf":
        return RBFInterpolant(
            dim=dim,
            lb=lb,
            ub=ub,
            kernel=CubicKernel(),
            tail=LinearTail(dim),
        )

    if surrogate_type == "gp":
        return GPRegressor(
            dim=dim,
            lb=lb,
            ub=ub,
        )

    if surrogate_type == "poly":
        return PolyRegressor(
            dim=dim,
            lb=lb,
            ub=ub,
            degree=2,
        )

    raise ValueError(f"Neznámy surrogate_type='{surrogate_type}'")

def _build_strategy(
    strategy_type: str,
    problem_adapter,
    surrogate,
    max_evals: int,
    n_init: int,
    batch_size: int,
    num_cand: int | None,
    use_restarts: bool,
    asynchronous: bool,
    extra_config: dict[str, Any] | None,
):
    strategy_type = strategy_type.lower()
    extra_config = dict(extra_config or {})

    exp_design = SymmetricLatinHypercube(
        dim=problem_adapter.dim,
        num_pts=n_init,
    )

    common_kwargs = {
        "max_evals": int(max_evals),
        "opt_prob": problem_adapter,
        "exp_design": exp_design,
        "surrogate": surrogate,
        "asynchronous": bool(asynchronous),
        "batch_size": int(batch_size),
        "use_restarts": bool(use_restarts),
    }

    if strategy_type in {"dycors", "srbf", "sop"} and num_cand is not None:
        common_kwargs["num_cand"] = int(num_cand)

    common_kwargs.update(extra_config)

    if strategy_type == "dycors":
        return DYCORSStrategy(**common_kwargs)

    if strategy_type == "srbf":
        return SRBFStrategy(**common_kwargs)

    if strategy_type == "sop":
        return SOPStrategy(**common_kwargs)

    if strategy_type == "ei":
        return EIStrategy(**common_kwargs)

    if strategy_type == "lcb":
        return LCBStrategy(**common_kwargs)

    raise ValueError(f"Neznámy strategy_type='{strategy_type}'")


def run_pysot_laqn(
    problem,
    surrogate_type: str,
    strategy_type: str,
    budget: int = 500,
    seed: int = 0,
    run_id: int | None = None,
    n_init: int = 10,
    batch_size: int = 1,
    num_cand: int | None = 1000,
    use_restarts: bool = True,
    asynchronous: bool = True,
    verbose: bool = False,
    extra_config: dict[str, Any] | None = None,
) -> PySOTLAQNResult:
    print("run_pysot_laqn() entered", flush=True)

    if budget <= 0:
        raise ValueError("budget musí byť kladný")
    if n_init <= 0:
        raise ValueError("n_init musí byť kladné")
    if n_init > budget:
        raise ValueError("n_init nesmie byť väčšie ako budget")

    _validate_pysot_combination(
        surrogate_type=surrogate_type,
        strategy_type=strategy_type,
    )

    start_total = time.perf_counter()
    np.random.seed(seed)

    domain = np.asarray(problem.domain, dtype=float)
    if domain.ndim != 2 or domain.shape[1] != 2:
        raise ValueError(f"Očakávam 2D doménu shape (n,2), dostal som {domain.shape}")

    lb = domain.min(axis=0)
    ub = domain.max(axis=0)
    dim = int(domain.shape[1])

    objective = LAQNPySOTObjective(
        problem=problem,
        total_budget=budget,
        verbose=verbose,
    )

    problem_adapter = LAQNPySOTProblem(
        objective=objective,
        dim=dim,
        lb=lb,
        ub=ub,
    )

    surrogate = _build_surrogate(
        surrogate_type=surrogate_type,
        dim=dim,
        lb=lb,
        ub=ub,
    )

    strategy = _build_strategy(
        strategy_type=strategy_type,
        problem_adapter=problem_adapter,
        surrogate=surrogate,
        max_evals=int(budget),
        n_init=int(n_init),
        batch_size=int(batch_size),
        num_cand=num_cand,
        use_restarts=use_restarts,
        asynchronous=asynchronous,
        extra_config=extra_config,
    )

    controller = SerialController(problem_adapter.eval)
    controller.strategy = strategy

    if verbose:
        print(
            f"Running pySOT LAQN with surrogate={surrogate_type}, "
            f"strategy={strategy_type}, budget={budget}, n_init={n_init}",
            flush=True,
        )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        controller.run()

    total_time = time.perf_counter() - start_total

    if objective.best_x_arr is None:
        raise RuntimeError("Nepodarilo sa určiť best_x")

    final_best = objective.best_so_far[-1]
    evals_to_f_best = next(
        i + 1 for i, v in enumerate(objective.best_so_far)
        if np.isclose(v, final_best)
    )

    optimum = float(problem.maximum)
    optimum_x = np.asarray(problem.maximiser, dtype=float)
    deviation = float(optimum - objective.best_y)
    success = bool(np.isclose(objective.best_y, optimum))

    return PySOTLAQNResult(
        algorithm_name=f"PySOT-{surrogate_type.upper()}-{strategy_type.upper()}",
        problem_id=str(problem.identifier),
        dimension=dim,
        run_id=run_id,
        best_x=objective.best_x_arr.astype(float).tolist(),
        best_y=float(objective.best_y),
        best_so_far=[float(v) for v in objective.best_so_far],
        x_hist=objective.x_hist,
        y_hist=[float(v) for v in objective.y_hist],
        budget=int(budget),
        call_count=int(objective.call_count),
        unique_eval_count=int(objective.unique_eval_count),
        evals_to_f_best=int(evals_to_f_best),
        total_time=float(total_time),
        deviation_from_optimum=deviation,
        optimum=optimum,
        optimum_x=optimum_x.astype(float).tolist(),
        success=success,
        surrogate_type=str(surrogate_type),
        strategy_type=str(strategy_type),
        n_init=int(n_init),
        batch_size=int(batch_size),
        num_cand=None if num_cand is None else int(num_cand),
        use_restarts=bool(use_restarts),
        asynchronous=bool(asynchronous),
        verbose=bool(verbose),
        seed=int(seed),
        lower_bounds=lb.astype(float).tolist(),
        upper_bounds=ub.astype(float).tolist(),
        extra_config=extra_config,
    )

def save_result_json(result: PySOTLAQNResult, out_path: str | Path) -> None:
    import json

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)