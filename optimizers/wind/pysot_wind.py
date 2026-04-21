from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import time
import numpy as np

from poap.controller import SerialController

from pySOT.surrogate import (
    RBFInterpolant,
    GPRegressor,
    PolyRegressor,
    CubicKernel,
    LinearTail,
)

from pySOT.strategy import (
    DYCORSStrategy,
    SRBFStrategy,
    SOPStrategy,
    EIStrategy,
    LCBStrategy,
)

from pySOT.experimental_design import SymmetricLatinHypercube
from data.wind.problems.windwake import WindWakeLayout
from pySOT.optimization_problems import OptimizationProblem

@dataclass
class PySOTWindResult:
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

    surrogate_type: str
    strategy_type: str
    n_init: int
    batch_size: int
    num_cand: int | None
    use_restarts: bool
    asynchronous: bool
    verbose: bool

    extra_config: dict[str, Any] | None = None

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

class WindPySOTObjective:
    """
    Wrapper, ktorý zbiera jednotnú históriu evaluácií.
    """

    def __init__(self, problem):
        self.problem = problem

        self.X_hist: list[list[float]] = []
        self.y_hist: list[float] = []
        self.best_so_far: list[float] = []

        self.best_y = np.inf
        self.best_x_arr: np.ndarray | None = None
        self.call_count = 0

    def __call__(self, x):
        x = np.asarray(x, dtype=float).reshape(-1)
        y = float(self.problem.evaluate(x))

        self.call_count += 1
        self.X_hist.append(x.astype(float).tolist())
        self.y_hist.append(y)

        if y < self.best_y:
            self.best_y = y
            self.best_x_arr = x.copy()

        self.best_so_far.append(float(self.best_y))
        return y

class WindPySOTProblem(OptimizationProblem):
    """
    Adaptér na rozhranie očakávané pySOT.
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

def run_pysot_wind(
    problem,
    surrogate_type: str,
    strategy_type: str,
    budget: int = 20,
    seed: int = 0,
    run_id: int | None = None,
    n_init: int = 10,
    batch_size: int = 1,
    num_cand: int | None = 1000,
    use_restarts: bool = True,
    asynchronous: bool = True,
    verbose: bool = False,
    extra_config: dict[str, Any] | None = None,
) -> PySOTWindResult:
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

    lb = np.asarray(problem.lbs(), dtype=float)
    ub = np.asarray(problem.ubs(), dtype=float)
    dim = int(problem.dims())

    objective = WindPySOTObjective(problem=problem)
    problem_adapter = WindPySOTProblem(
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
            f"Running pySOT with surrogate={surrogate_type}, "
            f"strategy={strategy_type}, budget={budget}, n_init={n_init}"
        )

    controller.run()

    total_time = time.perf_counter() - start_total

    if objective.best_x_arr is None:
        raise RuntimeError("Nepodarilo sa určiť best_x")

    final_best = objective.best_so_far[-1]
    evals_to_f_best = next(
        i + 1 for i, v in enumerate(objective.best_so_far)
        if np.isclose(v, final_best)
    )

    problem_id = f"windwake_n{problem.n_turbines}_wseed{problem.wind_seed}"

    return PySOTWindResult(
        algorithm_name=f"PySOT-{surrogate_type.upper()}-{strategy_type.upper()}",
        problem_id=problem_id,
        dimension=dim,
        run_id=run_id,
        X_hist=objective.X_hist,
        y_hist=objective.y_hist,
        best_so_far=objective.best_so_far,
        best_x=objective.best_x_arr.astype(float).tolist(),
        best_y=float(objective.best_y),
        budget=int(budget),
        call_count=int(objective.call_count),
        evals_to_f_best=int(evals_to_f_best),
        total_time=float(total_time),
        seed=int(seed),
        file=str(problem.file),
        n_turbines=int(problem.n_turbines),
        wind_seed=int(problem.wind_seed),
        n_samples=None if problem.n_samples is None else int(problem.n_samples),
        lower_bounds=lb.astype(float).tolist(),
        upper_bounds=ub.astype(float).tolist(),
        surrogate_type=str(surrogate_type),
        strategy_type=str(strategy_type),
        n_init=int(n_init),
        batch_size=int(batch_size),
        num_cand=None if num_cand is None else int(num_cand),
        use_restarts=bool(use_restarts),
        asynchronous=bool(asynchronous),
        verbose=bool(verbose),
        extra_config=extra_config,
    )