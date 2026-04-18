from __future__ import annotations

from pathlib import Path

from optimizers.pybads_laqn import load_problem, run_pybads_on_problem


problem_path = next(Path("data/laqn/2015/preprocessed").glob("*.p"))
problem = load_problem(problem_path)

result = run_pybads_on_problem(
    problem=problem,
    total_budget=500,
    random_seed=0,
    run_id=1,
    display="off",
)

print("algorithm_name:", result.algorithm_name)
print("problem_id:", result.problem_id)
print("best_y:", result.best_y)
print("optimum:", result.optimum)
print("deviation:", result.deviation_from_optimum)
print("call_count:", result.call_count)
print("unique_eval_count:", result.unique_eval_count)
print("success:", result.success)