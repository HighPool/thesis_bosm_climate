from pathlib import Path
from optimizers.turbo_laqn import load_problem, run_turbo_on_problem

problem_path = next(Path("data/laqn/2015/preprocessed").glob("*.p"))
problem = load_problem(problem_path)

result = run_turbo_on_problem(
    problem,
    total_budget=1000,
    random_seed=0,
    verbose=False,
)

print("identifier:", result.identifier)
print("best_y:", result.best_y)
print("optimum:", result.optimum)
print("deviation:", result.deviation_from_optimum)
print("call_count:", result.call_count)
print("unique_eval_count:", result.unique_eval_count)
print("success:", result.success)