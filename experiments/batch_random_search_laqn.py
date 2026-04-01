from dataclasses import asdict
from pathlib import Path
import pickle
import json
import numpy as np

from experiments.random_search_laqn import run_random_search_laqn


def load_problem(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    problem_dir = Path("data/laqn/2015/preprocessed")
    problem_files = sorted(problem_dir.glob("*.p"))

    budget = 20
    seed = 42
    include_initial = True

    results_summary = []

    for problem_file in problem_files:
        problem = load_problem(problem_file)

        result = run_random_search_laqn(
            problem=problem,
            budget=budget,
            seed=seed,
            include_initial=include_initial,
        )

        regret = float(problem.maximum - result.best_y)
        found_optimum = bool(np.isclose(result.best_y, problem.maximum))

        summary_row = {
            "file": str(problem_file),
            "identifier": problem.identifier,
            "n_domain_points": int(problem.domain.shape[0]),
            "budget": int(budget),
            "n_evals": int(result.n_evals),
            "best_y": float(result.best_y),
            "true_maximum": float(problem.maximum),
            "regret": regret,
            "found_optimum": found_optimum,
            "best_x": result.best_x.tolist(),
            "true_maximiser": np.asarray(problem.maximiser).tolist(),
            "best_hist": result.best_hist.tolist(),
        }

        results_summary.append(summary_row)

        print(
            f"{problem.identifier} | "
            f"best_y={result.best_y:.6f} | "
            f"max={problem.maximum:.6f} | "
            f"regret={regret:.6f} | "
            f"optimum_found={found_optimum}"
        )

    out_path = Path("results_random_search_2015_preprocessed.json")
    with open(out_path, "w") as f:
        json.dump(results_summary, f, indent=2)

    regrets = np.array([row["regret"] for row in results_summary], dtype=float)
    found = np.array([row["found_optimum"] for row in results_summary], dtype=bool)

    print("\n===== OVERALL SUMMARY =====")
    print(f"Number of problems: {len(results_summary)}")
    print(f"Mean regret: {np.mean(regrets):.6f}")
    print(f"Median regret: {np.median(regrets):.6f}")
    print(f"Std regret: {np.std(regrets):.6f}")
    print(f"Optimum found in: {np.sum(found)} / {len(found)} problems")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()