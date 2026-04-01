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
    n_runs = 10
    include_initial = True

    all_problem_summaries = []

    for problem_file in problem_files:
        problem = load_problem(problem_file)

        regrets = []
        best_values = []
        success_flags = []
        all_curves = []

        for run_idx in range(n_runs):
            result = run_random_search_laqn(
                problem=problem,
                budget=budget,
                seed=run_idx,
                include_initial=include_initial,
            )

            regret = float(problem.maximum - result.best_y)
            success = bool(np.isclose(result.best_y, problem.maximum))

            regrets.append(regret)
            best_values.append(result.best_y)
            success_flags.append(success)
            all_curves.append(result.best_hist)

        regrets = np.array(regrets)
        best_values = np.array(best_values)
        success_flags = np.array(success_flags)
        all_curves = np.array(all_curves)

        summary = {
            "identifier": problem.identifier,
            "n_domain_points": int(problem.domain.shape[0]),
            "budget": budget,
            "n_runs": n_runs,

            "mean_regret": float(np.mean(regrets)),
            "median_regret": float(np.median(regrets)),
            "std_regret": float(np.std(regrets)),

            "mean_best_y": float(np.mean(best_values)),
            "median_best_y": float(np.median(best_values)),

            "success_rate": float(np.mean(success_flags)),

            "mean_curve": np.mean(all_curves, axis=0).tolist(),
            "median_curve": np.median(all_curves, axis=0).tolist(),

            "true_maximum": float(problem.maximum),
        }

        all_problem_summaries.append(summary)

        print(
            f"{problem.identifier} | "
            f"mean_regret={summary['mean_regret']:.4f} | "
            f"success_rate={summary['success_rate']:.2f}"
        )

    # ===== GLOBAL SUMMARY =====
    mean_regrets = np.array([p["mean_regret"] for p in all_problem_summaries])
    success_rates = np.array([p["success_rate"] for p in all_problem_summaries])

    print("\n===== GLOBAL SUMMARY =====")
    print(f"Problems: {len(all_problem_summaries)}")
    print(f"Mean regret (across problems): {np.mean(mean_regrets):.6f}")
    print(f"Median regret: {np.median(mean_regrets):.6f}")
    print(f"Std regret: {np.std(mean_regrets):.6f}")
    print(f"Mean success rate: {np.mean(success_rates):.4f}")

    # uloženie
    out_path = Path("results/results_random_search_2015_multirun.json")
    with open(out_path, "w") as f:
        json.dump(all_problem_summaries, f, indent=2)

    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()