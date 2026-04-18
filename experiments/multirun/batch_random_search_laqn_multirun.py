from pathlib import Path
import pickle
import json
import time
import numpy as np

from experiments.singlerun.random_search_laqn import run_random_search_laqn

def load_problem(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def main():
    problem_dir = Path("data/laqn/2015/preprocessed")
    problem_files = sorted(problem_dir.glob("*.p"))

    budget = 40
    n_runs = 30
    include_initial = True

    all_problem_summaries = []

    for problem_file in problem_files:
        problem = load_problem(problem_file)

        regrets = []
        best_values = []
        success_flags = []
        all_curves = []
        times = []
        evals_to_best_list = []

        for run_idx in range(n_runs):
            start = time.perf_counter()

            result = run_random_search_laqn(
                problem=problem,
                budget=budget,
                seed=run_idx,
                include_initial=include_initial,
            )

            elapsed = time.perf_counter() - start

            regret = float(problem.maximum - result.best_y)
            success = bool(np.isclose(result.best_y, problem.maximum))

            final_best = result.best_hist[-1]
            evals_to_f_best = next(
                i + 1 for i, v in enumerate(result.best_hist)
                if np.isclose(v, final_best)
            )

            regrets.append(regret)
            best_values.append(float(result.best_y))
            success_flags.append(success)
            all_curves.append(result.best_hist)
            times.append(elapsed)
            evals_to_best_list.append(evals_to_f_best)

        regrets = np.array(regrets, dtype=float)
        best_values = np.array(best_values, dtype=float)
        success_flags = np.array(success_flags, dtype=float)
        all_curves = np.array(all_curves, dtype=float)
        times = np.array(times, dtype=float)
        evals_to_best_list = np.array(evals_to_best_list, dtype=float)

        summary = {
            "identifier": problem.identifier,
            "n_domain_points": int(problem.domain.shape[0]),
            "budget": budget,
            "n_runs": n_runs,

            "mean_deviation": float(np.mean(regrets)),
            "median_deviation": float(np.median(regrets)),
            "std_deviation": float(np.std(regrets)),

            "mean_best_y": float(np.mean(best_values)),
            "median_best_y": float(np.median(best_values)),
            "std_best_y": float(np.std(best_values)),

            "success_rate": float(np.mean(success_flags)),

            "mean_evals_to_f_best": float(np.mean(evals_to_best_list)),
            "median_evals_to_f_best": float(np.median(evals_to_best_list)),
            "std_evals_to_f_best": float(np.std(evals_to_best_list)),

            "mean_total_time": float(np.mean(times)),
            "std_total_time": float(np.std(times)),

            "mean_curve": np.mean(all_curves, axis=0).tolist(),
            "median_curve": np.median(all_curves, axis=0).tolist(),

            "true_maximum": float(problem.maximum),
        }

        all_problem_summaries.append(summary)

        print(
            f"{problem.identifier} | "
            f"mean_deviation={summary['mean_deviation']:.4f} | "
            f"success_rate={summary['success_rate']:.2f} | "
            f"mean_evals_to_best={summary['mean_evals_to_f_best']:.2f}"
        )

    # ===== GLOBÁLNY SÚHRN =====
    mean_deviations = np.array([p["mean_deviation"] for p in all_problem_summaries], dtype=float)
    success_rates = np.array([p["success_rate"] for p in all_problem_summaries], dtype=float)
    mean_best_values = np.array([p["mean_best_y"] for p in all_problem_summaries], dtype=float)
    mean_evals_to_best = np.array([p["mean_evals_to_f_best"] for p in all_problem_summaries], dtype=float)
    mean_times = np.array([p["mean_total_time"] for p in all_problem_summaries], dtype=float)

    all_problem_mean_curves = np.array([p["mean_curve"] for p in all_problem_summaries], dtype=float)
    global_mean_curve = np.mean(all_problem_mean_curves, axis=0).tolist()

    global_summary = {
        "n_problems": len(all_problem_summaries),
        "mean_deviation": float(np.mean(mean_deviations)),
        "median_deviation": float(np.median(mean_deviations)),
        "std_deviation": float(np.std(mean_deviations)),
        "success_rate": float(np.mean(success_rates)),
        "mean_best_y": float(np.mean(mean_best_values)),
        "mean_evals_to_f_best": float(np.mean(mean_evals_to_best)),
        "mean_total_time": float(np.mean(mean_times)),
        "mean_best_so_far_curve": global_mean_curve,
        "min_curve_length": len(global_mean_curve),
    }

    print("\n===== GLOBÁLNY SÚHRN =====")
    print(f"Problems: {global_summary['n_problems']}")
    print(f"Mean deviation: {global_summary['mean_deviation']:.6f}")
    print(f"Median deviation: {global_summary['median_deviation']:.6f}")
    print(f"Std deviation: {global_summary['std_deviation']:.6f}")
    print(f"Mean success rate: {global_summary['success_rate']:.4f}")
    print(f"Mean evals_to_f_best: {global_summary['mean_evals_to_f_best']:.4f}")
    print(f"Mean total_time: {global_summary['mean_total_time']:.4f} s")

    payload = {
        "config": {
            "problem_dir": str(problem_dir),
            "budget": budget,
            "n_runs": n_runs,
            "include_initial": include_initial,
        },
        "summary": global_summary,
        "results": all_problem_summaries,
    }

    out_path = Path("results/random_search_2015_budget40.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved to: {out_path}")

if __name__ == "__main__":
    main()