from pathlib import Path
import pickle
import json
import numpy as np

from optimizers.pybads_laqn import load_problem, run_pybads_on_problem

def main():
    problem_dir = Path("data/laqn/2015/preprocessed")
    problem_files = sorted(problem_dir.glob("*.p"))

    budget = 1000
    n_runs = 3
    display = "off"

    all_problem_summaries = []

    for problem_file in problem_files:
        problem = load_problem(problem_file)

        deviations = []
        best_values = []
        success_flags = []
        all_curves = []
        evals_to_best_list = []
        times = []

        for run_idx in range(n_runs):
            result = run_pybads_on_problem(
                problem=problem,
                total_budget=budget,
                random_seed=run_idx,
                display=display,
            )

            deviations.append(float(result.deviation_from_optimum))
            best_values.append(float(result.best_y))
            success_flags.append(bool(result.success))
            all_curves.append(result.best_so_far)
            evals_to_best_list.append(int(result.evals_to_f_best))
            times.append(float(result.total_time))

        deviations = np.array(deviations)
        best_values = np.array(best_values)
        success_flags = np.array(success_flags)
        all_curves = np.array(all_curves)
        evals_to_best_list = np.array(evals_to_best_list)
        times = np.array(times)

        summary = {
            "identifier": problem.identifier,
            "n_domain_points": int(problem.domain.shape[0]),
            "budget": budget,
            "n_runs": n_runs,

            "mean_deviation": float(np.mean(deviations)),
            "median_deviation": float(np.median(deviations)),
            "std_deviation": float(np.std(deviations)),

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

    # ===== GLOBAL SUMMARY =====
    mean_deviations = np.array([p["mean_deviation"] for p in all_problem_summaries])
    success_rates = np.array([p["success_rate"] for p in all_problem_summaries])
    mean_best_values = np.array([p["mean_best_y"] for p in all_problem_summaries])
    mean_evals_to_best = np.array([p["mean_evals_to_f_best"] for p in all_problem_summaries])
    mean_times = np.array([p["mean_total_time"] for p in all_problem_summaries])

    min_curve_len = min(len(p["mean_curve"]) for p in all_problem_summaries)
    all_problem_mean_curves = np.array(
        [p["mean_curve"][:min_curve_len] for p in all_problem_summaries],
        dtype=float
    )
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
        "min_curve_length": int(min_curve_len),
    }

    print("\n===== GLOBAL SUMMARY =====")
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
            "display": display,
        },
        "summary": global_summary,
        "results": all_problem_summaries,
    }

    out_path = Path("results/pybads_2015_multirun.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved to: {out_path}")

if __name__ == "__main__":
    main()