from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from optimizers.turbo_laqn import load_problem, run_turbo_on_problem


def main():
    problem_dir = Path("data/laqn/2015/preprocessed")
    out_path = Path("results/turbo_2015_budget1000.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    problem_files = sorted(problem_dir.glob("*.p"))
    if not problem_files:
        raise FileNotFoundError(f"V {problem_dir} sa nenašli žiadne .p súbory.")

    budget = 1000
    n_runs = 30

    all_problem_summaries = []

    for problem_idx, problem_file in enumerate(problem_files, start=1):
        print(f"\n[{problem_idx}/{len(problem_files)}] {problem_file.name}")

        problem = load_problem(problem_file)

        deviations = []
        best_values = []
        success_flags = []
        all_curves = []
        evals_to_best_list = []
        times = []
        unique_eval_counts = []
        call_counts = []

        for run_idx in range(n_runs):
            print(f"  run {run_idx + 1}/{n_runs}", flush=True)

            result = run_turbo_on_problem(
                problem=problem,
                total_budget=budget,
                random_seed=run_idx,
                verbose=False,
            )

            deviations.append(float(result.deviation_from_optimum))
            best_values.append(float(result.best_y))
            success_flags.append(bool(result.success))
            all_curves.append(result.best_so_far)
            evals_to_best_list.append(int(result.evals_to_f_best))
            times.append(float(result.total_time))
            unique_eval_counts.append(int(result.unique_eval_count))
            call_counts.append(int(result.call_count))

        deviations = np.array(deviations, dtype=float)
        best_values = np.array(best_values, dtype=float)
        success_flags = np.array(success_flags, dtype=float)
        evals_to_best_list = np.array(evals_to_best_list, dtype=float)
        times = np.array(times, dtype=float)
        unique_eval_counts = np.array(unique_eval_counts, dtype=float)
        call_counts = np.array(call_counts, dtype=float)

        min_len = min(len(c) for c in all_curves)
        curves = np.array([np.array(c[:min_len], dtype=float) for c in all_curves], dtype=float)

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

            "mean_unique_eval_count": float(np.mean(unique_eval_counts)),
            "mean_call_count": float(np.mean(call_counts)),

            "mean_curve": np.mean(curves, axis=0).tolist(),
            "median_curve": np.median(curves, axis=0).tolist(),

            "true_maximum": float(problem.maximum),
        }

        all_problem_summaries.append(summary)

        print(
            f"{problem.identifier} | "
            f"mean_deviation={summary['mean_deviation']:.4f} | "
            f"success_rate={summary['success_rate']:.2f} | "
            f"mean_unique={summary['mean_unique_eval_count']:.2f}"
        )

    mean_deviations = np.array([p["mean_deviation"] for p in all_problem_summaries], dtype=float)
    success_rates = np.array([p["success_rate"] for p in all_problem_summaries], dtype=float)
    mean_best_values = np.array([p["mean_best_y"] for p in all_problem_summaries], dtype=float)
    mean_evals_to_best = np.array([p["mean_evals_to_f_best"] for p in all_problem_summaries], dtype=float)
    mean_times = np.array([p["mean_total_time"] for p in all_problem_summaries], dtype=float)
    mean_unique_counts = np.array([p["mean_unique_eval_count"] for p in all_problem_summaries], dtype=float)
    mean_call_counts = np.array([p["mean_call_count"] for p in all_problem_summaries], dtype=float)

    min_curve_len = min(len(p["mean_curve"]) for p in all_problem_summaries)
    all_problem_mean_curves = np.array(
        [p["mean_curve"][:min_curve_len] for p in all_problem_summaries],
        dtype=float,
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
        "mean_unique_eval_count": float(np.mean(mean_unique_counts)),
        "mean_call_count": float(np.mean(mean_call_counts)),
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
    print(f"Mean unique_eval_count: {global_summary['mean_unique_eval_count']:.4f}")
    print(f"Mean call_count: {global_summary['mean_call_count']:.4f}")

    payload = {
        "config": {
            "problem_dir": str(problem_dir),
            "budget": budget,
            "n_runs": n_runs,
            "counting_mode": "algorithm_calls",
        },
        "summary": global_summary,
        "results": all_problem_summaries,
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()