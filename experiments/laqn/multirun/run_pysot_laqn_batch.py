from __future__ import annotations

import time
import csv
import json
import numpy as np
from pathlib import Path

from optimizers.laqn.pysot_laqn import (
    load_problem,
    run_pysot_laqn,
)

"""
Multirun experiment metódy pySOT na úlohách LAQN
- načíta všetky problémové inštancie z určeného priečinka
- na každej úlohe vykoná viac behov algoritmu
- agreguje metriky cez behy pre každú úlohu zvlášť
- vytvorí globálny súhrn cez všetky úlohy
- uloží výsledky do summary JSON
- uloží detailné run-level JSON súbory po problémoch
- a vytvorí CSV výstup pre IOHanalyzer
"""


def main():
    batch_start = time.perf_counter()

    problem_dir = Path("data/laqn/2015/preprocessed")

    # pySOT konfigurácia
    surrogate_type = "rbf"
    strategy_type = "dycors"

    # Experimentálne nastavenie
    budget = 20
    n_runs = 5

    n_init = 10
    batch_size = 1
    num_cand = 1000
    use_restarts = True
    asynchronous = True
    verbose = False
    extra_config = None

    algorithm_name = f"PySOT-{surrogate_type.upper()}-{strategy_type.upper()}"
    algo_tag = f"pysot_{surrogate_type}_{strategy_type}"
    experiment_tag = f"budget{budget}_runs{n_runs}"

    base_out_dir = Path("results/laqn/final") / experiment_tag / algo_tag
    per_problem_dir = base_out_dir / "per_problem"
    base_out_dir.mkdir(parents=True, exist_ok=True)
    per_problem_dir.mkdir(parents=True, exist_ok=True)

    summary_path = base_out_dir / f"{algo_tag}_summary_2015_budget{budget}_runs{n_runs}.json"
    csv_path = base_out_dir / f"{algo_tag}_ioh_2015_budget{budget}_runs{n_runs}.csv"

    problem_files = sorted(problem_dir.glob("*.p"))
    if not problem_files:
        raise FileNotFoundError(f"V {problem_dir} sa nenašli žiadne .p súbory.")

    all_problem_summaries = []

    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "algorithm_id",
                "problem_id",
                "dimension",
                "run_id",
                "evaluation",
                "raw_y",
                "best_so_far_y",
            ],
        )
        writer.writeheader()

        for problem_idx, problem_file in enumerate(problem_files, start=1):
            print(f"\n[{problem_idx}/{len(problem_files)}] {problem_file.name}", flush=True)

            problem = load_problem(problem_file)

            deviations = []
            best_values = []
            success_flags = []
            all_curves = []
            evals_to_best_list = []
            times = []
            unique_eval_counts = []
            call_counts = []

            run_payloads = []

            for run_idx in range(n_runs):
                print(f"  run {run_idx + 1}/{n_runs}", flush=True)

                result = run_pysot_laqn(
                    problem=problem,
                    surrogate_type=surrogate_type,
                    strategy_type=strategy_type,
                    budget=budget,
                    seed=run_idx,
                    run_id=run_idx + 1,
                    n_init=n_init,
                    batch_size=batch_size,
                    num_cand=num_cand,
                    use_restarts=use_restarts,
                    asynchronous=asynchronous,
                    verbose=verbose,
                    extra_config=extra_config,
                )

                deviations.append(float(result.deviation_from_optimum))
                best_values.append(float(result.best_y))
                success_flags.append(bool(result.success))
                all_curves.append(result.best_so_far)
                evals_to_best_list.append(int(result.evals_to_f_best))
                times.append(float(result.total_time))
                unique_eval_counts.append(int(result.unique_eval_count))
                call_counts.append(int(result.call_count))

                run_payloads.append(result.to_dict())

                for eval_idx, (raw_y, best_y) in enumerate(
                    zip(result.y_hist, result.best_so_far), start=1
                ):
                    writer.writerow(
                        {
                            "algorithm_id": result.algorithm_name,
                            "problem_id": result.problem_id,
                            "dimension": result.dimension,
                            "run_id": result.run_id,
                            "evaluation": eval_idx,
                            "raw_y": float(raw_y),
                            "best_so_far_y": float(best_y),
                        }
                    )

            deviations = np.array(deviations, dtype=float)
            best_values = np.array(best_values, dtype=float)
            success_flags = np.array(success_flags, dtype=float)
            evals_to_best_list = np.array(evals_to_best_list, dtype=float)
            times = np.array(times, dtype=float)
            unique_eval_counts = np.array(unique_eval_counts, dtype=float)
            call_counts = np.array(call_counts, dtype=float)

            min_len = min(len(c) for c in all_curves)
            curves = np.array(
                [np.array(c[:min_len], dtype=float) for c in all_curves],
                dtype=float,
            )

            summary = {
                "identifier": problem.identifier,
                "algorithm_name": algorithm_name,
                "n_domain_points": int(problem.domain.shape[0]),
                "budget": budget,
                "n_runs": n_runs,
                "counting_mode": "algorithm_calls",

                "surrogate_type": surrogate_type,
                "strategy_type": strategy_type,
                "n_init": int(n_init),
                "batch_size": int(batch_size),
                "num_cand": None if num_cand is None else int(num_cand),
                "use_restarts": bool(use_restarts),
                "asynchronous": bool(asynchronous),

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

            per_problem_payload = {
                "algorithm_name": algorithm_name,
                "problem_id": str(problem.identifier),
                "dimension": int(problem.domain.shape[1]),
                "budget": budget,
                "n_runs": n_runs,
                "counting_mode": "algorithm_calls",
                "surrogate_type": surrogate_type,
                "strategy_type": strategy_type,
                "n_init": int(n_init),
                "batch_size": int(batch_size),
                "num_cand": None if num_cand is None else int(num_cand),
                "use_restarts": bool(use_restarts),
                "asynchronous": bool(asynchronous),
                "true_maximum": float(problem.maximum),
                "runs": run_payloads,
            }

            per_problem_path = per_problem_dir / f"{problem.identifier}_runs.json"
            with per_problem_path.open("w", encoding="utf-8") as f:
                json.dump(per_problem_payload, f, indent=2, ensure_ascii=False)

            print(
                f"{problem.identifier} | "
                f"mean_deviation={summary['mean_deviation']:.4f} | "
                f"success_rate={summary['success_rate']:.2f} | "
                f"mean_unique={summary['mean_unique_eval_count']:.2f}",
                flush=True,
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

    batch_total_time = time.perf_counter() - batch_start

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
        "batch_total_time_seconds": float(batch_total_time),
        "batch_total_time_minutes": float(batch_total_time / 60.0),
        "mean_time_per_problem": float(batch_total_time / len(all_problem_summaries)),
    }

    payload = {
        "config": {
            "algorithm_name": algorithm_name,
            "problem_dir": str(problem_dir),
            "budget": budget,
            "n_runs": n_runs,
            "counting_mode": "algorithm_calls",
            "surrogate_type": surrogate_type,
            "strategy_type": strategy_type,
            "n_init": n_init,
            "batch_size": batch_size,
            "num_cand": num_cand,
            "use_restarts": use_restarts,
            "asynchronous": asynchronous,
        },
        "summary": global_summary,
        "results": all_problem_summaries,
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\nSaved summary JSON to: {summary_path}")
    print(f"Saved IOHanalyzer CSV to: {csv_path}")
    print(f"Batch total time: {batch_total_time:.4f} s")
    print(f"Batch total time: {batch_total_time / 60.0:.4f} min")


if __name__ == "__main__":
    main()