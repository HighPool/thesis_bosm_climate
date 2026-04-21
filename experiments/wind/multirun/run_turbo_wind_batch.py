from __future__ import annotations

import time
import csv
import json
from pathlib import Path
import numpy as np

from optimizers.wind.turbo_wind import (
    make_windwake_problem,
    run_turbo_wind,
)


def main():
    batch_start = time.perf_counter()

    file_path = Path("data/wind/windwake_input.json")

    n_turbines = 3
    wind_seed = 0
    n_samples = 5

    budget = 20
    n_runs = 20
    algorithm_name = "TuRBO"

    n_init = 5
    batch_size = 1
    use_ard = True
    n_training_steps = 50
    device = "cpu"
    dtype = "float64"

    experiment_tag = f"budget{budget}_runs{n_runs}"

    base_out_dir = Path("results/wind/final") / experiment_tag / "turbo"
    per_problem_dir = base_out_dir / "per_problem"
    base_out_dir.mkdir(parents=True, exist_ok=True)
    per_problem_dir.mkdir(parents=True, exist_ok=True)

    summary_path = (
        base_out_dir
        / f"turbo_summary_wind_n{n_turbines}_budget{budget}_runs{n_runs}.json"
    )
    csv_path = (
        base_out_dir
        / f"turbo_ioh_wind_n{n_turbines}_budget{budget}_runs{n_runs}.csv"
    )

    best_values = []
    all_curves = []
    evals_to_best_list = []
    times = []
    run_payloads = []

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

        for run_idx in range(n_runs):
            print(f"run {run_idx + 1}/{n_runs}", flush=True)

            problem = make_windwake_problem(
                file_path=file_path,
                n_turbines=n_turbines,
                wind_seed=wind_seed,
                n_samples=n_samples,
            )

            result = run_turbo_wind(
                problem=problem,
                budget=budget,
                seed=run_idx,
                run_id=run_idx + 1,
                n_init=n_init,
                batch_size=batch_size,
                use_ard=use_ard,
                n_training_steps=n_training_steps,
                device=device,
                dtype=dtype,
                verbose=False,
            )

            best_values.append(float(result.best_y))
            all_curves.append(result.best_so_far)
            evals_to_best_list.append(int(result.evals_to_f_best))
            times.append(float(result.total_time))
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

    best_values = np.array(best_values, dtype=float)
    evals_to_best_list = np.array(evals_to_best_list, dtype=float)
    times = np.array(times, dtype=float)

    min_len = min(len(c) for c in all_curves)
    curves = np.array(
        [np.array(c[:min_len], dtype=float) for c in all_curves],
        dtype=float,
    )

    problem_id = f"windwake_n{n_turbines}_wseed{wind_seed}"

    summary = {
        "identifier": problem_id,
        "algorithm_name": algorithm_name,
        "budget": budget,
        "n_runs": n_runs,
        "counting_mode": "algorithm_calls",
        "dimension": int(problem.dims()),
        "file": str(file_path),
        "n_turbines": int(n_turbines),
        "wind_seed": int(wind_seed),
        "n_samples": None if n_samples is None else int(n_samples),
        "n_init": int(n_init),
        "batch_size": int(batch_size),
        "use_ard": bool(use_ard),
        "n_training_steps": int(n_training_steps),
        "device": str(device),
        "dtype": str(dtype),
        "mean_best_y": float(np.mean(best_values)),
        "median_best_y": float(np.median(best_values)),
        "std_best_y": float(np.std(best_values)),
        "mean_evals_to_f_best": float(np.mean(evals_to_best_list)),
        "median_evals_to_f_best": float(np.median(evals_to_best_list)),
        "std_evals_to_f_best": float(np.std(evals_to_best_list)),
        "mean_total_time": float(np.mean(times)),
        "std_total_time": float(np.std(times)),
        "mean_curve": np.mean(curves, axis=0).tolist(),
        "median_curve": np.median(curves, axis=0).tolist(),
    }

    per_problem_payload = {
        "algorithm_name": algorithm_name,
        "problem_id": problem_id,
        "dimension": int(problem.dims()),
        "budget": budget,
        "n_runs": n_runs,
        "counting_mode": "algorithm_calls",
        "file": str(file_path),
        "n_turbines": int(n_turbines),
        "wind_seed": int(wind_seed),
        "n_samples": None if n_samples is None else int(n_samples),
        "n_init": int(n_init),
        "batch_size": int(batch_size),
        "use_ard": bool(use_ard),
        "n_training_steps": int(n_training_steps),
        "device": str(device),
        "dtype": str(dtype),
        "runs": run_payloads,
    }

    per_problem_path = per_problem_dir / f"{problem_id}_runs.json"
    with per_problem_path.open("w", encoding="utf-8") as f:
        json.dump(per_problem_payload, f, indent=2, ensure_ascii=False)

    batch_total_time = time.perf_counter() - batch_start

    global_summary = {
        "n_problems": 1,
        "mean_best_y": float(summary["mean_best_y"]),
        "median_best_y": float(summary["median_best_y"]),
        "std_best_y": float(summary["std_best_y"]),
        "mean_evals_to_f_best": float(summary["mean_evals_to_f_best"]),
        "mean_total_time": float(summary["mean_total_time"]),
        "mean_best_so_far_curve": summary["mean_curve"],
        "min_curve_length": int(min_len),
        "batch_total_time_seconds": float(batch_total_time),
        "batch_total_time_minutes": float(batch_total_time / 60.0),
        "mean_time_per_problem": float(batch_total_time),
    }

    payload = {
        "config": {
            "algorithm_name": algorithm_name,
            "file": str(file_path),
            "n_turbines": n_turbines,
            "wind_seed": wind_seed,
            "n_samples": n_samples,
            "budget": budget,
            "n_runs": n_runs,
            "counting_mode": "algorithm_calls",
            "n_init": n_init,
            "batch_size": batch_size,
            "use_ard": use_ard,
            "n_training_steps": n_training_steps,
            "device": device,
            "dtype": dtype,
        },
        "summary": global_summary,
        "results": [summary],
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\nSaved per-problem JSON to: {per_problem_path}")
    print(f"Saved summary JSON to: {summary_path}")
    print(f"Saved IOHanalyzer CSV to: {csv_path}")
    print(f"Batch total time: {batch_total_time:.4f} s")
    print(f"Batch total time: {batch_total_time / 60.0:.4f} min")


if __name__ == "__main__":
    main()