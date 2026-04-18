from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from optimizers.pybads_laqn import load_problem, run_pybads_on_problem


def summarize_results(results: list[dict]) -> dict:
    deviations = np.array([r["deviation_from_optimum"] for r in results], dtype=float)
    success = np.array([1.0 if r["success"] else 0.0 for r in results], dtype=float)

    min_len = min(len(r["best_so_far"]) for r in results)
    curves = np.array([r["best_so_far"][:min_len] for r in results], dtype=float)

    return {
        "n_problems": len(results),
        "mean_deviation": float(np.mean(deviations)),
        "median_deviation": float(np.median(deviations)),
        "std_deviation": float(np.std(deviations)),
        "success_rate": float(np.mean(success)),
        "mean_best_so_far_curve": curves.mean(axis=0).tolist(),
        "min_curve_length": int(min_len),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--problems_dir",
        type=str,
        required=True,
        help="Priečinok s .p problémami",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=1000,
        help="Počet unikátnych evaluácií vrátane počiatočných bodov",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/pybads_results.json",
        help="Výstupný JSON súbor",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Voliteľne obmedzí počet problémov",
    )
    parser.add_argument(
        "--display",
        type=str,
        default="off",
        choices=["off", "iter", "final"],
        help="Verbosity pyBADS",
    )
    args = parser.parse_args()

    problems_dir = Path(args.problems_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    problem_files = sorted(problems_dir.glob("*.p"))
    if args.limit is not None:
        problem_files = problem_files[:args.limit]

    if not problem_files:
        raise FileNotFoundError(f"V {problems_dir} som nenašiel žiadne .p súbory")

    all_results = []

    for i, problem_file in enumerate(problem_files, start=1):
        print(f"[{i}/{len(problem_files)}] Running {problem_file.name}")

        problem = load_problem(problem_file)
        result = run_pybads_on_problem(
            problem=problem,
            total_budget=args.budget,
            random_seed=123,
            display=args.display,
        )

        result_dict = result.to_dict()
        result_dict["source_file"] = str(problem_file)
        all_results.append(result_dict)

        print(
            f"  best_y={result.best_y:.6f} | "
            f"optimum={result.optimum:.6f} | "
            f"deviation={result.deviation_from_optimum:.6f} | "
            f"evals={result.eval_count}"
        )

    summary = summarize_results(all_results)

    payload = {
        "config": {
            "problems_dir": str(problems_dir),
            "budget": args.budget,
            "display": args.display,
        },
        "summary": summary,
        "results": all_results,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("\nDone.")
    print(f"Saved to: {output_path}")
    print("Summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()