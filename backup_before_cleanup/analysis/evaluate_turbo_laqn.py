from __future__ import annotations

import json
from pathlib import Path
import numpy as np


def load_results(path: str | Path) -> list[dict]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_summary(results: list[dict]) -> dict:
    if not results:
        raise ValueError("Zoznam výsledkov je prázdny.")

    deviations = np.array([float(r["deviation_from_optimum"]) for r in results], dtype=float)
    successes = np.array([bool(r["success"]) for r in results], dtype=bool)
    evals_to_best = np.array([int(r["evals_to_f_best"]) for r in results], dtype=float)
    times = np.array([float(r["total_time"]) for r in results], dtype=float)

    # priemerná konvergenčná krivka
    curves = [np.array(r["best_so_far"], dtype=float) for r in results]
    min_len = min(len(c) for c in curves)
    aligned_curves = np.vstack([c[:min_len] for c in curves])
    mean_curve = aligned_curves.mean(axis=0)

    summary = {
        "method": "TuRBO",
        "n_problems": int(len(results)),
        "mean_deviation": float(np.mean(deviations)),
        "median_deviation": float(np.median(deviations)),
        "std_deviation": float(np.std(deviations)),
        "success_rate": float(np.mean(successes)),
        "n_successes": int(np.sum(successes)),
        "mean_evals_to_best": float(np.mean(evals_to_best)),
        "mean_runtime_sec": float(np.mean(times)),
        "comparable_curve_length": int(min_len),
        "mean_convergence_curve": mean_curve.tolist(),
    }
    return summary


def print_summary(summary: dict) -> None:
    print("\n=== SÚHRN TUrBO NA LAQN ===")
    print(f"Počet problémov: {summary['n_problems']}")
    print(f"Priemerná odchýlka od optima: {summary['mean_deviation']:.6f}")
    print(f"Medián odchýlky: {summary['median_deviation']:.6f}")
    print(f"Smerodajná odchýlka odchýlky: {summary['std_deviation']:.6f}")
    print(f"Úspešnosť: {summary['success_rate'] * 100:.2f}% ({summary['n_successes']} / {summary['n_problems']})")
    print(f"Priemerný počet vyhodnotení do najlepšieho riešenia: {summary['mean_evals_to_best']:.2f}")
    print(f"Priemerný čas behu [s]: {summary['mean_runtime_sec']:.6f}")
    print(f"Dĺžka porovnateľnej krivky: {summary['comparable_curve_length']}")


def save_summary(summary: dict, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def main():
    results_path = Path("results/turbo_laqn/all_results.json")
    out_path = Path("results/turbo_laqn/summary.json")

    results = load_results(results_path)
    summary = compute_summary(results)

    print_summary(summary)
    save_summary(summary, out_path)

    print(f"\nUložené do: {out_path}")


if __name__ == "__main__":
    main()