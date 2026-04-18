from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

OUTPUT_PATH = Path("results/method_comparison_2015.json")

# Uprav len cesty, ak ich máš inde.
METHODS = [
    {
        "name": "Random Search",
        "results_path": Path("results/results_random_search_2015_preprocessed.json"),
    },
    {
        "name": "pyBADS",
        "results_path": Path("results/pybads_2015_full.json"),
    },
    {
        "name": "TuRBO",
        "results_path": Path("results/turbo_laqn/all_results.json"),
    },
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_method_record(name: str, results: list[dict]) -> dict[str, Any]:
    if not results:
        raise ValueError(f"Metóda {name} má prázdny zoznam výsledkov.")

    deviations = np.array([float(r["deviation_from_optimum"]) for r in results], dtype=float)
    successes = np.array([bool(r["success"]) for r in results], dtype=bool)
    evals_to_best = np.array([int(r["evals_to_f_best"]) for r in results], dtype=float)
    total_times = np.array([float(r["total_time"]) for r in results], dtype=float)

    curves = [np.array(r["best_so_far"], dtype=float) for r in results]
    min_curve_length = min(len(c) for c in curves)
    aligned_curves = np.vstack([c[:min_curve_length] for c in curves])
    mean_curve = aligned_curves.mean(axis=0)

    return {
        "name": name,
        "n_problems": int(len(results)),
        "mean_deviation": float(np.mean(deviations)),
        "median_deviation": float(np.median(deviations)),
        "std_deviation": float(np.std(deviations)),
        "success_rate": float(np.mean(successes)),
        "mean_evals_to_f_best": float(np.mean(evals_to_best)),
        "mean_total_time": float(np.mean(total_times)),
        "min_curve_length": int(min_curve_length),
        "mean_convergence_curve": mean_curve.tolist(),
    }


def align_method_curves(method_records: list[dict[str, Any]]) -> dict[str, list[float]]:
    global_min_len = min(int(r["min_curve_length"]) for r in method_records)

    aligned = {}
    for r in method_records:
        curve = np.array(r["mean_convergence_curve"], dtype=float)[:global_min_len]
        aligned[r["name"]] = curve.tolist()

    return aligned


def main():
    method_records: list[dict[str, Any]] = []

    for spec in METHODS:
        name = spec["name"]
        path = spec["results_path"]

        if not path.exists():
            raise FileNotFoundError(
                f"Pre metódu '{name}' sa nenašiel súbor: {path}"
            )

        results = load_json(path)
        record = compute_method_record(name=name, results=results)
        method_records.append(record)

    aligned_curves = align_method_curves(method_records)

    payload = {
        "methods": method_records,
        "aligned_curves": {
            "length": min(r["min_curve_length"] for r in method_records),
            "curves": aligned_curves,
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("Vytvorený súbor:")
    print(OUTPUT_PATH.resolve())

    print("\n=== SÚHRNNÁ TABUĽKA METRÍK ===")
    header = (
        f"{'Metóda':<14} | {'Počet problémov':<14} | {'Priemerná odchýlka':<18} | "
        f"{'Medián odchýlky':<16} | {'Smer. odchýlka odchýlky':<25} | "
        f"{'Úspešnosť':<10} | {'Priemerný počet vyhodnotení do najlepšieho riešenia':<53} | "
        f"{'Priemerný čas behu [s]':<24} | {'Dĺžka porovnateľnej krivky':<28}"
    )
    print(header)
    print("-" * len(header))

    for r in method_records:
        print(
            f"{r['name']:<14} | "
            f"{r['n_problems']:<14} | "
            f"{r['mean_deviation']:<18.6f} | "
            f"{r['median_deviation']:<16.6f} | "
            f"{r['std_deviation']:<25.6f} | "
            f"{r['success_rate']:<10.4f} | "
            f"{r['mean_evals_to_f_best']:<53.2f} | "
            f"{r['mean_total_time']:<24.6f} | "
            f"{r['min_curve_length']:<28}"
        )


if __name__ == "__main__":
    main()