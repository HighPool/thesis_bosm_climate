from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


METHODS = [
    {
        "name": "Random Search",
        "path": "results/random_search_2015_budget40.json",
    },
    {
        "name": "pyBADS",
        "path": "results/pybads_2015_multirun.json",
    },
    # Sem neskôr len doplníš ďalšie metódy, napríklad:
    # {
    #     "name": "BO-GP",
    #     "path": "results/bo_gp_2015.json",
    # },
]


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_summary_value(payload: dict[str, Any], key: str, default=np.nan):
    return payload.get("summary", {}).get(key, default)


def format_number(x: Any, digits: int = 6) -> str:
    try:
        if x is None:
            return "N/A"
        if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
            return "N/A"
        return f"{float(x):.{digits}f}"
    except Exception:
        return "N/A"


def extract_method_record(method_def: dict[str, str]) -> dict[str, Any]:
    payload = load_json(method_def["path"])

    record = {
        "name": method_def["name"],
        "path": method_def["path"],
        "n_problems": get_summary_value(payload, "n_problems"),
        "mean_deviation": get_summary_value(payload, "mean_deviation"),
        "median_deviation": get_summary_value(payload, "median_deviation"),
        "std_deviation": get_summary_value(payload, "std_deviation"),
        "success_rate": get_summary_value(payload, "success_rate"),
        "mean_best_y": get_summary_value(payload, "mean_best_y"),
        "mean_evals_to_f_best": get_summary_value(payload, "mean_evals_to_f_best"),
        "mean_total_time": get_summary_value(payload, "mean_total_time"),
        "min_curve_length": get_summary_value(payload, "min_curve_length"),
        "mean_best_so_far_curve": get_summary_value(payload, "mean_best_so_far_curve", []),
    }

    # Pokus o doplnenie metrík, ak nie sú priamo v summary,
    # ale dajú sa dopočítať z results.
    results = payload.get("results", [])

    if (record["mean_evals_to_f_best"] is np.nan or np.isnan(record["mean_evals_to_f_best"])) and results:
        per_problem = []
        for r in results:
            value = r.get("mean_evals_to_f_best")
            if value is not None:
                per_problem.append(float(value))
        if per_problem:
            record["mean_evals_to_f_best"] = float(np.mean(per_problem))

    if (record["mean_total_time"] is np.nan or np.isnan(record["mean_total_time"])) and results:
        per_problem = []
        for r in results:
            value = r.get("mean_total_time")
            if value is not None:
                per_problem.append(float(value))
        if per_problem:
            record["mean_total_time"] = float(np.mean(per_problem))

    return record


def print_summary_table(records: list[dict[str, Any]]) -> None:
    headers = [
        "Metóda",
        "Počet problémov",
        "Priemerná odchýlka",
        "Medián odchýlky",
        "Smer. odchýlka odchýlky",
        "Úspešnosť",
        "Priemerný počet vyhodnotení do najlepšieho riešenia",
        "Priemerný čas behu [s]",
        "Dĺžka porovnateľnej krivky",
    ]

    rows = []
    for r in records:
        rows.append([
            r["name"],
            str(int(r["n_problems"])) if not np.isnan(r["n_problems"]) else "N/A",
            format_number(r["mean_deviation"]),
            format_number(r["median_deviation"]),
            format_number(r["std_deviation"]),
            format_number(r["success_rate"], digits=4),
            format_number(r["mean_evals_to_f_best"], digits=4),
            format_number(r["mean_total_time"], digits=4),
            str(int(r["min_curve_length"])) if not np.isnan(r["min_curve_length"]) else "N/A",
        ])

    col_widths = []
    for i, h in enumerate(headers):
        width = len(h)
        for row in rows:
            width = max(width, len(row[i]))
        col_widths.append(width)

    def fmt_row(values):
        return " | ".join(v.ljust(col_widths[i]) for i, v in enumerate(values))

    print("\n=== SÚHRNNÁ TABUĽKA METRÍK ===")
    print(fmt_row(headers))
    print("-+-".join("-" * w for w in col_widths))
    for row in rows:
        print(fmt_row(row))


def align_curves(records: list[dict[str, Any]]) -> tuple[int, dict[str, list[float]]]:
    available_lengths = [
        len(r["mean_best_so_far_curve"])
        for r in records
        if isinstance(r["mean_best_so_far_curve"], list) and len(r["mean_best_so_far_curve"]) > 0
    ]

    if not available_lengths:
        return 0, {}

    min_len = min(available_lengths)
    aligned = {
        r["name"]: r["mean_best_so_far_curve"][:min_len]
        for r in records
        if isinstance(r["mean_best_so_far_curve"], list) and len(r["mean_best_so_far_curve"]) >= min_len
    }

    return min_len, aligned


def save_comparison_json(records: list[dict[str, Any]], out_path: str | Path) -> None:
    min_curve_len, aligned_curves = align_curves(records)

    payload = {
        "methods": records,
        "aligned_curves": {
            "min_curve_length": min_curve_len,
            "curves": aligned_curves,
        },
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\nUložené do: {out_path}")


def main():
    records = []

    for method in METHODS:
        path = Path(method["path"])
        if not path.exists():
            print(f"Preskakujem {method['name']}: súbor neexistuje -> {path}")
            continue

        try:
            record = extract_method_record(method)
            records.append(record)
        except Exception as e:
            print(f"Preskakujem {method['name']}: chyba pri načítaní -> {e}")

    if not records:
        raise RuntimeError("Nenašiel som žiadne použiteľné výsledkové JSON súbory.")

    print_summary_table(records)
    save_comparison_json(records, "results/method_comparison_2015.json")


if __name__ == "__main__":
    main()