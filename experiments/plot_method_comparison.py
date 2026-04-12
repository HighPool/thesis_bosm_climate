from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


INPUT_PATH = Path("results/method_comparison_2015.json")
OUTPUT_DIR = Path("results/plots")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def save_bar_chart(names, values, ylabel, title, filename):
    plt.figure(figsize=(10, 6))
    x = np.arange(len(names))
    plt.bar(x, values)
    plt.xticks(x, names, rotation=20, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close()


def save_convergence_plot(curves_dict, title, filename, ylabel="Priemerná najlepšia doterajšia hodnota"):
    plt.figure(figsize=(10, 6))

    for method_name, curve in curves_dict.items():
        x = np.arange(1, len(curve) + 1)
        plt.plot(x, curve, label=method_name)

    plt.xlabel("Počet vyhodnotení")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close()


def save_summary_csv(records, filename="method_summary_table.csv"):
    out_path = OUTPUT_DIR / filename

    headers = [
        "Metóda",
        "Počet problémov",
        "Priemerná odchýlka",
        "Medián odchýlky",
        "Smerodajná odchýlka odchýlky",
        "Úspešnosť",
        "Priemerný počet vyhodnotení do najlepšieho riešenia",
        "Priemerný čas behu [s]",
        "Dĺžka porovnateľnej krivky",
    ]

    lines = [";".join(headers)]

    for r in records:
        row = [
            str(r.get("name", "")),
            str(r.get("n_problems", "")),
            str(r.get("mean_deviation", "")),
            str(r.get("median_deviation", "")),
            str(r.get("std_deviation", "")),
            str(r.get("success_rate", "")),
            str(r.get("mean_evals_to_f_best", "")),
            str(r.get("mean_total_time", "")),
            str(r.get("min_curve_length", "")),
        ]
        lines.append(";".join(row))

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ensure_dir(OUTPUT_DIR)

    payload = load_json(INPUT_PATH)
    records = payload["methods"]
    aligned_curves = payload["aligned_curves"]["curves"]

    names = [r["name"] for r in records]

    mean_deviation = [r["mean_deviation"] for r in records]
    success_rate = [r["success_rate"] for r in records]
    mean_evals_to_best = [r["mean_evals_to_f_best"] for r in records]
    mean_total_time = [r["mean_total_time"] for r in records]

    save_convergence_plot(
        curves_dict=aligned_curves,
        title="Porovnanie konvergencie metód",
        filename="convergence_comparison.png",
    )

    save_bar_chart(
        names=names,
        values=mean_deviation,
        ylabel="Priemerná odchýlka od optima",
        title="Porovnanie priemernej odchýlky od optima",
        filename="mean_deviation_comparison.png",
    )

    save_bar_chart(
        names=names,
        values=success_rate,
        ylabel="Úspešnosť",
        title="Porovnanie úspešnosti metód",
        filename="success_rate_comparison.png",
    )

    save_bar_chart(
        names=names,
        values=mean_evals_to_best,
        ylabel="Priemerný počet vyhodnotení do najlepšieho riešenia",
        title="Porovnanie efektivity metód",
        filename="evals_to_best_comparison.png",
    )

    save_bar_chart(
        names=names,
        values=mean_total_time,
        ylabel="Priemerný čas behu [s]",
        title="Porovnanie priemerného času behu",
        filename="total_time_comparison.png",
    )

    save_summary_csv(records)

    print("Grafy a tabuľka boli uložené do priečinka:")
    print(OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()