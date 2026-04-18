import json
from pathlib import Path
import subprocess
import sys
import os

import numpy as np
import matplotlib.pyplot as plt


def open_file(path: Path):
    """Otvorí súbor v predvolenej aplikácii (neblokuje proces)."""
    try:
        if sys.platform == "darwin":          # macOS
            subprocess.Popen(["open", str(path)])
        elif sys.platform == "win32":         # Windows
            os.startfile(str(path))
        else:                                # Linux
            subprocess.Popen(["xdg-open", str(path)])
    except Exception as e:
        print(f"Warning: nepodarilo sa otvoriť súbor: {e}")


def main():
    plt.style.use("seaborn-v0_8")

    with open("results/results_random_search_2015_multirun.json", "r") as f:
        data = json.load(f)

    all_curves = np.array([p["mean_curve"] for p in data], dtype=float)

    global_mean_curve = np.mean(all_curves, axis=0)
    global_median_curve = np.median(all_curves, axis=0)

    x = np.arange(1, len(global_mean_curve) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(x, global_mean_curve, label="Mean", linewidth=2)
    plt.plot(x, global_median_curve, linestyle="--", label="Median", linewidth=2)

    plt.xlabel("Počet evaluácií")
    plt.ylabel("Best-so-far hodnota")
    plt.title("Random Search – convergence (LAQN 2015)")
    plt.legend()
    plt.grid(True)

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    out_path_png = out_dir / "random_search_convergence.png"
    out_path_svg = out_dir / "random_search_convergence.svg"

    plt.savefig(out_path_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_path_svg, bbox_inches="tight")

    plt.close()

    print(f"Plot saved to: {out_path_png}")
    print(f"Plot saved to: {out_path_svg}")

    # 🔥 automatické otvorenie (PNG stačí)
    open_file(out_path_png)


if __name__ == "__main__":
    main()