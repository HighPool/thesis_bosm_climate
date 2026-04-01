import numpy as np


def summarize_runs(best_final: np.ndarray, best_curves: np.ndarray) -> dict:
    """
    best_final: shape (n_runs,)
    best_curves: shape (n_runs, budget)
    """
    return {
        "best_final_mean": float(np.mean(best_final)),
        "best_final_median": float(np.median(best_final)),
        "best_final_std": float(np.std(best_final)),
        "mean_curve": np.mean(best_curves, axis=0),
        "median_curve": np.median(best_curves, axis=0),
    }