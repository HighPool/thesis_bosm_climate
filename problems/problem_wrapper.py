import numpy as np


class SphereProblem:
    """
    Jednoduchý testovací problém:
    f(x) = sum(x_i^2), minimum je v x = 0.
    """

    def __init__(self, dim: int = 2):
        self.dim = dim
        self.lb = -5.0 * np.ones(dim)
        self.ub = 5.0 * np.ones(dim)

    def evaluate(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        return float(np.sum(x ** 2))