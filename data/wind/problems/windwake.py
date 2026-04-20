from __future__ import annotations
from pathlib import Path
import numpy as np
import floris.tools as wfct

class WindWakeLayout:
    """
    Lokálna verzia windwake problému pre thesis_bosm_climate.

    Zachováva API:
    - evaluate(x)
    - lbs()
    - ubs()
    - vartype()
    - dims()
    """

    def __init__(
        self,
        file: str | Path,
        n_turbines: int = 3,
        wind_seed: int = 0,
        width: float | None = None,
        height: float | None = None,
        n_samples: int | None = 5,
    ):
        self.file = str(Path(file))
        self.wind_seed = int(wind_seed)
        self.n_turbines = int(n_turbines)

        self.width = float(width) if width is not None else 333.33 * self.n_turbines
        self.height = float(height) if height is not None else 333.33 * self.n_turbines
        self.n_samples = n_samples

        self.wind_rng = np.random.RandomState(self.wind_seed)
        self.wd, self.ws, self.freq = self._gen_random_wind()

        self.fi = wfct.floris_interface.FlorisInterface(self.file)

        rand_layout_x = np.random.uniform(0.0, self.width, size=self.n_turbines)
        rand_layout_y = np.random.uniform(0.0, self.height, size=self.n_turbines)
        self.fi.reinitialize_flow_field(layout_array=(rand_layout_x, rand_layout_y))

        self.boundaries = [
            [0.0, 0.0],
            [self.width, 0.0],
            [self.width, self.height],
            [0.0, self.height],
        ]

        self.aep_initial = 1
        self.lo = wfct.optimization.scipy.layout.LayoutOptimization(
            self.fi,
            self.boundaries,
            self.wd,
            self.ws,
            self.freq,
            self.aep_initial,
        )
        self.min_dist = self.lo.min_dist

    def _gen_random_wind(self):
        rng = self.wind_rng
        wd = np.arange(0.0, 360.0, 5.0)
        ws = 8.0 + rng.randn(len(wd)) * 0.5
        freq = np.abs(np.sort(rng.randn(len(wd))))
        freq = freq / freq.sum()
        return wd, ws, freq

    def evaluate(self, x):
        x = np.asarray(x, dtype=float).reshape(-1)

        if x.shape[0] != self.dims():
            raise ValueError(
                f"Neplatný rozmer x. Očakáva sa {self.dims()}, dostal som {x.shape[0]}"
            )

        c1 = self.lo._space_constraint(x, self.min_dist)
        c2 = self.lo._distance_from_boundaries(x, self.boundaries)

        if c1 < 0 or c2 < 0:
            return 0.0

        if self.n_samples is None:
            obj = self.lo._AEP_layout_opt(x)
        else:
            obj = 0.0
            for _ in range(int(self.n_samples)):
                self.ws = 8.0 + self.wind_rng.randn(len(self.wd)) * 0.5
                self.lo = wfct.optimization.scipy.layout.LayoutOptimization(
                    self.fi,
                    self.boundaries,
                    self.wd,
                    self.ws,
                    self.freq,
                    self.aep_initial,
                )
                obj += self.lo._AEP_layout_opt(x)
            obj = obj / float(self.n_samples)

        return float(obj)

    def lbs(self):
        return np.zeros(2 * self.n_turbines, dtype=float)

    def ubs(self):
        return np.ones(2 * self.n_turbines, dtype=float)

    def vartype(self):
        return np.array(["cont"] * self.dims())

    def dims(self):
        return self.n_turbines * 2

    def __str__(self):
        return (
            f"WindWakeLayout("
            f"file={self.file}, "
            f"n_turbines={self.n_turbines}, "
            f"width={self.width}, "
            f"height={self.height}, "
            f"wind_seed={self.wind_seed}, "
            f"n_samples={self.n_samples})"
        )