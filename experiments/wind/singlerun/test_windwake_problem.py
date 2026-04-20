import numpy as np
from data.wind.problems.windwake import WindWakeLayout

def main():
    problem = WindWakeLayout(
        file="data/wind/windwake_input.json",
        n_turbines=3,
        wind_seed=0,
        n_samples=5,
    )

    print("dims:", problem.dims())
    print("lbs:", problem.lbs())
    print("ubs:", problem.ubs())

    x = np.array([0.1, 0.1, 0.5, 0.5, 0.9, 0.9], dtype=float)
    y = problem.evaluate(x)

    print("x:", x)
    print("y:", y)


if __name__ == "__main__":
    main()