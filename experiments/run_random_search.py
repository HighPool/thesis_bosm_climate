from problems.problem_wrapper import SphereProblem
from optimizers.random_search import run_random_search


def main():
    problem = SphereProblem(dim=2)
    result = run_random_search(problem=problem, budget=20, seed=42)

    print("Best y:", result.best_y)
    print("Best x:", result.best_x)
    print("Number of evaluations:", result.n_evals)
    print("Best history:", result.best_hist)


if __name__ == "__main__":
    main()