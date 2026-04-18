from experiments.batch_random_search_laqn_multirun import main as run_experiment
from experiments.plot_random_search_convergence import main as plot_results


def main():
    print("=== STEP 1: Running random search experiment ===")
    run_experiment()

    print("\n=== STEP 2: Plotting results ===")
    plot_results()

    print("\nPipeline finished successfully.")


if __name__ == "__main__":
    main()