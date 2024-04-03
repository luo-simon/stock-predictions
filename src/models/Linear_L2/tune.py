import mlflow

from src.models.Linear_L2.train import train

if __name__ == "__main__":
    experiment_id = mlflow.get_experiment_by_name("linear_l2").experiment_id

    alphas = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]

    for alpha in alphas:
        # Search for runs with the same parameters in this experiment
        query = f"params.alpha = '{alpha}'"
        existing_runs = mlflow.search_runs(experiment_ids=[experiment_id], filter_string=query)
        if not existing_runs.empty:
            print(f"alpha={alpha} has already been run.")
            continue
        print(f"Running alpha={alpha}.")
        train(alpha)
