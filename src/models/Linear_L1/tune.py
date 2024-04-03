import mlflow

from src.models.Linear_L1.train import train

if __name__ == "__main__":
    experiment_id = mlflow.get_experiment_by_name("linear_l1").experiment_id

    alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    
    for alpha in alphas:
        # Search for runs with the same parameters in this experiment
        query = f"params.alpha = '{alpha}'"
        existing_runs = mlflow.search_runs(experiment_ids=[experiment_id], filter_string=query)
        if not existing_runs.empty:
            print(f"alpha={alpha} has already been run.")
            continue
        print(f"Running alpha={alpha}.")
        train(alpha)
