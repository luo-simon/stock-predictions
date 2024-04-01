from src.models.Linear.data import load_data
from src.misc import split_data, evaluate, plot
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import mlflow

# TODO:
# - fix up training loop (use X_train appropriately)
# - create config, and put in model params
# - separate out train and eval files


def train():
    # # Enable MLflow autologging
    # mlflow.sklearn.autolog()

    # Load dataset
    X, y = load_data()

    # Split
    _, _, y_test = split_data(y, verbose=False)

    # Train
    def linear_model_N(N):
        preds = []
        model = LinearRegression(fit_intercept=True)
        for i in range(int(len(X) * 0.9), len(X)):
            model.fit(X.iloc[i - N : i], y.iloc[i - N : i])
            pred = model.predict(X.iloc[i : i + 1])
            preds.append(pred[0])
        preds = pd.Series(preds, index=y_test.index)
        return preds

    plot(linear_model_N(1), y_test)
    evaluate(linear_model_N(1), y_test, verbose=True)

    # Evaluate
    RMSEs = [0]
    for n in range(1, 80):
        r2, mse, rmse, mae, mape = evaluate(linear_model_N(n), y_test)
        RMSEs.append(rmse)

    # Plot
    plt.plot(RMSEs, "x-")
    plt.grid()
    plt.xlabel("N")
    plt.ylabel("RMSE")
    # plt.xlim([0, 10])
    # plt.ylim([0, 50])
    plt.show()


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Train LSTM model")
    # parser.add_argument("--config-file", "-c", type=str, default='configs/lstm.yaml')
    # args = parser.parse_args()

    # # Load the configuration file:
    # with open(args.config_file, 'r') as file:
    #     config = yaml.safe_load(file)

    train()
