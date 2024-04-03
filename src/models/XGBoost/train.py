from src.models.XGBoost.data import load_data
from src.misc import split_data, evaluate, plot
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

# TODO:
# - fix up training loop (use X_train appropriately)
# - create config, and put in model params
# - separate out train and eval files


def train():
    X, y = load_data()

    # Split
    X_train, X_val, X_test = split_data(X, verbose=False)
    y_train, y_val, y_test = split_data(y, verbose=False)

    # Train
    reg = xgb.XGBRegressor(
        base_score=0.5,
        booster="gbtree",
        n_estimators=1000,
        early_stopping_rounds=50,
        objective="reg:linear",
        max_depth=3,
        learning_rate=0.01,
    )
    reg.fit(
        X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=100
    )

    # Feature importance
    fi = pd.DataFrame(
        data=reg.feature_importances_,
        index=reg.feature_names_in_,
        columns=["importance"],
    )
    fi.sort_values("importance").plot(kind="barh", title="Feature Importance")
    plt.show()

    # Predict
    preds = reg.predict(X_test)
    preds = pd.Series(preds, index=y_test.index)

    # Evaluate
    r2, mse, rmse, mae, mape = evaluate(preds, y_test, verbose=True)

    # Plot
    plot(preds, y_test)

    return preds, y_test


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Train LSTM model")
    # parser.add_argument("--config-file", "-c", type=str, default='configs/lstm.yaml')
    # args = parser.parse_args()

    # # Load the configuration file:
    # with open(args.config_file, 'r') as file:
    #     config = yaml.safe_load(file)

    train()
