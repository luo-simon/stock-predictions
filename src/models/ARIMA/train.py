from src.models.ARIMA.data import load_data
from src.misc import split_data, evaluate, plot
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd

# TODO: 
# - fix up training loop (use X_train appropriately)
# - create config, and put in model params (p,d,q)
# - separate out train and eval

def train():
    X, y = load_data()

    # Split
    X_train, X_val, X_test = split_data(X, verbose=False)
    y_train, y_val, y_test = split_data(y, verbose=False)

    
    # Train
    history = [y for y in y_train]
    preds = []

    for t in range(len(y_test)):
        model = ARIMA(history, order=(1,1,0))
        model_fit = model.fit()
        output = model_fit.forecast()
        preds.append(output[0])
        history.append(y_test[t])

    preds = pd.Series(preds, index=y_test.index)

    # Evaluate
    evaluate(preds, y_test, verbose=True)

    # Plot
    plot(preds, y_test)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description="Train LSTM model")
    # parser.add_argument("--config-file", "-c", type=str, default='configs/lstm.yaml')
    # args = parser.parse_args()

    # # Load the configuration file:
    # with open(args.config_file, 'r') as file:
    #     config = yaml.safe_load(file)

    train()