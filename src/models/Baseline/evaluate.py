from src.models.Baseline.model import model
from src.models.Baseline.data import load_data
from src.misc import split_data, evaluate, plot


def eval():
    # Load test data
    X, y = load_data()
    
    # Split
    _, _, X_test = split_data(X, verbose=False)
    _, _, y_test = split_data(y, verbose=False)
    
    # Load model
    loaded_model = model
    
    # Predict
    preds = loaded_model(X_test) 

    # Evaluate
    evaluate(preds, y_test, verbose=True)
    
    # Plot
    plot(preds, y_test)

    return preds, y_test

if __name__ == '__main__':
    eval()