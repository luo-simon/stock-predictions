from sklearn.linear_model import LinearRegression
from src.misc import evaluate

class LinearModel():
    def __init__(self):
        self.model = LinearRegression(fit_intercept=True)
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def test(self, X_test, y_test):
        preds = self.model.predict(X_test)
        r2, mse, rmse, mae, mape = evaluate(preds, y_test, verbose=True)
