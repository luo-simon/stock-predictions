import matplotlib.pyplot as plt
import numpy as np

from src.models.Linear.data import LinearDataModule
from src.models.Linear.model import LinearModel


def train(stock, dataset_len, feature_set, hparams):
    datamodule = LinearDataModule(
        stock = stock,
        feature_set = feature_set,
        dataset_len = dataset_len
    )

    model = LinearModel()

    model.train(*datamodule.train_dataset)
    model.test(*datamodule.test_dataset)
    
    features_coeffs = np.vstack((feature_set, model.model.coef_)).T
    features_coeffs = sorted(
        features_coeffs, key=lambda x: abs(float(x[1])), reverse=False
    )
    feature_names, coeffs = zip(*features_coeffs)
    
    plt.figure(figsize=(12, 10))
    plt.barh(feature_names, coeffs)
    plt.ylabel("Features")
    plt.xlabel("Coefficient Value")
    plt.title("Feature Coefficients from Linear Regression")
    plt.show()
    # plot_path = "/Users/simon/Documents/II/Dissertation/figures/linear.png"
    # plt.savefig(plot_path)

if __name__ == "__main__":
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    train(
        stock="brk-b",
        dataset_len=1,
        feature_set=features,
        hparams=None
    )
