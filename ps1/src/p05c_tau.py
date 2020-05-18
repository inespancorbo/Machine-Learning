import matplotlib.pyplot as plt
import numpy as np
import util

from p05b_lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_val, y_val = util.load_dataset(valid_path, add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)

    best_tau = None
    best_mse = float('inf')

    # finding best tau based on lowest MSE on valid data
    for tau in tau_values:
        model = LocallyWeightedLinearRegression(tau)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_val)
        mse = ((y_pred-y_val) ** 2).mean()
        plt.figure()
        plt.title(f'$tau = {tau}$, MSE = {mse: 0.4f}')
        plt.plot(x_train, y_train, 'bx', x_val, y_pred, 'ro')
        if mse < best_mse:
            best_tau = tau
            best_mse = mse

    # getting mse of test
    model = LocallyWeightedLinearRegression(best_tau)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    mse = ((y_pred - y_val) ** 2).mean()
    print(f'Test MSE: {mse:0.4f} || Best tau based on tuning on valid data: {best_tau}.')



