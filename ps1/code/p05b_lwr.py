import matplotlib.pyplot as plt
import numpy as np
import util

from linear_model import LinearModel


def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_val, y_val = util.load_dataset(eval_path, add_intercept=True)

    model = LocallyWeightedLinearRegression(tau)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_val)

    mse = ((y_pred-y_val)**2).mean()
    print(f'The MSE is {mse}.')

    plt.figure()
    plt.plot(x_val, y_val, 'bx')
    plt.plot(x_val, y_pred, 'ro')

class LocallyWeightedLinearRegression(LinearModel):
    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        Args:
            x: Inputs of shape (m, n).
            y: Inputs of shape (m,)
        """
        self.x = x
        self.y = y

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        m, n = x.shape
        f = lambda z: np.exp(- (z**2) / (2*self.tau**2))
        w = f(np.linalg.norm(self.x[None] - x[:, None], axis=2))

        y_pred = np.ndarray(m)
        for i, W in enumerate(w):
            W = np.diag(W)
            theta = np.linalg.inv(self.x.T.dot(W).dot(self.x)).dot(self.x.T).dot(W).dot(self.y)
            y_pred[i] = x[i].dot(theta)

        return y_pred


