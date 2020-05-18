import numpy as np
import util
import random
from scipy import linalg
from linear_model import LinearModel


def main(lr, train_path, eval_path, pred_path):
    """Problem 3(d): Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, label_col='y', add_intercept=False)

    model = PoissonRegression(max_iter=1000000, step_size=lr, eps=1e-4)
    model.fit(x_train, y_train)

    x_val, y_val = util.load_dataset(eval_path, add_intercept=False)
    y_pred = model.predict(x_val)
    np.savetxt(pred_path, np.array([y_pred, y_val]).T, fmt='%1.4e')
    return model.iter


class PoissonRegression(LinearModel):
    @staticmethod
    def sg(x, y, theta):
        """Stochastic gradient, using one sample, chosen randomly

        Args:
            theta: parameter
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        sample = random.randint(1, x.shape[0]-1)
        sg = (y[sample].item()-np.exp(theta.dot(x[sample, :])))*x[sample, :]
        return sg

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        m, n = x.shape
        if self.theta is None:
            self.theta = np.zeros(n)

        for i in range(self.max_iter):
            theta = self.theta
            # grad = self.sg(x, y, self.theta)
            grad = (y-np.exp(x.dot(theta))).dot(x)
            grad = grad/np.linalg.norm(grad)
            self.theta = theta + self.step_size*grad
            self.iter = self.iter + 1
            if np.linalg.norm(self.theta-theta, ord=1) < self.eps:
            #if np.linalg.norm(self.theta) < self.eps:
                break

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        return np.exp(x.dot(self.theta))


if __name__ == "__main__":
    iterations = main(1e-7, "../data/ds4_train.csv", "../data/ds4_valid.csv", "./p03d_poisson_results.csv")
    print(f'Stochastic gradient ascent took {iterations} iterations.')