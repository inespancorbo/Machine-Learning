import numpy as np
import util

import matplotlib.pyplot as plt
from linear_model import LinearModel


def main(train_path, eval_path, intercept):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        intercept: add (True) or not (False) intercept to model
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_val, y_val = util.load_dataset(eval_path, add_intercept=False)

    model = LogisticRegression(intercept)
    model.fit(x_train, y_train)

    # lets find the best prob cutoff that gives highest accuracy
    # and plot the decision boundary for that scenario
    prob_cutoff = np.arange(0.1, 1, 0.1)
    accuracies = np.empty(prob_cutoff.shape[0])
    best_accuracy = -1
    best_prob_cutoff = None
    for i, p in enumerate(prob_cutoff):
        y_pred = model.predict(x_val, p)
        accuracies[i] = accuracy(y_val, y_pred)
        if accuracies[i] > best_accuracy:
            best_accuracy = accuracies[i]
            best_prob_cutoff = i

    plt.figure()
    plt.plot(prob_cutoff, accuracies, 'bx--', linewidth=1)
    plt.xlabel('probability cut-off')
    plt.ylabel('accuracy')
    plt.show()

    model.plot_with_decision_boundary(x_val, y_val, prob_cutoff[best_prob_cutoff])
    print(f'Accuracy achieved is {best_accuracy*100: 0.2f} %')


def accuracy(y_val, y_pred):
    return 1 - np.abs(y_val - y_pred).sum() / y_val.shape[0]


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the optimizer."""
    def __init__(self, intercept):
        super(LogisticRegression, self).__init__()
        self.intercept = intercept

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        if self.intercept is True:
            x = util.add_intercept(x)

        g = lambda x: 1 / (1 + np.exp(-x))
        m, n = x.shape
        
        # initialize theta
        if self.theta is None:
            self.theta = np.zeros(n)
        
        # optimize theta
        while True:
            theta = self.theta
            # compute gradient
            G = - (1 / m) * (y - g(x.dot(theta))).dot(x)
            
            # compute H
            x_theta = x.dot(theta)
            H = (1 / m) * g(x_theta).dot(g(1 - x_theta)) * (x.T).dot(x)
            H_inv = np.linalg.inv(H)
            
            # update
            self.theta = theta - H_inv.dot(G)
            
            # if norm is small, terminate
            if np.linalg.norm(self.theta - theta, ord=1) < self.eps:
                break

    def predict(self, x, p=None):
        """Make a prediction given new inputs x.

        Args:
            p: Cut-off probability
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        if self.intercept is True:
            x = util.add_intercept(x)

        g = lambda x: 1 / (1 + np.exp(-x))
        preds = g(x.dot(self.theta))
        if p is not None:
            preds = (preds >= p).astype(np.float64)
        return preds

    def plot_with_decision_boundary(self, x, y, p, alpha=1):
        """ Plot data set with decision boundary

        Args:
            alpha: to include correction factor alpha when data is partially labeled.
            Default is 1 so that we get the boundary decision for vanilla logistic regression.
            x: Inputs of shape (m, n).
            y: Inputs of shape (m,).
        """
        plt.figure()
        plt.plot(x[y == 1, -2], x[y == 1, -1], 'bx', linewidth=2)
        plt.plot(x[y == 0, -2], x[y == 0, -1], 'go', linewidth=2)

        x1 = np.arange(min(x[:, -2]), max(x[:, -2]), 0.01)
        if self.intercept is True:
            x2 = -(np.log(1/p-1) + np.log(2/alpha - 1) + self.theta[0] + self.theta[1] * x1) / self.theta[2]
        else:
            x2 = - (np.log(1/p-1) + np.log(2/alpha - 1) + self.theta[0] * x1) / self.theta[1]

        plt.plot(x1, x2, c='red', linewidth=2)
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.show()


if __name__ == "__main__":

    main('../data/ds1_train.csv', '../data/ds1_valid.csv', True)
