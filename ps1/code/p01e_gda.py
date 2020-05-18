import numpy as np
import util
import matplotlib.pyplot as plt

from linear_model import LinearModel


def main(train_path, eval_path, intercept):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        intercept: add intercept to model
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_val, y_val = util.load_dataset(eval_path, add_intercept=False)

    model = GDA(intercept)
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


class GDA(LinearModel):
    def __init__(self, intercept):
        super(GDA, self).__init__()
        self.intercept = intercept

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        m, _ = x.shape
        phi = (y == 1).sum() / m
        mu_1 = x[y == 1, :].sum(axis=-2) / (y == 1).sum()
        mu_0 = x[y == 0, :].sum(axis=-2) / (y == 0).sum()
        diff = x.copy()
        diff[y == 1, :] -= mu_1
        diff[y == 0, :] -= mu_0
        sigma = diff.T.dot(diff) / m

        theta = (mu_1-mu_0).T.dot(np.linalg.inv(sigma))
        theta_0 = (1/2) * (mu_0.T.dot(np.linalg.inv(sigma)).dot(mu_0)-mu_1.T.dot(np.linalg.inv(sigma)).dot(mu_1))-np.log((1-phi) / phi)

        if self.intercept is True:
            self.theta = np.insert(theta, 0, theta_0)
        else:
            self.theta = theta


    def predict(self, x, p):
        """Make a prediction given new inputs x.

        Args:
            p: Cut-off probability
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        sigmoid = lambda z: 1 / (1 + np.exp(- z))
        if self.intercept is True:
            x = util.add_intercept(x)

        probs = sigmoid(np.dot(x, self.theta))
        preds = (probs >= p).astype(np.float64)
        return preds

    def plot_with_decision_boundary(self, x, y, p):
        """ Plot data set with decision boundary

        Args:
            x: Inputs of shape (m, n).
            y: Inputs of shape (m,).
        """
        plt.figure()
        plt.plot(x[y == 1, -2], x[y == 1, -1], 'bx', linewidth=2)
        plt.plot(x[y == 0, -2], x[y == 0, -1], 'go', linewidth=2)

        x1 = np.arange(min(x[:, -2]), max(x[:, -2]), 0.01)
        if self.intercept is True:
            x2 = -(np.log(1 / p - 1) + self.theta[0] + self.theta[1] * x1) / self.theta[2]
        else:
            x2 = - (np.log(1 / p - 1) + self.theta[0] * x1) / self.theta[1]

        plt.plot(x1, x2, c='red', linewidth=2)
        plt.title('Validation data with decision boundary')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.show()


if __name__ == "__main__":

    main('../data/ds1_train.csv', '../data/ds1_valid.csv', True)