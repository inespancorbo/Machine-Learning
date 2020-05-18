import numpy as np
import util

from p01b_logreg import LogisticRegression, accuracy


def main(train_path, valid_path, test_path, condition, intercept):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on t-labels,
        3. on t-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
    """
    x_test, y_test_y = util.load_dataset(test_path, label_col='y', add_intercept=False)
    _, y_test_t = util.load_dataset(test_path, label_col='t', add_intercept=False)

    # logistic regression on y-labels or t_labels with correction factor alpha
    if condition == 1 or condition == 3:
        x_train, y_train = util.load_dataset(train_path, label_col='y', add_intercept=False)

    # logistic regression on t-labels
    elif condition == 2:
         x_train, y_train = util.load_dataset(train_path, label_col='t', add_intercept=False)

    else:
        return "Wrong condition: expecting 1, 2, or 3."

    model = LogisticRegression(intercept)
    model.fit(x_train, y_train)

    # for purposes of this exercise, let us just use a probability cut-off of 50%
    if condition == 3:
        y_pred = model.predict(x_test)
        alpha = y_pred[y_test_y == 1].sum() / (y_test_y == 1).sum()
        y_pred = y_pred / alpha
        y_pred = (y_pred >= 0.5).astype(np.float64)
        model.plot_with_decision_boundary(x_test, y_test_t, 0.5, alpha)
    else:
        y_pred = model.predict(x_test, 0.5)
        model.plot_with_decision_boundary(x_test, y_test_t, 0.5)

    acc = accuracy(y_pred, y_test_t)
    print(f'Accuracy achieved is {acc * 100: 0.2f} %')


if __name__ == "__main__":

    main('../data/ds3_train.csv', '../data/ds3_valid.csv', '../data/ds3_test.csv', 2, True)




