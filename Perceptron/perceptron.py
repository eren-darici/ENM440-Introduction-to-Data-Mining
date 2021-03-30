import numpy as np


class Perceptron(object):
    """"
    Perceptron Classifier.
    Parameters
    ----------
    eta : float, learning rate (between 0.0 and 1.0)
    n_iter: int, passes over the training dataset.

    Attributes
    ----------
    w_: 1d-array, weights after fitting
    errors_: list, number of misclassifications in every epoch
    """

    def __init__(self, eta=0.01, n_iter=10, random_state=42):
        self.eta = eta
        self.n_iter = n_iter
        self.w_ = np.ndarray
        self.errors_ = list()
        np.random.seed(random_state)

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

    def fit(self, X, y):
        self.w_ = np.random.uniform(0, 1.0, size=(1+X.shape[1], ))
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def accuracy(self, y_pred, y_true):
        correct = 0

        for idx in range(len(y_pred)):
            if y_pred[idx] == y_true[idx]:
                correct += 1
        return correct / len(y_pred)