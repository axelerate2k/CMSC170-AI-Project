"""
knn.py
K-Nearest Neighbors classifier, written from scratch.
NumPy is only used for basic math and sorting.
"""
import numpy as np

class KNN:
    def __init__(self, k=5):
        # number of neighbors that vote; use an odd k so a 50/50 tie cannot happen.
        self.k = k

    def fit(self, X, y):
        # KNN is a "lazy" learner: training just means remembering the training data.
        self.X_train = X
        self.y_train = y
        return self

    def distances(self, X):
        """
        Euclidean distance from each game in X to every training game.

        uses the identity |a - b|^2 = |a|^2 + |b|^2 - 2(a . b)
        so it runs as matrix op instead of a slow double loop.
        """
        test_sq = np.sum(X ** 2, axis=1)
        train_sq = np.sum(self.X_train ** 2, axis=1)

        # turn them into a column and a row so they add up into a full (test x train) table.
        test_sq = test_sq.reshape(-1, 1)
        train_sq = train_sq.reshape(1, -1)

        cross = X @ self.X_train.T
        sq_dist = test_sq + train_sq - 2 * cross

        sq_dist = np.maximum(sq_dist, 0)
        return np.sqrt(sq_dist)

    def predict(self, X, batch_size=500):
        """Majority vote of the k nearest neighbors."""
        predictions = []

        for start in range(0, len(X), batch_size):
            batch = X[start:start + batch_size]
            dist = self.distances(batch)

            for row in dist:
                # row numbers of the k closest games.
                nearest = np.argsort(row)[:self.k]
                votes = self.y_train[nearest]

                # blue wins if more than half of the k neighbors are wins
                win_votes = np.sum(votes)
                if win_votes > self.k / 2:
                    predictions.append(1)
                else:
                    predictions.append(0)

        return np.array(predictions)
