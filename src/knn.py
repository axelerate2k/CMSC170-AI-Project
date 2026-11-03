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

