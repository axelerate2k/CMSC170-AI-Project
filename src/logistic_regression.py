import numpy as np

class LogisticRegression:

    def __init__(self, lr=0.1, epochs=1000):
        #how big each update step is.
        self.lr = lr #learning rate
        #how many times we go over the training data.
        self.epochs = epochs

    def sigmoid(self, z):
        #Squash any number into a probability between 0 and 1.
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Start with all weights and the bias at zero.
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.loss_history = []

        for epoch in range(self.epochs):
            #Predict = probability that Blue wins.
            z = X @ self.w + self.b
            p = self.sigmoid(z)

            #Cross-entropy loss (eps avoids log(0)).
            eps = 1e-12
            loss_win = y * np.log(p + eps)
            loss_lose = (1 - y) * np.log(1 - p + eps)
            loss = -np.mean(loss_win + loss_lose)
            self.loss_history.append(loss)

            #Gradients = how the loss changes
            #  if we nudge each weight.
            error = p - y
            grad_w = (X.T @ error) / n_samples
            grad_b = np.mean(error)

            #Step against the gradient.
            self.w = self.w - self.lr * grad_w
            self.b = self.b - self.lr * grad_b

        return self

    def predict_proba(self, X):
        #Probability that Blue wins, for each game.
        z = X @ self.w + self.b
        return self.sigmoid(z)

    def predict(self, X):
        #1 if the probability is at least 0.5, else 0.
        p = self.predict_proba(X)
        return (p >= 0.5).astype(int)
