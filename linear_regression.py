import numpy as np


class Loss:
    @staticmethod
    def mse(y_true, y_pred):
        # Calculate mean squared error loss
        n = len(y_true)
        loss_value = np.sum(np.square(y_pred - y_true))

        return loss_value

    @staticmethod
    def mse_gradient(X, y_true, y_pred):
        # Compute gradient of MSE with respect to weights
        n = len(y_true)
        grad = (2 / n) * X.T @ (y_pred - y_true)
        return grad


class GradientDescent:
    def __init__(self, learning_rate=0.01, epochs=1000):
        # Initialize hyper‑parameters
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None

    def fit(self,X,y):
        # Training loop for gradient descent optimization
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)

        for i in range(self.epochs):
            y_pred = X @ self.weights
            grad = Loss.mse_gradient(X, y, y_pred)
            # Update parameter along negative gradient direction
            self.weights = self.weights - self.lr * grad

            if i % 100 == 0:
                current_loss = Loss.mse(y, y_pred)
                print(f"Iteration {i} , MSE Loss: {current_loss:.4f}")

    def predict(self, X):
        # Make prediction using trained weights
        return X @ self.weights


if __name__ == '__main__':
    np.random.seed(42)
    n_samples = 500
    X_raw = np.random.randn(n_samples, 3)
    # Append column of ones for bias term
    X = np.hstack([np.ones((n_samples, 1)), X_raw])
    y = 2 + 3 * X_raw[:, 0] - 1.5 * X_raw[:, 1] + 4 * X_raw[:, 2] + np.random.randn(n_samples) * 0.3

    optimizer = GradientDescent(learning_rate=0.02, epochs=1500)
    print("Start Gradient Descent Training for machine_learning Linear Regression")
    optimizer.fit(X, y)

    test_sample = np.array([1, 0.5, -0.3, 0.2])
    pred = optimizer.predict(test_sample)
    print("\nPrediction for test sample:", pred)
    print("Final weights:", optimizer.weights)
