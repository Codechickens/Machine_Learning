import numpy as np 
import matplotlib.pyplot as plt

class RidgeRegression:
    def __init__(self, lambda_=1.0):
        self.lambda_ = lambda_
        self.weights = None

    def fit(self, X, y):
        # Add bias term to feature matrix
        X_b = np.c_[np.ones((X.shape[0], 1)), X]

        # Closed-form solution for Ridge Regression and print weights
        n_features = X_b.shape[1]
        I = np.eye(n_features)
        I[0, 0] = 0  # Do not regularize the bias term
        self.weights = np.linalg.inv(X_b.T @ X_b + self.lambda_ * I) @ (X_b.T @ y)
        print("Weights:", self.weights)


    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b @ self.weights

    def plot_regression_line(self, X, y):
        plt.scatter(X, y, color='blue', label='Data points')
        X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_plot = self.predict(X_plot)
        plt.plot(X_plot, y_plot, color='red', label='Ridge Regression Line')
        plt.xlabel('Feature')
        plt.ylabel('Target')
        plt.title('Ridge Regression')
        plt.legend()
        plt.show()

# Example usage
# Generate synthetic data
data = np.loadtxt("C:\\Users\\BACHDO\\Documents\\GitHub\\train_data\\drive-download-20251209T073956Z-1-001\\data2.csv", delimiter=",", skiprows=1)
X = data[:, 0]
y = data[:, -1]
# Fit Ridge Regression model
model = RidgeRegression(lambda_=1.0)
model.fit(X, y)

    # Predict and plot
model.plot_regression_line(X, y)
