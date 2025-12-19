import numpy as np 
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self):
        self.weights = None

    def fit(self, X, y):
        # Add bias term to feature matrix
        X_b = np.c_[np.ones((X.shape[0], 1)), X]

        # Closed-form solution for Linear Regression and print weights
        self.weights = np.linalg.inv(X_b.T @ X_b ) @ (X_b.T @ y)
        print("Weights:", self.weights)


    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b @ self.weights

    def plot_regression_line(self, X, y):
        plt.scatter(X, y, color='blue', label='Data points')
        X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_plot = self.predict(X_plot)
        plt.plot(X_plot, y_plot, color='red', label='Linear Regression Line')
        plt.xlabel('Feature')
        plt.ylabel('Target')
        plt.title('Linear Regression')
        plt.legend()
        plt.show()

# Example usage
# Generate synthetic data
data = np.loadtxt("C:\\Users\\BACHDO\\Documents\\GitHub\\train_data\\drive-download-20251209T073956Z-1-001\\data2.csv", delimiter=",", skiprows=1)
X = data[:, 0]  # Use both X1 and X2 as features
y = data[:, -1]

model = LinearRegression()
model.fit(X, y)

    # Predict and plot
model.plot_regression_line(X, y)
