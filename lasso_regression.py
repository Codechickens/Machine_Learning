import numpy as np
import matplotlib.pyplot as plt

def soft_threshold(z, alpha):
    """Hàm soft-thresholding dùng trong Lasso."""
    if z > alpha:
        return z - alpha
    elif z < -alpha:
        return z + alpha
    else:
        return 0.0

class LassoRegression:
    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-6):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)

        # Precompute: norm squared of each column
        X_norm_sq = np.sum(X ** 2, axis=0)

        for iteration in range(self.max_iter):
            w_old = self.w.copy()

            for j in range(n_features):
                # Compute partial residual excluding feature j
                y_pred = X @ self.w
                residual = y - y_pred + self.w[j] * X[:, j]

                # Compute rho
                rho = np.dot(X[:, j], residual) / n_samples

                # Update weight using soft-thresholding
                self.w[j] = soft_threshold(rho, self.alpha) / (X_norm_sq[j] / n_samples)

            # Check convergence
            if np.linalg.norm(self.w - w_old, ord=1) < self.tol:
                break

    def predict(self, X):
        return X @ self.w
    
    def plot_regression_line(self, X, y):
        plt.scatter(X, y, color='blue', label='Data points')
        X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_plot = self.predict(X_plot)
        plt.plot(X_plot, y_plot, color='red', label='Lasso Regression Line')
        plt.xlabel('Feature')
        plt.ylabel('Target')
        plt.title('Lasso Regression')
        plt.legend()
        plt.show()

data = np.loadtxt("C:\\Users\\BACHDO\\Documents\\GitHub\\train_data\\drive-download-20251209T073956Z-1-001\\data2.csv", delimiter=",", skiprows=1)
X = data[:, :1]  # Use both X1 and X2 as features
y = data[:, 2]   # Use y as target

model = LassoRegression(alpha=1.0)
model.fit(X, y)

print("Weights:", model.w)

    # Predict and plot
model.plot_regression_line(X, y)
