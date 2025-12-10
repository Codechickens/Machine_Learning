import numpy as np
import matplotlib.pyplot as plt

class LassoRegression:
	def __init__(self, alpha=1.0, n_iter=1000):
		self.alpha = alpha
		self.n_iter = n_iter
		self.coef_ = None
		self.intercept_ = None

	def soft_threshold(self, rho, lam):
		if rho < -lam:
			return rho + lam
		elif rho > lam:
			return rho - lam
		else:
			return 0.0

	def fit(self, X, y):
		X_b = np.c_[np.ones((X.shape[0], 1)), X]
		n_samples, n_features = X_b.shape
		w = np.zeros(n_features)
		for _ in range(self.n_iter):
			for j in range(n_features):
				X_j = X_b[:, j]
				y_pred = X_b @ w
				residual = y - y_pred + w[j] * X_j
				rho = X_j @ residual
				if j == 0:
					w[j] = rho / (X_j @ X_j)
				else:
					w[j] = self.soft_threshold(rho, self.alpha) / (X_j @ X_j)
		self.intercept_ = w[0]
		self.coef_ = w[1:]
		Wa = np.concatenate(([self.intercept_], self.coef_))
		print("Weights:", Wa)

	def predict(self, X):
		X_b = np.c_[np.ones((X.shape[0], 1)), X]
		w = np.concatenate(([self.intercept_], self.coef_))
		return X_b @ w

	def plot_regression_line(self, X, y):
		if X.ndim == 1 or X.shape[1] == 1:
			plt.scatter(X, y, color='blue', label='Data points')
			X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
			y_plot = self.predict(X_plot)
			plt.plot(X_plot, y_plot, color='red', label='Lasso Regression Line')
			plt.xlabel('Feature')
			plt.ylabel('Target')
			plt.title('Lasso Regression')
			plt.legend()
			plt.show()
		

# Example usage
data = np.loadtxt("C:\\Users\\BACHDO\\Documents\\GitHub\\train_data\\drive-download-20251209T073956Z-1-001\\data2.csv", delimiter=",", skiprows=1)
X = data[:, :2]  # Sử dụng cả X1 và X2 làm feature
y = data[:, -1]

model = LassoRegression(alpha=1.0, n_iter=1000)
model.fit(X, y)

# Predict and plot
model.plot_regression_line(X, y)

