import numpy as np
import matplotlib.pyplot as plt

class LassoRegression:

	# Khởi tạo với hệ số alpha và số vòng lặp tối đa
	def __init__(self, alpha=1.0, n_iter=1000):
		self.alpha = alpha
		self.n_iter = n_iter
		self.coef_ = None
		self.intercept_ = None

	# Hàm soft thresholding để cập nhật trọng số
	def soft_threshold(self, rho, lam):
		if rho < -lam:
			return rho + lam
		elif rho > lam:
			return rho - lam
		else:
			return 0.0
		
	# Huấn luyện mô hình sử dụng coordinate descent
	def fit(self, X, y):
		X_b = np.c_[np.ones((X.shape[0], 1)), X]
		n_samples, n_features = X_b.shape # Lấy số mẫu và số đặc trưng
		w = np.zeros(n_features) # Khởi tạo trọng số bằng 0

		# Cập nhật trọng số qua các vòng lặp
		for _ in range(self.n_iter):
			for j in range(n_features):
				X_j = X_b[:, j]
				y_pred = X_b @ w # Dự đoán hiện tại
				residual = y - y_pred + w[j] * X_j #
				rho = X_j @ residual
				if j == 0:
					w[j] = rho / (X_j @ X_j)
				else:
					w[j] = self.soft_threshold(rho, self.alpha) / (X_j @ X_j)
		self.intercept_ = w[0]
		self.coef_ = w[1:]
		print("Lasso regression weights:", w)	
			
	# Dự đoán giá trị mới
	def predict(self, X):
		X_b = np.c_[np.ones((X.shape[0], 1)), X]
		w = np.concatenate(([self.intercept_], self.coef_))
		return X_b @ w

		

# Example usage
data = np.loadtxt("C:\\Users\\BACHDO\\Documents\\GitHub\\train_data\\drive-download-20251209T073956Z-1-001\\data3.csv", delimiter=",", skiprows=2)
X = data[:, 0] 
y = data[:, -1]

model = LassoRegression(alpha=1.0, n_iter=1000)
model.fit(X, y)

# Predict and plot
predictions = model.predict(X)
plt.scatter(X, y, color='blue', label='Dữ liệu đầu vào')
plt.plot(X, predictions, color='red', label='Đường hồi quy')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('Lasso Regression từ con số 0')
plt.legend()
plt.show()