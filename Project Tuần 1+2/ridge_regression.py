import numpy as np
import matplotlib.pyplot as plt

class RidgeRegression:
    def __init__(self, lambda_=1.0):
        self.lambda_ = lambda_
        self.weights = None

    def fit(self, X, y):
        # Thêm cột bias vào ma trận X
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        n_features = X_b.shape[1]
        # Tạo ma trận đơn vị (loại trừ hệ số bias)
        I = np.eye(n_features)
        I[0, 0] = 0
        # Tính toán trọng số bằng công thức đóng
        self.weights = np.linalg.inv(X_b.T @ X_b + self.lambda_ * I) @ (X_b.T @ y)
        print("Ridge regression weights:", self.weights)

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b @ self.weights
    
data = np.loadtxt("C:\\Users\\BACHDO\\Documents\\GitHub\\train_data\\drive-download-20251209T073956Z-1-001\\data3.csv", delimiter=",", skiprows=2)
X = data[:, 0]  # Sử dụng cột 7 làm biến độc lập
y = data[:, -1] # Sử dụng cột cuối cùng làm biến phụ thuộc

model = RidgeRegression(lambda_=1.0)
model.fit(X, y)

