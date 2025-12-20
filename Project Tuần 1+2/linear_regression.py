# khai báo các thư viện cần thiết

import numpy as np 
import matplotlib.pyplot as plt

# Định nghĩa lớp LinearRegression từ đầu

class LinearRegression:
    def __init__(self):
        self.w = None

    def fit(self, X, y):
        # Thêm cột bias (hệ số chặn) vào ma trận X

        X_b = np.c_[np.ones((X.shape[0], 1)), X]

        # Công thức nghiệm đóng để tính trọng số: (X_b^T * X_b)^-1 * X_b^T * y

        self.w = np.linalg.inv(X_b.T @ X_b ) @ (X_b.T @ y)
        print("Linear regression weights:", self.w)


    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return np.dot(X_b, self.w)


# Tải dữ liệu từ file CSV
data = np.loadtxt("C:\\Users\\BACHDO\\Documents\\GitHub\\train_data\\drive-download-20251209T073956Z-1-001\\data2.csv", delimiter=",", skiprows=1)
X = data[:, 0] #Dùng cột X1 làm biến độc lập  
y = data[:, -1] #Dùng cột Y làm biến phụ thuộc

model = LinearRegression()
model.fit(X, y)

