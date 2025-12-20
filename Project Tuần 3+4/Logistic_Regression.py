import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.lr = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    def _sigmoid(self, z):
        # Giới hạn giá trị z để tránh lỗi tràn số (overflow) với hàm exp
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        # Ép kiểu dữ liệu numpy để đảm bảo tính toán chính xác
        X = np.array(X)
        y = np.array(y)
        m, n = X.shape
        
        # Khởi tạo tham số ban đầu
        self.weights = np.zeros(n)
        self.bias = 0.0

        for i in range(self.iterations):
            # 1. Tính toán giá trị dự đoán (Forward)
            z = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(z)

            # 2. Tính hàm mất mát (Log Loss)
            loss = - (1 / m) * np.sum(y * np.log(y_pred + 1e-15) + (1 - y) * np.log(1 - y_pred + 1e-15))
            self.loss_history.append(loss)

            # 3. Tính đạo hàm (Backward)
            dw = (1 / m) * np.dot(X.T, (y_pred - y))
            db = (1 / m) * np.sum(y_pred - y)

            # 4. Cập nhật trọng số
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def get_weights(self):
        """Trả về trọng số và hệ số chặn dưới dạng dictionary"""
        return {
            "weights": self.weights,
            "bias": self.bias
        }

    def predict_proba(self, X):
        X = np.array(X)
        z = np.dot(X, self.weights) + self.bias
        return self._sigmoid(z)

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)
