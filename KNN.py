import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        # Lưu trữ dữ liệu huấn luyện (Lazy Learning)
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
        predictions = [self._predict(x) for x in np.array(X)]
        return np.array(predictions)

    def _predict(self, x):
        # 1. Tính khoảng cách Euclidean dùng Vectorization của Numpy
        distances = np.sqrt(np.sum((self.X_train - x)**2, axis=1))
        
        # 2. Lấy k láng giềng gần nhất (indices)
        k_indices = np.argsort(distances)[:self.k]
        
        # 3. Lấy nhãn của k láng giềng đó
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # 4. Bỏ phiếu số đông (Majority Vote)
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
