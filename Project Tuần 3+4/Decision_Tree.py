import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature     # Chỉ số cột đặc trưng
        self.threshold = threshold # Giá trị ngưỡng để chia
        self.left = left           # Nhánh trái
        self.right = right         # Nhánh phải
        self.value = value         # Giá trị nhãn (chỉ dành cho nút lá)

class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def _gini(self, y):
        m = len(y)
        if m == 0: return 0
        return 1.0 - sum((np.sum(y == c) / m) ** 2 for c in np.unique(y))

    def fit(self, X, y):
        self.root = self._grow_tree(np.array(X), np.array(y))

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Điều kiện dừng
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = Counter(y).most_common(1)[0][0]
            return Node(value=leaf_value)

        # Tìm điểm chia tốt nhất
        best_feat, best_thresh = self._best_split(X, y, n_features)
        
        # If no good split found, create leaf
        if best_feat is None:
            leaf_value = Counter(y).most_common(1)[0][0]
            return Node(value=leaf_value)

        # Tạo cây con (Recursive)
        left_idxs = X[:, best_feat] <= best_thresh
        right_idxs = ~left_idxs
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feat, best_thresh, left, right)

    def _best_split(self, X, y, n_features):
        best_gini = float('inf')
        split_idx, split_thresh = None, None
        
        for feat_idx in range(n_features):
            unique_vals = np.unique(X[:, feat_idx])
            if len(unique_vals) == 1:
                continue
                
            for threshold in unique_vals:
                left_mask = X[:, feat_idx] <= threshold
                right_mask = ~left_mask
                
                left_y = y[left_mask]
                right_y = y[right_mask]
                
                m, m_l, m_r = len(y), len(left_y), len(right_y)
                if m_l == 0 or m_r == 0: continue
                
                current_gini = (m_l/m) * self._gini(left_y) + (m_r/m) * self._gini(right_y)
                if current_gini < best_gini:
                    best_gini, split_idx, split_thresh = current_gini, feat_idx, threshold
        
        return split_idx, split_thresh

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in np.array(X)])

    def _traverse_tree(self, x, node):
        if node.value is not None: return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)