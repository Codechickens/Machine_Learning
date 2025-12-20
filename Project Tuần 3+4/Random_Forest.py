import numpy as np
from tree_model import DecisionTree

class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            # Bootstrapping: Lấy mẫu ngẫu nhiên có lặp lại
            indices = np.random.choice(X.shape[0], X.shape[0], replace=True)
            tree.fit(X[indices], y[indices])
            self.trees.append(tree)

    def predict(self, X):
        # Lấy dự đoán từ tất cả các cây
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        # tree_preds có hình dạng (n_trees, n_samples)
        
        # Bỏ phiếu số đông cho mỗi mẫu
        predictions = []
        for i in range(X.shape[0]):
            sample_preds = tree_preds[:, i]
            most_common = Counter(sample_preds).most_common(1)[0][0]
            predictions.append(most_common)
        return np.array(predictions)
