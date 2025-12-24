from Decision_Tree import DecisionTree
from Random_Forest import RandomForest
from KNN import KNN
from Logistic_Regression import LogisticRegression

import pandas as pd

# Đọc dữ liệu từ file CSV
data = pd.read_csv('C:\\Users\\BACHDO\\Documents\\GitHub\\Machine_Learning\\Project Tuần 3+4\\test.csv')

# Tiền xử lý dữ liệu: Chuyển đổi các biến phân loại thành biến số
data['Sex'] = data['Sex'].replace({'male': 0, 'female': 1})
data['Embarked'] = data['Embarked'].replace({'S': 0, 'C': 1, 'Q': 2})
# Fill missing values for specific columns
data['Age'] = data['Age'].fillna(data['Age'].median())
data['Fare'] = data['Fare'].fillna(data['Fare'].median())
data['Embarked'] = data['Embarked'].fillna(0)  # Most common embarked port

# Chia dữ liệu thành tập đặc trưng và nhãn
X = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']].values
y = data['Survived'].values

# Khởi tạo và huấn luyện mô hình Decision Tree
dt_model = DecisionTree(max_depth=5)
dt_model.fit(X, y)
# Dự đoán với mô hình Decision Tree
dt_predictions = dt_model.predict(X)
print("Decision Tree Predictions:", dt_predictions)

# Khởi tạo và huấn luyện mô hình Random Forest
rf_model = RandomForest(n_trees=10, max_depth=5)
rf_model.fit(X, y)
# Dự đoán với mô hình Random Forest
rf_predictions = rf_model.predict(X)
print("Random Forest Predictions:", rf_predictions)

# Khởi tạo và huấn luyện mô hình KNN
knn_model = KNN(k=5)
knn_model.fit(X, y)
# Dự đoán với mô hình KNN
knn_predictions = knn_model.predict(X)
print("KNN Predictions:", knn_predictions)

# Khởi tạo và huấn luyện mô hình Logistic Regression
lr_model = LogisticRegression(learning_rate=0.01, iterations=1000)
lr_model.fit(X, y)
# Dự đoán với mô hình Logistic Regression
lr_predictions = lr_model.predict(X)
print("Logistic Regression Predictions:", lr_predictions)

