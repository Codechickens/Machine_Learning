import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 1. Tải và Chuẩn bị Dữ liệu
# Tải file 'data2.csv'
try:
    df = pd.read_csv('C:\\Users\\BACHDO\\Documents\\GitHub\\train_data\\drive-download-20251209T073956Z-1-001\\data3.csv')
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file 'data2.csv'. Vui lòng đảm bảo file đã được tải lên.")
    exit()

# Phân tách Đặc trưng (X) và Mục tiêu (y)
X = df[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17','X18','X19','X20']]  # Đặc trưng
y = df['y']           # Mục tiêu

feature_names = X.columns.tolist()

# Chia dữ liệu để huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Chuẩn hóa dữ liệu (QUAN TRỌNG cho Ridge và Lasso)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 2. Huấn luyện các mô hình
linear = LinearRegression()
linear.fit(X_train_scaled, y_train)

# Thử nghiệm với các giá trị alpha khác nhau
# Ridge (alpha=10) sẽ thu nhỏ hệ số mạnh hơn Linear
ridge = Ridge(alpha=10) 
ridge.fit(X_train_scaled, y_train)

# Lasso (alpha=0.1) sẽ bắt đầu đặt hệ số bằng 0
lasso = Lasso(alpha=0.1)
lasso.fit(X_train_scaled, y_train)


# 3. Trích xuất hệ số
coefficients = {
    'Linear': linear.coef_,
    'Ridge (alpha=10)': ridge.coef_,
    'Lasso (alpha=0.1)': lasso.coef_
}

# 4. Biểu diễn bằng Matplotlib
fig, ax = plt.subplots(figsize=(8, 5))

x_indices = np.arange(len(feature_names))
width = 0.25 

# Vẽ các cột cho từng mô hình
rects_linear = ax.bar(x_indices - width, coefficients['Linear'], width, label='Linear')
rects_ridge = ax.bar(x_indices, coefficients['Ridge (alpha=10)'], width, label='Ridge (alpha=10)')
rects_lasso = ax.bar(x_indices + width, coefficients['Lasso (alpha=0.1)'], width, label='Lasso (alpha=0.1)')

# Thiết lập biểu đồ
ax.set_ylabel('Giá trị Hệ số (Coefficients)')
ax.set_xlabel('Các Đặc trưng (Features)')
ax.set_title('So sánh Hệ số Hồi quy (Linear vs Ridge vs Lasso) trên data3.csv')
ax.set_xticks(x_indices)
ax.set_xticklabels(feature_names, rotation=45, ha='right')
ax.axhline(0, color='grey', linewidth=0.8) # Đường tham chiếu tại y=0
ax.legend(loc='upper right')
ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# In giá trị hệ số cụ thể (để so sánh dễ hơn)
print("\n--- Bảng Giá trị Hệ số ---")
coef_df = pd.DataFrame(coefficients, index=feature_names)
print(coef_df.round(4))