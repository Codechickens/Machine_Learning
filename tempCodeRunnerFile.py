# Predict and plot
predictions = model.predict(X)
plt.scatter(X, y, color='blue', label='Dữ liệu đầu vào')
plt.plot(X, predictions, color='red', label='Đường hồi quy')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('Lasso Regression từ con số 0')
plt.legend()
plt.show()