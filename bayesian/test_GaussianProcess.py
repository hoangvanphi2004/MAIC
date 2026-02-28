import numpy as np
import torch
import matplotlib.pyplot as plt
from bayesian.GaussianProcessContinuous import GaussianProcess

# Tạo dữ liệu mẫu
np.random.seed(42)
n_train = 50
n_test = 100
x_train = np.random.uniform(-2, 2, n_train).reshape(-1, 1)
y_train = np.sin(x_train.flatten()) + 0.1 * np.random.randn(n_train)

x_test = np.linspace(-3, 3, n_test).reshape(-1, 1)

# Khởi tạo và fit Gaussian Process
gp = GaussianProcess()
gp.fit(x_train, y_train, training_iter=100, lr=0.1)

# Dự đoán
mean, var = gp.predict(x_test)
std = np.sqrt(var)

# Plot kết quả
plt.figure(figsize=(10, 6))
plt.scatter(x_train, y_train, c='red', label='Training data')
plt.plot(x_test, np.sin(x_test.flatten()), 'b-', label='True function')
plt.plot(x_test, mean, 'g-', label='GP mean')
plt.fill_between(x_test.flatten(), mean - 2*std, mean + 2*std, alpha=0.3, color='g', label='95% confidence interval')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Gaussian Process Regression')
plt.show()

# In ra một số giá trị
print("Sample predictions:")
for i in [10, 50, 90]:
    print(f"x={x_test[i,0]:.2f}: mean={mean[i]:.3f}, std={std[i]:.3f}")

# Test sampling
samples = gp.sample(x_test[:5], n_samples=3)
print("\nSamples from GP at first 5 test points:")
print(samples)