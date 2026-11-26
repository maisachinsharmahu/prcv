import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)

n = 80
x = rng.uniform(0, 10, n)
y = 4 + 3 * x + rng.normal(0, 2.0, n)

X = np.column_stack([np.ones(n), x])
theta = np.zeros(2)
alpha, epochs = 0.02, 3000
losses = []

for _ in range(epochs):
    error = X @ theta - y
    theta -= alpha * (X.T @ error) / n
    losses.append(np.mean(error ** 2) / 2)

y_hat = X @ theta
ss_res = np.sum((y - y_hat) ** 2)
ss_tot = np.sum((y - y.mean()) ** 2)
r2 = 1 - ss_res / ss_tot

theta_ne = np.linalg.inv(X.T @ X) @ X.T @ y

print(f"Learning rate = {alpha}, epochs = {epochs}")
print(f"Intercept (theta0) by gradient descent : {theta[0]:.4f}")
print(f"Slope     (theta1) by gradient descent : {theta[1]:.4f}")
print(f"Intercept / slope by normal equation   : "
      f"{theta_ne[0]:.4f} / {theta_ne[1]:.4f}")
print(f"Final cost J = {losses[-1]:.4f}")
print(f"R-squared    = {r2:.4f}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(losses, color="darkgreen")
ax[0].set_yscale("log")
ax[0].set_title("Cost J(theta) per epoch (log scale)")
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("cost")
ax[1].scatter(x, y, s=15, label="data")
xs = np.array([0, 10])
label = f"fit: y = {theta[0]:.2f} + {theta[1]:.2f}x"
ax[1].plot(xs, theta[0] + theta[1] * xs, "r", label=label)
ax[1].set_title("Fitted regression line")
ax[1].set_xlabel("x"); ax[1].set_ylabel("y"); ax[1].legend()
plt.tight_layout()
plt.show()
