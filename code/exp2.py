import numpy as np
import matplotlib.pyplot as plt

def cost(w):
    return (w - 5) ** 2 + 2

def grad(w):
    return 2 * (w - 5)

def gradient_descent(w0, lr, steps):
    w = w0
    path = [w]
    for _ in range(steps):
        w = w - lr * grad(w)
        path.append(w)
    return np.array(path)

w_start, lr, steps = -4.0, 0.2, 15
path = gradient_descent(w_start, lr, steps)

print(f"Start = {w_start}, learning rate = {lr}, iterations = {steps}")
print("-" * 44)
print(f"{'Iter':>4} | {'w':>10} | {'J(w)':>10} | {'dJ/dw':>10}")
print("-" * 44)
for i, w in enumerate(path):
    if i % 3 == 0 or i == steps:
        print(f"{i:>4} | {w:>10.4f} | {cost(w):>10.4f} | {grad(w):>10.4f}")
print("-" * 44)
print(f"Minimum reached at w = {path[-1]:.4f}, J(w) = {cost(path[-1]):.4f}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
w_axis = np.linspace(-6, 12, 200)
ax[0].plot(w_axis, cost(w_axis), color="gray")
ax[0].plot(path, cost(path), "o-", color="crimson", label="descent path")
ax[0].set_title("Path taken on the cost curve")
ax[0].set_xlabel("w"); ax[0].set_ylabel("J(w)"); ax[0].legend()

for rate in [0.05, 0.2, 0.45]:
    p = gradient_descent(w_start, rate, 30)
    ax[1].plot(cost(p), label=f"learning rate = {rate}")
ax[1].set_title("Cost vs iterations for different learning rates")
ax[1].set_xlabel("iteration"); ax[1].set_ylabel("J(w)"); ax[1].legend()
plt.tight_layout()
plt.show()
