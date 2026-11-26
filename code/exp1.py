import numpy as np
import matplotlib.pyplot as plt

def gaussian_pdf(x, mu, sigma2):
    coeff = 1.0 / np.sqrt(2.0 * np.pi * sigma2)
    return coeff * np.exp(-((x - mu) ** 2) / (2.0 * sigma2))

x = np.linspace(-6, 6, 600)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))

for mu in [-2, 0, 2]:
    ax[0].plot(x, gaussian_pdf(x, mu, 1.0), label=f"mean = {mu}, var = 1")
ax[0].set_title("Effect of changing the mean")

for var in [0.3, 1.0, 3.0]:
    ax[1].plot(x, gaussian_pdf(x, 0, var), label=f"mean = 0, var = {var}")
ax[1].set_title("Effect of changing the variance")

for a in ax:
    a.set_xlabel("x")
    a.set_ylabel("N(x | mean, var)")
    a.grid(alpha=0.3)
    a.legend()
plt.tight_layout()
plt.show()

print("Peak height for different variances (mean = 0):")
for var in [0.3, 1.0, 3.0]:
    peak = gaussian_pdf(0, 0, var)
    print(f"  variance = {var:<4} -> N(0 | 0, {var}) = {peak:.4f}")
dx = x[1] - x[0]
area = np.sum(gaussian_pdf(x, 0, 1.0)) * dx
print(f"Area under N(0,1) curve = {area:.4f}")
