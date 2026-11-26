import cv2
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(11)

P3 = np.column_stack([rng.uniform(-1, 1, 40),
                      rng.uniform(-1, 1, 40),
                      rng.uniform(4, 8, 40)])
K = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], np.float64)

R2, _ = cv2.Rodrigues(np.array([0.0, 0.12, 0.03]))
t2 = np.array([[-1.0], [0.05], [0.1]])

p1, _ = cv2.projectPoints(P3, np.zeros(3), np.zeros(3), K, None)
p2, _ = cv2.projectPoints(P3, R2, t2, K, None)
p1 = p1.reshape(-1, 2) + rng.normal(0, 0.3, (40, 2))
p2 = p2.reshape(-1, 2) + rng.normal(0, 0.3, (40, 2))

F, mask = cv2.findFundamentalMat(p1, p2, cv2.FM_RANSAC, 1.0, 0.99)
F = F / F[2, 2]
print("Fundamental matrix F:")
np.set_printoptions(precision=6, suppress=True)
print(F)
print("\nRank of F          :", np.linalg.matrix_rank(F, tol=1e-6))
print("det(F)             :", f"{np.linalg.det(F):.2e}")
print("Inliers used       :", int(mask.sum()), "/", len(p1))

h1 = np.column_stack([p1, np.ones(len(p1))])
h2 = np.column_stack([p2, np.ones(len(p2))])
res = np.abs(np.sum(h2 * (h1 @ F.T), axis=1))
print("Mean |x2' F x1|    :", f"{res.mean():.4f}")

pts = p1[:6].reshape(-1, 1, 2)
lines = cv2.computeCorrespondEpilines(pts, 1, F).reshape(-1, 3)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
cols = plt.cm.tab10(np.arange(6))
ax[0].scatter(p1[:, 0], p1[:, 1], s=12, c="gray")
ax[0].scatter(p1[:6, 0], p1[:6, 1], c=cols, s=50)
ax[0].set_xlim(0, 640); ax[0].set_ylim(480, 0)
ax[0].set_title("Image 1 : chosen points")
ax[1].scatter(p2[:, 0], p2[:, 1], s=12, c="gray")
xs = np.array([0, 640])
for (a, b, c), colr, q in zip(lines, cols, p2[:6]):
    ax[1].plot(xs, -(a * xs + c) / b, color=colr)
    ax[1].scatter(*q, color=colr, s=50, edgecolor="k")
ax[1].set_xlim(0, 640); ax[1].set_ylim(480, 0)
ax[1].set_title("Image 2 : epipolar lines")
plt.tight_layout()
plt.show()
