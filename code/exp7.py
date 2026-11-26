import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.cvtColor(cv2.imread("sample.jpg"), cv2.COLOR_BGR2RGB)
h, w = img.shape[:2]

src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
dst = np.float32([[60, 40], [w - 40, 0], [w - 110, h - 20], [20, h - 80]])

H = cv2.getPerspectiveTransform(src, dst)
warped = cv2.warpPerspective(img, H, (w, h))

print("Homography matrix H (3x3):")
print(np.round(H, 4))

p = H @ np.array([w - 1, h - 1, 1.0])
print("Corner (w-1, h-1) maps to:", np.round(p[:2] / p[2], 2))

restored = cv2.warpPerspective(warped, np.linalg.inv(H), (w, h))

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
axes[0].imshow(img)
axes[0].scatter(src[:, 0], src[:, 1], c="red", s=40)
axes[0].set_title("Original + source points")
axes[1].imshow(warped)
axes[1].scatter(dst[:, 0], dst[:, 1], c="lime", s=40)
axes[1].set_title("Perspective transformed")
axes[2].imshow(restored)
axes[2].set_title("Restored using inverse H")
for a in axes:
    a.axis("off")
plt.tight_layout()
plt.show()
