import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.cvtColor(cv2.imread("sample.jpg"), cv2.COLOR_BGR2RGB)
h, w = img.shape[:2]

scaled = cv2.resize(img, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)

T = np.float32([[1, 0, 80],
                [0, 1, 40]])
translated = cv2.warpAffine(img, T, (w, h))

R = cv2.getRotationMatrix2D((w / 2, h / 2), -30, 1.0)
rotated = cv2.warpAffine(img, R, (w, h))

S = np.float32([[1, 0.3, 0],
                [0, 1,   0]])
sheared = cv2.warpAffine(img, S, (int(w + 0.3 * h), h))

src = np.float32([[0, 0], [w - 1, 0], [0, h - 1]])
dst = np.float32([[30, 40], [w - 60, 10], [70, h - 50]])
A = cv2.getAffineTransform(src, dst)
affine = cv2.warpAffine(img, A, (w, h))

print("Translation matrix:\n", T)
print("Rotation matrix (-30 deg):\n", np.round(R, 3))
print("Affine matrix from 3 point pairs:\n", np.round(A, 3))
print("Scaled size :", scaled.shape[:2])
print("Sheared size:", sheared.shape[:2])

items = [(img, "Original"), (scaled, "Scaled (0.6x)"),
         (translated, "Translated (+80, +40)"),
         (rotated, "Rotated (-30 deg)"), (sheared, "Sheared (x, 0.3)"),
         (affine, "General affine")]
fig, axes = plt.subplots(2, 3, figsize=(12, 7))
for ax, (im, t) in zip(axes.ravel(), items):
    ax.imshow(im); ax.set_title(t, fontsize=10); ax.axis("off")
plt.tight_layout()
plt.show()
