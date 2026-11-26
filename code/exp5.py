import cv2
import numpy as np
import matplotlib.pyplot as plt

img_bgr = cv2.imread("sample.jpg")
if img_bgr is None:
    raise FileNotFoundError("sample.jpg not found")
img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
h, w = img.shape[:2]
print("Image shape  :", img.shape, "(height, width, channels)")
print("Data type    :", img.dtype)
print("Pixel [100,100] (R,G,B):", img[100, 100])

gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
resized = cv2.resize(img, (200, 200))
M       = cv2.getRotationMatrix2D((w // 2, h // 2), 45, 1.0)
rotated = cv2.warpAffine(img, M, (w, h))
flipped = cv2.flip(img, 1)
cropped = img[50:250, 100:350]
blurred = cv2.GaussianBlur(img, (15, 15), 0)

print("Gray shape   :", gray.shape)
print("Resized shape:", resized.shape)
print("Cropped shape:", cropped.shape)

items = [(img, "Original"), (gray, "Grayscale"),
         (resized, "Resized 200x200"),
         (rotated, "Rotated 45 deg"), (flipped, "Horizontal flip"),
         (cropped, "Cropped region"), (blurred, "Gaussian blur")]
fig, axes = plt.subplots(2, 4, figsize=(13, 6))
for ax, (im, title) in zip(axes.ravel(), items):
    ax.imshow(im, cmap="gray" if im.ndim == 2 else None)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
axes.ravel()[-1].axis("off")
plt.tight_layout()
plt.show()
