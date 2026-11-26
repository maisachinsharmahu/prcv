import cv2
import numpy as np
import matplotlib.pyplot as plt

cols, rows, sq = 9, 6, 40
board = np.full(((rows + 1) * sq, (cols + 1) * sq), 255, np.uint8)
for r in range(rows + 1):
    for c in range(cols + 1):
        if (r + c) % 2 == 0:
            board[r*sq:(r+1)*sq, c*sq:(c+1)*sq] = 0
board = cv2.copyMakeBorder(board, 60, 60, 60, 60,
                           cv2.BORDER_CONSTANT, value=255)

W, H = 640, 480
K_true = np.array([[700, 0, 320], [0, 700, 240], [0, 0, 1]], np.float64)
rng = np.random.default_rng(3)

obj_pts, img_pts, views = [], [], []
objp = np.zeros((rows * cols, 3), np.float32)
objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * sq
crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

for i in range(12):
    rvec = rng.uniform(-0.45, 0.45, 3)
    tvec = np.array([-180 + rng.uniform(-40, 40),
                     -120 + rng.uniform(-30, 30),
                     900 + rng.uniform(-100, 150)])
    Rm, _ = cv2.Rodrigues(rvec)
    s0 = 60 + sq
    off = np.array([[1, 0, -s0], [0, 1, -s0], [0, 0, 1]], np.float64)
    Hm = K_true @ np.column_stack([Rm[:, 0], Rm[:, 1], tvec]) @ off
    view = cv2.warpPerspective(board, Hm, (W, H), borderValue=200)
    ok, corners = cv2.findChessboardCorners(view, (cols, rows))
    if ok:
        corners = cv2.cornerSubPix(view, corners, (11, 11), (-1, -1), crit)
        obj_pts.append(objp); img_pts.append(corners); views.append(view)

print(f"Chessboard detected in {len(views)} / 12 images")

rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    obj_pts, img_pts, (W, H), None, None)
print("\nRMS reprojection error :", round(rms, 4), "pixels")
print("\nEstimated camera matrix K:")
print(np.round(K, 2))
print("\nTrue camera matrix used for rendering:")
print(K_true)
print("\nDistortion coefficients:", np.round(dist.ravel(), 4))

errs = []
for o, p, r, t in zip(obj_pts, img_pts, rvecs, tvecs):
    proj, _ = cv2.projectPoints(o, r, t, K, dist)
    d = cv2.norm(p.reshape(-1, 2), proj.reshape(-1, 2), cv2.NORM_L2)
    errs.append(d / len(proj))
print("Mean error per image   :", np.round(errs, 3))

vis = cv2.cvtColor(views[0], cv2.COLOR_GRAY2RGB)
cv2.drawChessboardCorners(vis, (cols, rows), img_pts[0], True)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].imshow(vis)
ax[0].set_title("Detected corners (view 1)")
ax[0].axis("off")
ax[1].bar(range(1, len(errs) + 1), errs, color="teal")
ax[1].set_title("Reprojection error per image")
ax[1].set_xlabel("image")
ax[1].set_ylabel("pixels")
plt.tight_layout()
plt.show()
