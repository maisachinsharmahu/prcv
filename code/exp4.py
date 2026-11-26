import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

tf.random.set_seed(1)
np.random.seed(1)

(x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.mnist.load_data()
x_tr = x_tr.astype("float32") / 255.0
x_te = x_te.astype("float32") / 255.0
print("Train:", x_tr.shape, " Test:", x_te.shape)

t0 = time.time()
svm = SVC(kernel="rbf", gamma="scale", C=5)
svm.fit(x_tr[:10000].reshape(10000, -1), y_tr[:10000])
svm_acc = accuracy_score(y_te, svm.predict(x_te.reshape(len(x_te), -1)))
svm_time = time.time() - t0
print(f"SVM  accuracy = {svm_acc*100:.2f}%   (time {svm_time:.0f}s)")

cnn = tf.keras.Sequential([
    tf.keras.layers.Input((28, 28, 1)),
    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(10, activation="softmax"),
])
cnn.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
t0 = time.time()
cnn.fit(x_tr[..., None], y_tr, epochs=3, batch_size=128,
        validation_split=0.1, verbose=2)
_, cnn_acc = cnn.evaluate(x_te[..., None], y_te, verbose=0)
cnn_time = time.time() - t0
print(f"CNN  accuracy = {cnn_acc*100:.2f}%   (time {cnn_time:.0f}s)")

print("\nSummary")
print(f"  SVM : {svm_acc*100:.2f}%")
print(f"  CNN : {cnn_acc*100:.2f}%")

plt.figure(figsize=(5, 4))
bars = plt.bar(["SVM", "CNN"], [svm_acc * 100, cnn_acc * 100],
               color=["#4c72b0", "#dd8452"])
for b in bars:
    plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.1,
             f"{b.get_height():.2f}%", ha="center")
plt.ylim(90, 100.5)
plt.ylabel("Test accuracy (%)")
plt.title("SVM vs CNN on MNIST")
plt.show()
