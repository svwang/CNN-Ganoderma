import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import time

from preprocessing import test_dataset

# Load model
model = tf.keras.models.load_model("model_ganoderma_dropout.h5")
print("Model berhasil dimuat.")

# Evaluasi
test_loss, test_acc = model.evaluate(test_dataset)
print(f"\nTest Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Confusion matrix dan laporan klasifikasi
y_true = []
y_pred = []
class_names = test_dataset.class_names

start_time = time.time()

for images, labels in test_dataset:
    preds = model.predict(images)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

end_time = time.time()
elapsed_time = end_time - start_time
inference_time = elapsed_time / len(y_true)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Classification report
report = classification_report(y_true, y_pred, target_names=class_names)
print("\nClassification Report:\n")
print(report)

# Waktu inferensi
print(f"Rata-rata waktu inferensi per gambar: {inference_time:.4f} detik")
