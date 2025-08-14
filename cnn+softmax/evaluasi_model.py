import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import seaborn as sns
import time
import os

from preprocessing import test_dataset

# Load model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# DATASET_DIR = os.path.join(BASE_DIR, "model/model_ganoderma1.h5")
DATASET_DIR = os.path.join(BASE_DIR, "model/model_ganoderma2.h5")
model = tf.keras.models.load_model(DATASET_DIR)
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

# Skor evaluasi
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average='binary')
rec = recall_score(y_true, y_pred, average='binary')
f1 = f1_score(y_true, y_pred, average='binary')

# Hitung AUC (untuk binary classification)
# Jika model output softmax 2 kelas: ambil probabilitas kelas 1
y_prob = []
for images, _ in test_dataset:
    preds = model.predict(images)
    y_prob.extend(preds[:, 1])  # ambil probabilitas kelas positif (kelas ke-1)

auc = roc_auc_score(y_true, y_prob)

print("\n=== Evaluation Metrics ===")
print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"F1-Score  : {f1:.4f}")
print(f"AUC       : {auc:.4f}")


# Waktu inferensi
print(f"Rata-rata waktu inferensi per gambar: {inference_time:.4f} detik")
