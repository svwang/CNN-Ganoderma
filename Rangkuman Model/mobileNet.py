# ======================
# Imports
# ======================
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mb_preprocess
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_score, recall_score, f1_score, accuracy_score
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings("ignore")

# ======================
# Konfigurasi
# ======================
IMG_SIZE = (224, 224)
BATCH = 32
EPOCHS = 30
ANNOTATION_JSON = "/content/drive/MyDrive/augmented_output/annotations_fixed.json"
DATASET_DIR = "/content/drive/MyDrive/dataset_classification"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "valid")
TEST_DIR = os.path.join(DATASET_DIR, "test")

# ======================
# Load class names
# ======================
class_names = sorted([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))])
NUM_CLASSES = len(class_names)
class_to_idx = {c: i for i, c in enumerate(class_names)}
print("Detected classes:", class_names)

# ======================
# Load augmented data dari JSON
# ======================
image_paths, labels_idx = [], []
if os.path.exists(ANNOTATION_JSON):
    with open(ANNOTATION_JSON, "r") as f:
        ann = json.load(f)
    for item in ann:
        cls = item.get("class")
        if cls not in class_to_idx:
            continue
        idx = class_to_idx[cls]
        orig, aug = item.get("original"), item.get("augmented")
        if orig and os.path.exists(orig):
            image_paths.append(orig)
            labels_idx.append(idx)
        if aug and os.path.exists(aug):
            image_paths.append(aug)
            labels_idx.append(idx)
print(f"Total augmented images loaded: {len(image_paths)}")

def preprocess_path_label(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)
    return img, tf.one_hot(label, NUM_CLASSES)

if len(image_paths) > 0:
    aug_ds = tf.data.Dataset.from_tensor_slices((image_paths, labels_idx))
    aug_ds = aug_ds.shuffle(len(image_paths))
    aug_ds = aug_ds.map(preprocess_path_label, num_parallel_calls=tf.data.AUTOTUNE)
    aug_ds = aug_ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)
else:
    aug_ds = None

# ======================
# Load dataset original
# ======================
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR, image_size=IMG_SIZE, batch_size=BATCH,
    label_mode="categorical", shuffle=True
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR, image_size=IMG_SIZE, batch_size=BATCH,
    label_mode="categorical", shuffle=False
)
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR, image_size=IMG_SIZE, batch_size=BATCH,
    label_mode="categorical", shuffle=False
)

# Cast ke float32
train_ds = train_ds.map(lambda x, y: (tf.cast(x, tf.float32), y))
val_ds = val_ds.map(lambda x, y: (tf.cast(x, tf.float32), y))
test_ds = test_ds.map(lambda x, y: (tf.cast(x, tf.float32), y))

# Gabungkan augmented data
if aug_ds is not None:
    train_ds = train_ds.concatenate(aug_ds).shuffle(1000)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# ======================
# Data Augmentasi
# ======================
data_aug = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
], name="augment")

# ======================
# Model MobileNetV2
# ======================
base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
base.trainable = False

inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = data_aug(inputs)
x = mb_preprocess(x)               # preprocessing MobileNetV2
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
model = Model(inputs, outputs)

model.compile(optimizer=Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# ======================
# Callbacks
# ======================
cb = [
    tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7),
    tf.keras.callbacks.ModelCheckpoint("best_mobilenet.keras", monitor="val_accuracy", save_best_only=True)
]

# ======================
# Training
# ======================
history_a = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=cb)

# Fine-tuning MobileNetV2 (unfreeze sebagian akhir)
base.trainable = True
# keep most layers frozen, unfreeze last ~30 layers for fine-tuning
for layer in base.layers[:-30]:
    layer.trainable = False
for layer in base.layers[-30:]:
    layer.trainable = True

model.compile(optimizer=Adam(5e-5), loss="categorical_crossentropy", metrics=["accuracy"])
history_b = model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=cb)

# ======================
# Fungsi Plot History
# ======================
def plot_history(histories):
    acc, val_acc, loss, val_loss = [], [], [], []
    for h in histories:
        acc += h.history.get("accuracy", [])
        val_acc += h.history.get("val_accuracy", [])
        loss += h.history.get("loss", [])
        val_loss += h.history.get("val_loss", [])
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(acc, label="Train Acc")
    plt.plot(val_acc, label="Val Acc")
    plt.title("Accuracy")
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(loss, label="Train Loss")
    plt.plot(val_loss, label="Val Loss")
    plt.title("Loss")
    plt.legend()
    plt.show()

plot_history([history_a, history_b])

# ======================
# Evaluasi Lengkap
# ======================
y_true, y_pred, y_prob = [], [], []
start_time = time.time()

for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))
    y_prob.extend(preds)

end_time = time.time()
inference_time = (end_time - start_time) / len(y_true)

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_prob = np.array(y_prob)

# --- Metrik Global ---
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
test_loss, test_acc = model.evaluate(test_ds, verbose=0)

print("\n=== Evaluation Metrics ===")
print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"F1-Score  : {f1:.4f}")
print(f"Loss      : {test_loss:.4f}")
print(f"Rata-rata waktu inferensi per gambar: {inference_time:.4f} detik")

# ======================
# Confusion Matrix
# ======================
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(7,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ======================
# Precision / Recall / F1 per kelas
# ======================
report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
precisions = [report[c]['precision'] for c in class_names]
recalls = [report[c]['recall'] for c in class_names]
f1s = [report[c]['f1-score'] for c in class_names]

x = np.arange(len(class_names))
width = 0.25

plt.figure(figsize=(10,6))
plt.bar(x - width, precisions, width, label='Precision')
plt.bar(x, recalls, width, label='Recall')
plt.bar(x + width, f1s, width, label='F1-Score')
plt.xticks(x, class_names, rotation=45)
plt.ylabel("Score")
plt.title("Precision, Recall, dan F1-score per Kelas")
plt.legend()
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()

# ======================
# ROC Curve
# ======================
if NUM_CLASSES > 2:
    y_true_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
    fpr, tpr, roc_auc = {}, {}, {}

    for i in range(NUM_CLASSES):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(NUM_CLASSES)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(NUM_CLASSES):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= NUM_CLASSES

    plt.figure(figsize=(7,6))
    plt.plot(all_fpr, mean_tpr, lw=2, label=f'Macro-average ROC (AUC={np.mean(list(roc_auc.values())):.4f})')
    plt.plot([0,1],[0,1],'--',color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Macro-average ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()
else:  # binary
    fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
    auc_value = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc_value:.4f}")
    plt.plot([0,1],[0,1],'--')
    plt.title("ROC Curve (Binary)")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.grid(True)
    plt.show()

# ======================
# Barplot Global Metrik
# ======================
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [acc, prec, rec, f1]

plt.figure(figsize=(8,5))
sns.barplot(x=metrics, y=values)
plt.title("Perbandingan Metrik Global")
plt.ylim(0, 1)
for i, v in enumerate(values):
    plt.text(i, v + 0.02, f"{v:.3f}", ha='center')
plt.grid(True, axis='y')
plt.show()

# ======================
# Inferensi per batch
# ======================
batch_times = []
for images, labels in test_ds.take(10):
    s = time.time()
    model.predict(images, verbose=0)
    e = time.time()
    batch_times.append(e - s)

plt.figure(figsize=(8,5))
plt.plot(batch_times, marker='o')
plt.title("Distribusi Waktu Inferensi per Batch (10 batch pertama)")
plt.xlabel("Batch ke-")
plt.ylabel("Detik")
plt.grid(True)
plt.show()

# ======================
# Classification Report Lengkap
# ======================
print("\n=== Classification Report ===")
print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
