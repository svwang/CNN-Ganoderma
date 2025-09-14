import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as rn_preprocess
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)
import os
import warnings
warnings.filterwarnings("ignore")

# ======================
# Config
# ======================
IMG_SIZE = 224
BATCH = 32
NUM_CLASSES = 2  # Sehat vs Ganoderma
DATA_DIR = "dataset_classification"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "valid")
TEST_DIR = os.path.join(DATA_DIR, "test")

# ======================
# Dataset Loader (paksa RGB)
# ======================
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    label_mode="categorical",
    shuffle=True,
    color_mode="rgb"
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    VAL_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    label_mode="categorical",
    shuffle=False,
    color_mode="rgb"
)
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    label_mode="categorical",
    shuffle=False,
    color_mode="rgb"
)

class_names = train_ds.class_names  # urutan label sesuai folder alfabet
print("Class order:", class_names)

# Prefetch
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# ======================
# Hitung class_weight otomatis (hindari salah mapping)
# ======================
counts = np.zeros(NUM_CLASSES, dtype=int)
for _, yb in train_ds.unbatch():
    counts[np.argmax(yb.numpy())] += 1

total = counts.sum()
class_weight = {i: total / (NUM_CLASSES * max(1, counts[i])) for i in range(NUM_CLASSES)}
print("Train counts:", dict(zip(class_names, counts.tolist())))
print("Class weight:", class_weight)

# ======================
# Augmentasi
# ======================
data_aug = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
    layers.RandomBrightness(0.2),
], name="augment")

# ======================
# Model: ResNet50 (stabil di banyak setup)
# ======================
base = ResNet50(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base.trainable = False

inputs = layers.Input((IMG_SIZE, IMG_SIZE, 3))
x = data_aug(inputs)
x = rn_preprocess(x)
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

# Callbacks
cb = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", 
        patience=12, 
        restore_best_weights=True,
        min_delta=0.001
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", 
        factor=0.5, 
        patience=6,
        min_lr=1e-7
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'best_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
]

# ======================
# Training Fase A (freeze backbone)
# ======================
history_a = model.fit(train_ds, validation_data=val_ds, epochs=30,
                      class_weight=class_weight, callbacks=cb, verbose=1)

# ======================
# Fase B: Fine-tune blok akhir
# ======================
# Fase B: Fine-tuning bertahap
base.trainable = True

# Unfreeze bertahap
for layer in base.layers:
    if 'conv5' in layer.name:  # Layer terakhir
        layer.trainable = True
    elif 'conv4' in layer.name:
        layer.trainable = True
    else:
        layer.trainable = False

model.compile(optimizer=Adam(5e-5), loss="categorical_crossentropy", metrics=["accuracy"])
history_b = model.fit(train_ds, validation_data=val_ds, epochs=15,
                      class_weight=class_weight, callbacks=cb, verbose=1)

# Fase C: Unfreeze lebih banyak
for layer in base.layers:
    layer.trainable = True

model.compile(optimizer=Adam(1e-6), loss="categorical_crossentropy", metrics=["accuracy"])
history_c = model.fit(train_ds, validation_data=val_ds, epochs=10,
                      class_weight=class_weight, callbacks=cb, verbose=1)

# ======================
# Plot: Training Curves (Accuracy & Loss)
# ======================
def plot_history(hist_a, hist_b):
    acc = (hist_a.history.get("accuracy", []) + hist_b.history.get("accuracy", []))
    val_acc = (hist_a.history.get("val_accuracy", []) + hist_b.history.get("val_accuracy", []))
    loss = (hist_a.history.get("loss", []) + hist_b.history.get("loss", []))
    val_loss = (hist_a.history.get("val_loss", []) + hist_b.history.get("val_loss", []))

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(6,4))
    plt.plot(epochs, acc, label="Train Acc")
    plt.plot(epochs, val_acc, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training & Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6,4))
    plt.plot(epochs, loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_history(history_a, history_b)

# ======================
# Evaluasi di test set
# ======================
# Ground truth
y_true_onehot = np.concatenate([y.numpy() for x, y in test_ds], axis=0)
y_true = np.argmax(y_true_onehot, axis=1)

# Prediksi
y_pred_proba = model.predict(test_ds)
y_pred = np.argmax(y_pred_proba, axis=1)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))

plt.figure(figsize=(5.5,5))
plt.imshow(cm, interpolation='nearest')
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(NUM_CLASSES)
plt.xticks(tick_marks, class_names, rotation=45, ha="right")
plt.yticks(tick_marks, class_names)
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()

# Classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

# Metrics agregat
acc = accuracy_score(y_true, y_pred)
# pilih index kelas positif = 'Ganoderma' jika ada, else ambil kelas kedua
pos_label_name = "Ganoderma"
pos_index = class_names.index(pos_label_name) if pos_label_name in class_names else 1

prec = precision_score(y_true, y_pred, pos_label=pos_index)
recall = recall_score(y_true, y_pred, pos_label=pos_index)  # Sensitivity / TPR
f1 = f1_score(y_true, y_pred, pos_label=pos_index)

# Specificity = TN / (TN + FP) untuk kelas positif = pos_index
tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[1-pos_index, pos_index]).ravel()
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

# ROC-AUC (pakai probabilitas kelas positif)
y_true_bin = (y_true == pos_index).astype(int)
fpr, tpr, _ = roc_curve(y_true_bin, y_pred_proba[:, pos_index])
roc_auc = auc(fpr, tpr)

print(f"\nAccuracy     : {acc:.4f}")
print(f"Precision    : {prec:.4f}")
print(f"Recall (TPR) : {recall:.4f}")
print(f"F1-Score     : {f1:.4f}")
print(f"Specificity  : {specificity:.4f}")
print(f"ROC-AUC      : {roc_auc:.4f}")

# ROC Curve
plt.figure(figsize=(5.5,5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title(f"ROC Curve (Positive class: {class_names[pos_index]})")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()
