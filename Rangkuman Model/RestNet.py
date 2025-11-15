import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as rn_preprocess
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_score, recall_score, f1_score, accuracy_score
)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ======================
# Konfigurasi
# ======================
IMG_SIZE = (224, 224)
BATCH = 32
EPOCHS = 30
ANNOTATION_JSON = "augmented_output/annotations.json"
DATASET_DIR = "dataset_classification"
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
# Load data dari annotations.json
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

train_ds = train_ds.map(lambda x, y: (tf.cast(x, tf.float32), y))
val_ds = val_ds.map(lambda x, y: (tf.cast(x, tf.float32), y))
test_ds = test_ds.map(lambda x, y: (tf.cast(x, tf.float32), y))

# Gabungkan data
if aug_ds is not None:
    train_ds = train_ds.concatenate(aug_ds).shuffle(1000)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# ======================
# Augmentasi
# ======================
data_aug = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
], name="augment")

# ======================
# Model ResNet50
# ======================
base = ResNet50(weights="imagenet", include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
base.trainable = False

inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
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

# ======================
# Callbacks
# ======================
cb = [
    tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7),
    tf.keras.callbacks.ModelCheckpoint("best_resnet.keras", monitor="val_accuracy", save_best_only=True)
]

# ======================
# Training
# ======================
history_a = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=cb)

# Fine-tuning ResNet50
base.trainable = True
for layer in base.layers:
    if 'conv5' in layer.name or 'conv4' in layer.name:
        layer.trainable = True
    else:
        layer.trainable = False

model.compile(optimizer=Adam(5e-5), loss="categorical_crossentropy", metrics=["accuracy"])
history_b = model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=cb)

# ======================
# Plot Training Curves
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
# Evaluasi
# ======================
y_true_onehot = np.concatenate([y.numpy() for _, y in test_ds], axis=0)
y_true = np.argmax(y_true_onehot, axis=1)
y_prob = model.predict(test_ds)
y_pred = np.argmax(y_prob, axis=1)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,5))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.xticks(range(NUM_CLASSES), class_names, rotation=45)
plt.yticks(range(NUM_CLASSES), class_names)
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        plt.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else "black")
plt.colorbar()
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

# Metrik Tambahan
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average="weighted")
recall = recall_score(y_true, y_pred, average="weighted")
f1 = f1_score(y_true, y_pred, average="weighted")

print(f"\nAccuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# ROC-AUC
if NUM_CLASSES == 2:
    fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0,1], [0,1], '--')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()
else:
    from sklearn.preprocessing import label_binarize
    y_true_bin = label_binarize(y_true, classes=range(NUM_CLASSES))
    aucs = []
    for i in range(NUM_CLASSES):
        try:
            aucs.append(auc(*roc_curve(y_true_bin[:, i], y_prob[:, i])[:2]))
        except:
            aucs.append(np.nan)
    plt.bar(class_names, aucs)
    plt.title("AUC per Class")
    plt.ylim(0, 1)
    plt.show()
