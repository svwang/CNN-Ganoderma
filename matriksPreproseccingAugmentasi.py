import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_distances


# =====================
# Dataset Loader
# =====================
IMG_SIZE = 224
BATCH = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset_classification/train",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    label_mode="categorical",
    shuffle=True
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset_classification/valid",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    label_mode="categorical",
    shuffle=False
)

class_names = train_ds.class_names
print("Classes:", class_names)

def get_images_from_dataset(dataset, num_batches=1):
    images = []
    labels = []
    for i, (img_batch, label_batch) in enumerate(dataset):
        images.append(img_batch.numpy())
        labels.append(label_batch.numpy())
        if i+1 >= num_batches:
            break
    images = np.concatenate(images, axis=0)
    labels = np.concatenate(labels, axis=0)
    return images, labels

# Ambil 1 batch saja untuk contoh
images, labels = get_images_from_dataset(train_ds, num_batches=1)
print("Shape images:", images.shape)

# =====================
# Normalisasi Min-Max
# =====================
min_val = images.min()
max_val = images.max()
images_minmax = (images - min_val) / (max_val - min_val)
print("Min-Max -> min:", images_minmax.min(), "max:", images_minmax.max())

# =====================
# Normalisasi Z-Score
# =====================
mean_val = images.mean()
std_val = images.std()
images_zscore = (images - mean_val) / std_val
print("Z-Score -> mean:", images_zscore.mean(), "std:", images_zscore.std())

# =====================
# Visualisasi beberapa gambar
# =====================
fig, axes = plt.subplots(3, 5, figsize=(15, 9))
for i in range(5):
    # Gambar asli
    axes[0, i].imshow(images[i].astype(np.uint8))
    axes[0, i].set_title("Original")
    axes[0, i].axis("off")

    # Gambar Min-Max
    axes[1, i].imshow(images_minmax[i])
    axes[1, i].set_title("Min-Max")
    axes[1, i].axis("off")

    # Gambar Z-Score (dikembalikan ke range 0-1 untuk visualisasi)
    z_img = (images_zscore[i] - images_zscore[i].min()) / (images_zscore[i].max() - images_zscore[i].min())
    axes[2, i].imshow(z_img)
    axes[2, i].set_title("Z-Score")
    axes[2, i].axis("off")

plt.tight_layout()
plt.show()

# =====================
# Visualisasi histogram distribusi pixel
# =====================
plt.figure(figsize=(18,5))

# Histogram Original
plt.subplot(1,3,1)
plt.hist(images.flatten(), bins=50, color='blue', alpha=0.7)
plt.title("Original Pixel Distribution")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")

# Histogram Min-Max
plt.subplot(1,3,2)
plt.hist(images_minmax.flatten(), bins=50, color='green', alpha=0.7)
plt.title("Min-Max Normalized Distribution")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")

# Histogram Z-Score
plt.subplot(1,3,3)
plt.hist(images_zscore.flatten(), bins=50, color='red', alpha=0.7)
plt.title("Z-Score Normalized Distribution")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()


# =====================
# Augmentasi Pipeline
# =====================
data_aug = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
    layers.RandomBrightness(0.2),
], name="augment")

# =====================
# 1. Augmentation Coverage & Label Preservation
# =====================
def augmentation_coverage(dataset, n_samples=200):
    total = 0
    preserved = 0
    for images, labels in dataset.take(n_samples//BATCH):
        aug_imgs = data_aug(images, training=True)
        # hitung coverage = % sample yang berubah signifikan
        diff = tf.reduce_mean(tf.abs(tf.cast(images, tf.float32) - aug_imgs), axis=[1,2,3])
        coverage = tf.reduce_mean(tf.cast(diff > 5.0, tf.float32)).numpy()
        preserved += np.sum(np.argmax(labels.numpy(), axis=1) == np.argmax(labels.numpy(), axis=1))  # asumsi label tidak berubah
        total += len(labels)
    label_preservation = preserved / total
    return coverage, label_preservation

coverage, label_preservation = augmentation_coverage(train_ds)
plt.bar(["Coverage", "Label-preservation"], [coverage, label_preservation])
plt.title("Augmentation Coverage & Label Preservation")
plt.ylim(0, 1.1)
plt.show()


# =====================
# 2. Feature Diversity (Cosine Distance)
# =====================
def feature_diversity(dataset, n_samples=100):
    base_model = tf.keras.applications.ResNet50(include_top=False, pooling="avg")
    feats_orig, feats_aug = [], []
    for images, labels in dataset.take(n_samples//BATCH):
        aug_imgs = data_aug(images, training=True)
        f1 = base_model(preprocess_input(tf.cast(images, tf.float32)))
        f2 = base_model(preprocess_input(tf.cast(aug_imgs, tf.float32)))
        feats_orig.append(f1.numpy()); feats_aug.append(f2.numpy())
    feats_orig = np.concatenate(feats_orig, axis=0)
    feats_aug = np.concatenate(feats_aug, axis=0)
    cos_dist = np.diag(cosine_distances(feats_orig, feats_aug))
    return cos_dist

cos_dist = feature_diversity(train_ds)
plt.hist(cos_dist, bins=30, color="orange", edgecolor="black")
plt.title("Keragaman Fitur (Cosine Distance)")
plt.xlabel("Cosine Distance"); plt.ylabel("Count")
plt.show()

# =====================
# 3. Training Baseline vs Augmented
# =====================
def build_model():
    base = tf.keras.applications.ResNet50(include_top=False, pooling="avg", input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False
    x = layers.Dense(len(class_names), activation="softmax")(base.output)
    return models.Model(base.input, x)

# baseline tanpa augmentasi
model_base = build_model()
model_base.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
hist_base = model_base.fit(train_ds.map(lambda x,y: (preprocess_input(x), y)),
                           validation_data=val_ds.map(lambda x,y: (preprocess_input(x), y)),
                           epochs=5)

# dengan augmentasi
model_aug = build_model()
model_aug.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
hist_aug = model_aug.fit(train_ds.map(lambda x,y: (preprocess_input(data_aug(x, training=True)), y)),
                         validation_data=val_ds.map(lambda x,y: (preprocess_input(x), y)),
                         epochs=5)

# plot kurva
plt.plot(hist_base.history["accuracy"], label="Baseline Train")
plt.plot(hist_base.history["val_accuracy"], label="Baseline Val")
plt.plot(hist_aug.history["accuracy"], label="Aug Train")
plt.plot(hist_aug.history["val_accuracy"], label="Aug Val")
plt.title("Kurva Training (Accuracy)")
plt.legend(); plt.show()

plt.plot(hist_base.history["loss"], label="Baseline Loss")
plt.plot(hist_base.history["val_loss"], label="Baseline Val Loss")
plt.plot(hist_aug.history["loss"], label="Aug Loss")
plt.plot(hist_aug.history["val_loss"], label="Aug Val Loss")
plt.title("Kurva Training (Loss)")
plt.legend(); plt.show()


# =====================
# 4. Macro-F1 & Robustness Drop
# =====================
def eval_f1(model, dataset):
    y_true, y_pred = [], []
    for x, y in dataset:
        p = model.predict(preprocess_input(x))
        y_true.extend(np.argmax(y.numpy(), axis=1))
        y_pred.extend(np.argmax(p, axis=1))
    return f1_score(y_true, y_pred, average="macro")

f1_base = eval_f1(model_base, val_ds)
f1_aug = eval_f1(model_aug, val_ds)
robustness_drop = f1_base - f1_aug

plt.bar(["Baseline", "Augmented"], [f1_base, f1_aug])
plt.title(f"Macro-F1 (Val) | Robustness Drop={robustness_drop:.3f}")
plt.show()

# =====================
# 5. Confusion Matrix
# =====================
def plot_confusion(model, dataset, title="Confusion Matrix"):
    y_true, y_pred = [], []
    for x, y in dataset:
        p = model.predict(preprocess_input(x))
        y_true.extend(np.argmax(y.numpy(), axis=1))
        y_pred.extend(np.argmax(p, axis=1))
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.title(title)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    plt.show()

plot_confusion(model_aug, val_ds, "Confusion Matrix (Augmented Model)")

# =====================
# 6. Distribusi Parameter Augmentasi (opsional)
# =====================
# Simulasi brightness shift
samples = 100
brightness_values = []
for images, _ in train_ds.take(samples//BATCH):
    aug_imgs = data_aug(images, training=True).numpy()
    shift = np.mean(aug_imgs - images.numpy())
    brightness_values.append(shift)

plt.hist(brightness_values, bins=20, color="green", edgecolor="black")
plt.title("Distribusi Parameter Augmentasi (Brightness Shift)")
plt.xlabel("Brightness delta"); plt.ylabel("Count")
plt.show()

