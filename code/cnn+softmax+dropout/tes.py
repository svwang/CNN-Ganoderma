import os
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

# ====== Konfigurasi ======
IMG_SIZE = (256, 256)
BATCH_SIZE = 16
EPOCHS = 50
DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../dataset_classification")

# ====== Data Augmentasi & Preprocessing ======
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True
)

valid_test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, 'train'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

valid_data = valid_test_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, 'valid'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_data = valid_test_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, 'test'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

print("Kelas yang terdeteksi:", list(train_data.class_indices.keys()))

# ====== Model CNN + Dropout ======
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(2, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ====== Callback ======
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = callbacks.ModelCheckpoint("best_model.keras", save_best_only=True)

# ====== Training ======
history = model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=valid_data,
    callbacks=[early_stop, checkpoint]
)

# ====== Evaluasi ======
loss, acc = model.evaluate(test_data)
print(f"\nTest Accuracy: {acc:.4f} | Test Loss: {loss:.4f}")

# ====== Save final model (optional) ======
model.save("mode_test.keras")
