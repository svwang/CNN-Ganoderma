import tensorflow as tf
import os

# IMG_SIZE = (640, 640)
IMG_SIZE = (256, 256)
BATCH_SIZE = 16
# Path gambar
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "../dataset_classification")

# Load dataset dari folder terpisah: train, valid, test
train_dataset = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_DIR, 'train'),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'  # output softmax
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_DIR, 'valid'),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_DIR, 'test'),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

# Cek nama kelas
class_names = train_dataset.class_names
print('Kelas yang terdeteksi:', class_names)

# Cek satu batch jika file dijalankan langsung
if __name__ == "__main__":
    for images, labels in train_dataset.take(1):
        print('Shape gambar:', images.shape)
        print('Shape label:', labels.shape)
