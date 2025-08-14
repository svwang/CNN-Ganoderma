import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore

def build_model():
    model = models.Sequential([
        # layers.Input(shape=(640, 640, 3)),
        layers.Input(shape=(256, 256, 3)),

        layers.Conv2D(32, (3, 3), activation='relu'), # deteksi pola, tepi, dan warna
        layers.MaxPooling2D((2, 2)), # memperkecil fitur yang dihasilkan convolusi 2D

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(), # mengubah array 2D menjadi 1D agar bisa di baca oleh dense layer
        layers.Dense(64, activation='relu'), # membuat prediksi
        layers.Dense(2, activation='softmax')  # Output 2 kelas: Ganoderma, Sehat
    ])

    model.compile(optimizer='adam', # algoritma untuk update bobot model.
                  loss='categorical_crossentropy', # softmax dan label one hot
                  metrics=['accuracy'])
    
    return model
