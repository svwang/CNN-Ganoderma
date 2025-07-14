from cnn_model import build_model
from preprocessing import train_dataset, val_dataset

model = build_model()

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10
)

# Simpan model
model.save("model_ganoderma.h5")
