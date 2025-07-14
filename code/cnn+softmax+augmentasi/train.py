from preprocessing import train_dataset, val_dataset
from cnn_model import build_model_aug

model = build_model_aug()

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10
)

model.save("model_ganoderma_aug2.h5")