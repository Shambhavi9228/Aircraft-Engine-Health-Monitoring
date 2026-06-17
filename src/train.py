import numpy as np
from model import build_model

X_train = np.random.rand(100, 50, 10)
y_train = np.random.rand(100)

model = build_model((50, 10))

history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=16,
    validation_split=0.2
)

model.save("aircraft_engine_model.h5")

print("Model Training Completed")
