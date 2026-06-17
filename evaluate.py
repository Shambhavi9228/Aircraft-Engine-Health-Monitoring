import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

from data_preprocessing import load_train_data

print("Loading trained model...")

model = load_model("../results/engine_model.h5", compile=False)

df = load_train_data("../data/train_FD001.txt")

features = [col for col in df.columns if "sensor" in col]

X = df[features]
y_actual = df["RUL"]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_scaled = X_scaled.reshape(
    X_scaled.shape[0],
    1,
    X_scaled.shape[1]
)

print("Generating predictions...")

y_pred = model.predict(X_scaled)

plt.figure(figsize=(10, 6))

plt.plot(y_actual[:200], label="Actual RUL")
plt.plot(y_pred[:200], label="Predicted RUL")

plt.title("Actual vs Predicted RUL")
plt.xlabel("Samples")
plt.ylabel("Remaining Useful Life")

plt.legend()

plt.savefig("../results/actual_vs_predicted.png")

print("Graph saved successfully!")