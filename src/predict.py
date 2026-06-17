import numpy as np
from tensorflow.keras.models import load_model

model = load_model("aircraft_engine_model.h5")

sample_input = np.random.rand(1, 50, 10)

prediction = model.predict(sample_input)

print("Predicted Remaining Useful Life:", prediction[0][0])
