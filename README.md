# AI-Based Aircraft Engine Health Monitoring System

## Overview

This project uses Long Short-Term Memory (LSTM) deep learning models to predict the Remaining Useful Life (RUL) of aircraft engines using sensor data from the NASA C-MAPSS dataset. The system helps support predictive maintenance by estimating the number of operational cycles remaining before engine failure.

## Features

* Aircraft Engine Health Monitoring
* Remaining Useful Life (RUL) Prediction
* Predictive Maintenance Support
* Sensor Data Analysis
* LSTM-Based Deep Learning Model

## Technologies Used

* Python
* TensorFlow
* Keras
* Pandas
* NumPy
* Matplotlib
* Scikit-learn

## Dataset

NASA C-MAPSS Turbofan Engine Degradation Simulation Dataset

Note: The dataset files are not included in this repository due to GitHub storage limitations. Users can download the dataset separately and place it in a local `data/` directory.

## Project Workflow

1. Data Collection
2. Data Preprocessing
3. Feature Engineering
4. LSTM Model Development
5. Model Training
6. Remaining Useful Life Prediction
7. Model Evaluation

## Results

### Model Performance

* Training Loss: 1617.67
* Validation Loss: 2712.76
* Training MAE: 29.42
* Validation MAE: 38.00

### Training Output

The LSTM model was successfully trained on the NASA C-MAPSS FD001 dataset and learned degradation patterns from aircraft engine sensor data.

### Generated Artifacts

* Trained Model: `engine_model.h5`
* Loss Curve: `loss_curve.png`

## Results

### Training Loss Curve

![Training Loss](results/loss_curve.png)

### Model Performance

| Metric          | Value   |
| --------------- | ------- |
| Training Loss   | 1617.67 |
| Validation Loss | 2712.76 |
| Training MAE    | 29.42   |
| Validation MAE  | 38.00   |

The LSTM model was trained on the NASA C-MAPSS FD001 dataset to predict Remaining Useful Life (RUL) of aircraft engines.


## Applications

* Aircraft Predictive Maintenance
* Engine Health Monitoring
* Failure Prediction
* Reliability Engineering

## Project Structure

Aircraft-Engine-Health-Monitoring/
│
├── src/
├── results/
├── README.md
├── requirements.txt
└── .gitignore




## Author

Panjala Shambhavi
