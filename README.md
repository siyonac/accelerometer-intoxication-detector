# Accelerometer Intoxication Detector

Machine learning project that uses smartphone accelerometer data to classify 10-second windows of motion as **sober** or **intoxicated**.

## Overview

This project uses accelerometer data collected from smartphones along with transdermal alcohol concentration (TAC) measurements to build a binary classification model.

The pipeline:

1. Matches accelerometer readings with TAC measurements.
2. Uses a TAC threshold of 0.08 to create sober/intoxicated labels.
3. Splits accelerometer data into 10-second windows.
4. Extracts statistical features from the x, y, and z accelerometer axes.
5. Trains Decision Tree and Random Forest classifiers.
6. Evaluates model performance on held-out data.

## Features

For each 10-second window, the following features are extracted:

- Mean, standard deviation, minimum, and maximum of x-axis acceleration
- Mean, standard deviation, minimum, and maximum of y-axis acceleration
- Mean, standard deviation, minimum, and maximum of z-axis acceleration

This results in **12 accelerometer features** per window.

## Results

### Standard train/test split

| Model | Accuracy |
|---|---:|
| Decision Tree | 77.8% |
| Random Forest | 80.5% |

The Random Forest achieved:

- Sober recall: 91%
- Intoxicated recall: 48%
- Intoxicated F1-score: 0.54

### Unseen-participant evaluation

A separate evaluation tested the model on participants that were not included in training.

- Random Forest accuracy: 60.1%

This lower performance suggests that the model does not generalize as well to completely new participants. This is an important limitation of the current approach.

## Dataset

The project uses the UCI Bar Crawl: Detecting Heavy Drinking dataset.

The dataset contains smartphone accelerometer measurements from 13 participants**, along with TAC measurements used to determine intoxication labels.

The accelerometer data was collected at approximately **40 Hz**.

## Project Structure

```text
accelerometer-intoxication-detector/
│
├── data/
│   ├── clean_tac/
│   ├── raw/
│   └── processed_windows.csv
│
├── process_data.py
├── train_model.py
├── evaluate_model.py
├── model.pkl
├── requirements.txt
├── .gitignore
└── README.md