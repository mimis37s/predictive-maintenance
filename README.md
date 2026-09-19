# Predictive Maintenance - Bearing Fault Detection

A small machine learning project for detecting bearing faults using vibration data.

## Dataset

I used the Case Western Reserve University (CWRU) Bearing Data Center dataset.

The dataset contains vibration measurements from healthy and faulty bearings.

## What I did

- Split the vibration signals into 0.1 second windows.
- Calculated RMS, Peak and Kurtosis for each window.
- Created a dataset with healthy and faulty examples.
- Trained a Logistic Regression classifier.
- Used StandardScaler to scale the features.
- Tested the model on complete recordings that were not used for training.

## Results

The scaled Logistic Regression model achieved 100% accuracy on the selected test recordings (504/504 windows).

## Tools

Python, NumPy, SciPy, Pandas, Matplotlib, Scikit-learn