import scipy.io
from features import extract_features
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def process_recording(file_path):
    data = scipy.io.loadmat(file_path)
    de_key = [key for key in data.keys() if key.endswith("_DE_time")][0]
    vibration = data[de_key].ravel()
    print(vibration.shape)
    fs = 12000
    window_size = 0.1
    samples_per_window = int(window_size * fs)
    rms_values = []
    peak_values = []
    kurtosis_values = []

    for start in range(0, len(vibration), samples_per_window):
        end = start + samples_per_window
        window = vibration[start:end]

        if len(window) < samples_per_window:
           continue

   
        rms, peak, kurt = extract_features(window)

        rms_values.append(rms)
        peak_values.append(peak)
        kurtosis_values.append(kurt)
      
    return rms_values, peak_values, kurtosis_values


files = [
    "data/normal_0.mat",
    "data/Normal_1.mat",
    "data/IR007_0.mat",
    "data/IR014_1.mat",
    "data/B007_0.mat",
    "data/0R007@6_0.mat"
]

rms_all = []
peak_all = []
kurtosis_all = []
labels_all = []
recordings_all = []

for file_path in files:

    rms, peak, kurtosis = process_recording(file_path)

    if "normal" in file_path.lower():
        label = 0
    else:
        label = 1
    recording = file_path.split("/")[-1].replace(".mat", "")
    recordings = [recording] * len(rms)

    labels = [label] * len(rms)
    rms_all.extend(rms)
    peak_all.extend(peak)
    kurtosis_all.extend(kurtosis)
    labels_all.extend(labels)
    recordings_all.extend(recordings)
  

print(len(rms_all))
print(len(peak_all))
print(len(kurtosis_all))
print(len(labels_all))
print(len(recordings_all))
print(recordings_all[:5])



df=pd.DataFrame({
    'RMS': rms_all,
    'Peak': peak_all,
    'Kurtosis': kurtosis_all,
    'Label': labels_all,
    'Recording': recordings_all 
})


test_recordings = ["Normal_1", "IR014_1"]
test_df = df[df["Recording"].isin(test_recordings)]
train_df = df[~df["Recording"].isin(test_recordings)]

features = ["RMS", "Peak", "Kurtosis"]

X_train = train_df[features]
y_train = train_df["Label"]

X_test = test_df[features]
y_test = test_df["Label"]

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

cm = confusion_matrix(y_test, y_pred)

print(cm)

probabilities = model.predict_proba(X_test)

print(probabilities[:10])

wrong = (y_test.values == 0) & (y_pred == 1)


print(test_df.loc[wrong, ["RMS", "Peak", "Kurtosis", "Recording"]])
print(model.coef_)
print(model.intercept_)

healthy = df[df["Label"] == 0]
faulty = df[df["Label"] == 1]

p_faulty = probabilities[:, 1]


scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(X_train_scaled[:3])

scaled_model = LogisticRegression()
scaled_model.fit(X_train_scaled, y_train)
y_pred_scaled = scaled_model.predict(X_test_scaled)

accuracy_scaled = accuracy_score(y_test, y_pred_scaled)

print("Scaled accuracy:", accuracy_scaled)
print(confusion_matrix(y_test, y_pred_scaled))
print(classification_report(
    y_test,
    y_pred_scaled,
    target_names=["Healthy", "Faulty"]
))
import matplotlib.pyplot as plt

plt.scatter(
    df[df["Label"] == 0]["RMS"],
    df[df["Label"] == 0]["Peak"],
    label="Healthy"
)

plt.scatter(
    df[df["Label"] == 1]["RMS"],
    df[df["Label"] == 1]["Peak"],
    label="Faulty"
)

plt.xlabel("RMS")
plt.ylabel("Peak")
plt.title("Bearing condition: RMS vs Peak")
plt.legend()
plt.show()