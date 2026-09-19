import scipy.io
import numpy as np
from scipy.stats import kurtosis
import  matplotlib.pyplot as plt
import pandas as pd
from features import extract_features
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

data=scipy.io.loadmat('data/normal_0.mat')
faulty_data=scipy.io.loadmat('data/IR007_0.mat')

vibration = data['X097_DE_time'].ravel()
fault_vibration = faulty_data['X105_DE_time'].ravel()

fs = 12000

signal = vibration[:12000]
fft_signal=np.fft.fft(signal)
frequencies = np.fft.fftfreq(len(signal), 1/fs)
magnitude = np.abs(fft_signal)
positive_frequencies = frequencies[frequencies >= 0]
positive_magnitude = magnitude[frequencies >= 0]

fault_signal=fault_vibration[:12000]
fault_fft_signal=np.fft.fft(fault_signal)
fault_frequencies = np.fft.fftfreq(len(fault_signal), 1/fs)
fault_magnitude = np.abs(fault_fft_signal)
fault_positive_frequencies = fault_frequencies[fault_frequencies >= 0]
fault_positive_magnitude = fault_magnitude[fault_frequencies >= 0]

bpfi_mask = (positive_frequencies >= 150) & (positive_frequencies <= 175)

bpfi_max_magn = max(positive_magnitude[bpfi_mask])
bpfi_i = np.argmax(positive_magnitude[bpfi_mask])
bpfi_max_freq = positive_frequencies[bpfi_mask][bpfi_i]

fault_bpfi_mask = (fault_positive_frequencies >= 150) & (fault_positive_frequencies <= 175)

fault_bpfi_max_magn = max(fault_positive_magnitude[fault_bpfi_mask])
fault_bpfi_i = np.argmax(fault_positive_magnitude[fault_bpfi_mask])
fault_bpfi_max_freq = fault_positive_frequencies[fault_bpfi_mask][fault_bpfi_i]

bsf_mask = (positive_frequencies >= 130) & (positive_frequencies <= 150)

bsf_max_magn = max(positive_magnitude[bsf_mask])
bsf_i = np.argmax(positive_magnitude[bsf_mask])
bsf_max_freq = positive_frequencies[bsf_mask][bsf_i]

fault_bsf_mask = (fault_positive_frequencies >= 130) & (fault_positive_frequencies <= 150)

fault_bsf_max_magn = max(fault_positive_magnitude[fault_bsf_mask])
fault_bsf_i = np.argmax(fault_positive_magnitude[fault_bsf_mask])
fault_bsf_max_freq = fault_positive_frequencies[fault_bsf_mask][fault_bsf_i]

bpfo_mask = (positive_frequencies >= 95) & (positive_frequencies <= 120)

bpfo_max_magn = max(positive_magnitude[bpfo_mask])
bpfo_i = np.argmax(positive_magnitude[bpfo_mask])
bpfo_max_freq = positive_frequencies[bpfo_mask][bpfo_i]

fault_bpfo_mask = (fault_positive_frequencies >= 95) & (fault_positive_frequencies <= 120)

fault_bpfo_max_magn = max(fault_positive_magnitude[fault_bpfo_mask])
fault_bpfo_i = np.argmax(fault_positive_magnitude[fault_bpfo_mask])
fault_bpfo_max_freq = fault_positive_frequencies[fault_bpfo_mask][fault_bpfo_i]

print(f'BPFI: {bpfi_max_freq} Hz, Magnitude: {bpfi_max_magn}')
print(f'BSF: {bsf_max_freq} Hz, Magnitude: {bsf_max_magn}')
print(f'BPFO: {bpfo_max_freq} Hz, Magnitude: {bpfo_max_magn}')
print(f'Faulty BPFI: {fault_bpfi_max_freq} Hz, Magnitude: {fault_bpfi_max_magn}')
print(f'Faulty BSF: {fault_bsf_max_freq} Hz, Magnitude: {fault_bsf_max_magn}')
print(f'Faulty BPFO: {fault_bpfo_max_freq} Hz, Magnitude: {fault_bpfo_max_magn}')


window_size = 0.1
samples_per_window = int(window_size * fs)
rms_values = []
healthy_peak = []
healthy_kurtosis = []

for start in range(0, len(vibration), samples_per_window):
    end = start + samples_per_window
    window = vibration[start:end]
    
    if len(window) < samples_per_window:
       continue        

    rms_window, h_peak, h_kurtosis = extract_features(window)

    rms_values.append(rms_window)
    healthy_peak.append(h_peak)
    healthy_kurtosis.append(h_kurtosis) 
    



faulty_rms = []
faulty_peak = []
faulty_kurtosis = []
for faulty_start in range(0, len(fault_vibration), samples_per_window):
    fault_end = faulty_start + samples_per_window
    faulty_window = fault_vibration[faulty_start:fault_end]

    if len(faulty_window) < samples_per_window:
       continue

    f_rms_window, f_peak, f_kurtosis = extract_features(faulty_window)
    faulty_rms.append(f_rms_window)
    faulty_peak.append(f_peak)
    faulty_kurtosis.append(f_kurtosis)

print(len(faulty_rms))
print(len(faulty_peak))
print(len(faulty_kurtosis))



healthy_df = pd.DataFrame({
    "RMS": rms_values,
    "Peak": healthy_peak,
    "Kurtosis": healthy_kurtosis
})
print(healthy_df)

faulty_df=pd.DataFrame({
    "RMS": faulty_rms,
    "Peak": faulty_peak,
    "Kurtosis": faulty_kurtosis
})
print(faulty_df)

df = pd.concat([healthy_df, faulty_df], ignore_index=True)
print(healthy_df.shape)
print(faulty_df.shape)
print(df.shape)

list1 = [0] * len(healthy_df)
list2 = [1] * len(faulty_df)
labels = list1 + list2
print(len(labels))
df["Label"] = labels

print(df.head())
print(df.dtypes)
print(df.shape)
print(df["Label"].value_counts())

plt.scatter(rms_values, healthy_peak, label="Healthy")
plt.scatter(faulty_rms, faulty_peak, label="Faulty")
plt.legend()


X = df[["RMS", "Peak", "Kurtosis"]]
y = df["Label"]

print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(np.unique(y_pred))

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

print(y_test.to_numpy())
print(y_pred)

cm = confusion_matrix(y_test, y_pred)
print(cm)