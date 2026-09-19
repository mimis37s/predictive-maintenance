import numpy as np
from scipy.stats import kurtosis

def extract_features(vibration):
    values=vibration**2
    mean=sum(values)/len(values)
    rms=np.sqrt(mean)

    peak=max(np.abs(vibration))
    kurt=kurtosis(vibration)
    return rms, peak, kurt