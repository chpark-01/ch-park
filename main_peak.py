import heartpy.peakdetection
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import neurokit2 as nk
import math

if __name__ == '__main__':
    path1 = "ecg_00.csv"

    fs = 128  # sampling rate value   N:128hz / AF, ST, VF: 250hz
    shift = 0.3  # common shifted baseline
    df = pd.read_csv(path1)

normal = []
for it in range(40):
    ecg_data1 = df.iloc[:, it]
    a = len(ecg_data1) - np.count_nonzero(np.isnan(ecg_data1))
    ecg_data1 = ecg_data1[:a]

    # remove baseline wander by using notch filter
    ecg_data1 = heartpy.remove_baseline_wander(ecg_data1, fs, cutoff=0.05)

reak_method = 'zong2003'  # peak detection algorithm for open source data
reak_method = 'manikandan2012' # peak detection algorithm for clinical data

ecg_data_crop = ecg_data1.reshape(-1, 1)  # normalization
scaler = MinMaxScaler()
scaler.fit(ecg_data_crop)
scaler_scaled = scaler.transform(ecg_data_crop)
data = list(scaler_scaled)
normal_data = [x[0] for x in data]
mean_data = np.mean(normal_data)

shift_data1 = normal_data + (shift - mean_data)  # shift baseline
_, rpeaks1 = nk.ecg_peaks(shift_data1, sampling_rate=fs)  # peak detection

R_index1 = rpeaks1['ECG_R_Peaks']

interval1 = []
for i in range(len(R_index1) - 1):
    index = (R_index1[i + 1] - R_index1[i]) / fs
    interval1.append(index)

sdnn1 = np.std(interval1) * 1000

print('%d st sdnn :  %f' % (it + 1, sdnn1))
normal.append(sdnn1)
