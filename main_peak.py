import heartpy.peakdetection
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import neurokit2 as nk

if __name__ == '__main__':
    path = "ecg_00.csv" # select csv file in RAW_DATA
    df = pd.read_csv(path)

    fs = 128  # sampling rate value   N:128hz / AF, ST, VF: 250hz
    shift = 0.3  # common shifted baseline

    ecg_data_full = []
    for it in range(40):
        ecg_data = df.iloc[:, it]
        a = len(ecg_data) - np.count_nonzero(np.isnan(ecg_data))
        ecg_data = ecg_data[:a]

        # remove baseline wander by using notch filter
        ecg_data = heartpy.remove_baseline_wander(ecg_data, fs, cutoff=0.05)

        reak_method = 'zong2003'  # peak detection algorithm for open source data
        # reak_method = 'manikandan2012' # peak detection algorithm for clinical data

        ecg_data_crop = ecg_data.reshape(-1, 1)  # normalization
        scaler = MinMaxScaler()
        scaler.fit(ecg_data_crop)
        scaler_scaled = scaler.transform(ecg_data_crop)
        data = list(scaler_scaled)
        normalized_data = [x[0] for x in data]
        mean_data = np.mean(normalized_data)

        shift_data = normalized_data + (shift - mean_data)  # shift baseline
        _, rpeaks = nk.ecg_peaks(shift_data, sampling_rate=fs)  # peak detection

        R_index = rpeaks['ECG_R_Peaks']

        interval = []
        for i in range(len(R_index) - 1):
            index = (R_index[i + 1] - R_index[i]) / fs
            interval.append(index)

        sdnn = np.std(interval) * 1000

        print('%d st sdnn :  %f' % (it + 1, sdnn))
        ecg_data_full.append(sdnn)
