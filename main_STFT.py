import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import signal
from sklearn.preprocessing import MinMaxScaler


def signal_plot(data, fs, file_name):
    plt.close('all')
    plt.ioff()
    t = np.arange(0, len(data) / fs, 1 / fs)
    plt.plot(t, data, color='k')
    plt.savefig(file_name, format='png')
    print("Img. Signal Saved")


def Spectrogram_STFT(x, fs, file_name, nperseg_num):
    time_value = round(fs * nperseg_num)  # nperseg 값 계산
    f, t, Sxx = signal.stft(x, fs, nperseg=time_value)

    plt.close('all')
    plt.ioff()

    plt.pcolormesh(t, f, np.abs(Sxx), shading='gouraud')
    plt.ylim(0, 50)
    plt.savefig(file_name, format='png')
    print("Img. STFT Saved")

if __name__ == '__main__':
    path = " ecg_normal.csv"  # select csv file in RAW_DATA
    df = pd.read_csv(path)
    fs = 128  # sampling rate value   N:128hz / AF, ST, VF: 250hz / clinical : 500hz
    shift = 0.3  # common shifted baseline
    second_num = 7  # time value for STFT, signal image
    nperseg_num = 0.1  # STFT time window value(nperseg)
    img_num = 50

    for h in range(40):
        ecg_data = df.iloc[:, h]  # read subject’s raw data

        ecg_data_full = []
        a = (len(ecg_data)) - np.count_nonzero(np.isnan(ecg_data))
        b = fs * 5
        len_data = a // b

        for i in range(len_data):
            crop_data = ecg_data[(i * fs * 5): (i * fs * 5) + (5 * fs) - 1]  # cut data to 5s

            ecg_data_crop = crop_data.values.reshape(-1, 1)  # normalization
            scaler = MinMaxScaler()
            scaler.fit(ecg_data_crop)
            scaler_scaled = scaler.transform(ecg_data_crop)
            data = list(scaler_scaled)
            normalized_data = [x[0] for x in data]
            mean_data = np.mean(normalized_data)
            shift_data = normalized_data + (shift - mean_data)  # shift baseline
            shift_data = shift_data.tolist()
            ecg_data_full.append(shift_data)

        shifted_normal = [item for sublist in ecg_data_full for item in sublist]

        for it in range(img_num):
            data_crop = shifted_normal[(it * fs):(it * fs) + round(second_num * fs) - 1]

            # save img (set your file path in ‘path’)
            file_name1 = path.split('admin')[0] + 'path' + str(h) + '_' + str(it) + '.png'
            print(file_name1)
            print(path.split('.'))
            print(str(h + 1) + ' signal iter:' + str(it))
            signal_plot(data_crop, fs, file_name1)

            file_name2 = path.split('admin')[0] + 'path' + str(h) + '_' + str(it) + '.png'
            print(file_name2)
            print(path.split('.'))
            print(str(h + 1) + ' stft iter:' + str(it))
            Spectrogram_STFT(data_crop, fs, file_name2, nperseg_num)
