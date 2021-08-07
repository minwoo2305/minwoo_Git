import os
import numpy as np
import math
import csv
import hrvanalysis


def time_domain_features(data_list):
    diff_list = []
    temp = data_list[0]
    for data in data_list[1:]:
        diff_list.append(data - temp)
        temp = data

    data_list = np.array(data_list)

    sdnn = np.std(data_list)

    sum = 0
    length = len(diff_list)
    for d in diff_list:
        sum += math.pow(d, 2)
    mssd = sum / length

    rmssd = math.sqrt(mssd)

    diff_list = np.array(diff_list)

    sdsd = np.std(diff_list)

    diff_length = len(diff_list)
    ms5_count = 0
    ms10_count = 0
    ms50_count = 0
    ms100_count = 0
    for diff in diff_list:
        if abs(diff) >= 0.005:
            ms5_count += 1
        if abs(diff) >= 0.01:
            ms10_count += 1
        if abs(diff) >= 0.05:
            ms50_count += 1
        if abs(diff) >= 0.1:
            ms100_count += 1

    pnn5 = ms5_count / diff_length
    pnn10 = ms10_count / diff_length
    pnn50 = ms50_count / diff_length
    pnn100 = ms100_count / diff_length

    return sdnn, rmssd, sdsd, pnn5, pnn10, pnn50, pnn100


def wavelet(data_list):
    length = len(data_list)

    wavelet_a = []
    wavelet_d = []
    for idx in range(0, length, 2):
        if idx + 1 == length:
            pass
        else:
            wavelet_a.append(data_list[idx] + data_list[idx + 1])
            wavelet_d.append(data_list[idx] - data_list[idx + 1])

    return wavelet_a, wavelet_d


def wavelet_transformed_features(data_list):
    sum = 0
    length = len(data_list)
    for data in data_list:
        sum += abs(data)
    mean_abs = sum / length

    sum = 0
    for data in data_list:
        sum += math.pow(data, 2)
    power = sum / length

    data_list = np.array(data_list)
    std = np.std(data_list)

    return mean_abs, power, std


def poincare_transformed_features(data_list):
    ptf = hrvanalysis.get_poincare_plot_features(data_list)

    return ptf['sd1'], ptf['sd2']


def get_features_zscore(data_list):
    sdnn, rmssd, sdsd, pnn5, pnn10, pnn50, pnn100 = time_domain_features(data_list)

    wavelet_a1, wavelet_d1 = wavelet(data_list)
    wavelet_a2, wavelet_d2 = wavelet(wavelet_a1)
    wavelet_a3, wavelet_d3 = wavelet(wavelet_a2)

    mean_abs_a3, power_a3, std_a3 = wavelet_transformed_features(wavelet_a3)
    mean_abs_d3, power_d3, std_d3 = wavelet_transformed_features(wavelet_d3)
    mean_abs_d2, power_d2, std_d2 = wavelet_transformed_features(wavelet_d2)

    sd1, sd2 = poincare_transformed_features(data_list)

    sdnn_m = 0.182289816
    sdnn_s = 0.153656585
    sdnn = (sdnn - sdnn_m) / sdnn_s

    rmssd_m = 0.181264743
    rmssd_s = 0.229303862
    rmssd = (rmssd - rmssd_m) / rmssd_s

    sdsd_m = 0.181264711
    sdsd_s = 0.229303861
    sdsd = (sdsd - sdsd_m) / sdsd_s

    pnn5_m = 0.844068123
    pnn5_s = 0.0904039
    pnn5 = (pnn5 - pnn5_m) / pnn5_s

    pnn10_m = 0.694407656
    pnn10_s = 0.138989275
    pnn10 = (pnn10 - pnn10_m) / pnn10_s

    pnn50_m = 0.284970575
    pnn50_s = 0.222569544
    pnn50 = (pnn50 - pnn50_m) / pnn50_s

    pnn100_m = 0.178502518
    pnn100_s = 0.187612666
    pnn100 = (pnn100 - pnn100_m) / pnn100_s

    m_a3_m = 6.228011784
    m_a3_s = 0.955622514
    mean_abs_a3 = (mean_abs_a3 - m_a3_m) / m_a3_s

    m_d3_m = 0.175986742
    m_d3_s = 0.094305146
    mean_abs_d3 = (mean_abs_d3 - m_d3_m) / m_d3_s

    m_d2_m = 0.108836452
    m_d2_s = 0.065035656
    mean_abs_d2 = (mean_abs_d2 - m_d2_m) / m_d2_s

    p_a3_m = 40.88988068
    p_a3_s = 12.60319715
    power_a3 = (power_a3 - p_a3_m) / p_a3_s

    p_d3_m = 0.351688831
    p_d3_s = 1.452866003
    power_d3 = (power_d3 - p_d3_m) / p_d3_s

    p_d2_m = 0.169314871
    p_d2_s = 0.70886519
    power_d2 = (power_d2 - p_d2_m) / p_d2_s

    s_a3_m = 0.98982147
    s_a3_s = 0.485286864
    std_a3 = (std_a3 - s_a3_m) / s_a3_s

    s_d3_m = 0.387808978
    s_d3_s = 0.453933996
    std_d3 = (std_d3 - s_d3_m) / s_d3_s

    s_d2_m = 0.262869855
    s_d2_s = 0.320306499
    std_d2 = (std_d2 - s_d2_m) / s_d2_s

    sd1_m = 0.128174541
    sd1_s = 0.162143228
    sd1 = (sd1 - sd1_m) / sd1_s

    sd2_m = 0.217895944
    sd2_s = 0.15344188
    sd2 = (sd2 - sd2_m) / sd2_s

    return sdnn, rmssd, sdsd, pnn5, pnn10, pnn50, pnn100, mean_abs_a3, mean_abs_d3, mean_abs_d2, \
           power_a3, power_d3, power_d2, std_a3, std_d3, std_d2, sd1, sd2


def get_features(data_list):
    sdnn, rmssd, sdsd, pnn5, pnn10, pnn50, pnn100 = time_domain_features(data_list)

    wavelet_a1, wavelet_d1 = wavelet(data_list)
    wavelet_a2, wavelet_d2 = wavelet(wavelet_a1)
    wavelet_a3, wavelet_d3 = wavelet(wavelet_a2)

    mean_abs_a3, power_a3, std_a3 = wavelet_transformed_features(wavelet_a3)
    mean_abs_d3, power_d3, std_d3 = wavelet_transformed_features(wavelet_d3)
    mean_abs_d2, power_d2, std_d2 = wavelet_transformed_features(wavelet_d2)

    sd1, sd2 = poincare_transformed_features(data_list)

    return sdnn, rmssd, sdsd, pnn5, pnn10, pnn50, pnn100, mean_abs_a3, mean_abs_d3, mean_abs_d2, \
           power_a3, power_d3, power_d2, std_a3, std_d3, std_d2, sd1, sd2


def get_features_ver2(data_list):
    sdnn, rmssd, sdsd, pnn5, pnn10, pnn50, pnn100 = time_domain_features(data_list)

    sd1, sd2 = poincare_transformed_features(data_list)

    return sdnn, rmssd, sdsd, pnn5, pnn10, pnn50, pnn100, sd1, sd2


# file_path = 'C:/Users/minwo/바탕 화면/AF_Database/1min/MIT-BIH_Normal_Sinus_Rhythm_Database(nsrdb)'
file_path = 'C:/Users/minwo/바탕 화면/AF_Database/1min/MIT-BIH_Atrial_Fibrillation_Database(afdb)'
file_list = os.listdir(file_path)

f = open('AF_features.csv', 'a', encoding='utf-8', newline='')
wr = csv.writer(f)

wr.writerow(['SDNN', 'RMSSD', 'SDSD', 'pNN5', 'pNN10', 'pNN50', 'pNN100',
            'Mean(abs_a3)', 'Mean(abs_d3)', 'Mean(abs_d2)', 'Power_a3', 'Power_d3', 'Power_d2',
            'Std_a3', 'Std_d3', 'Std_d2', 'SD1', 'SD2', 'label'])

# wr.writerow(['SDNN', 'RMSSD', 'SDSD', 'pNN5', 'pNN10', 'pNN50', 'pNN100', 'SD1', 'SD2', 'label'])
for file_name in file_list:
    f = open(file_path + '/' + file_name, 'r')

    data_list = []
    dataset = f.read()
    dataset = dataset.split()

    for data in dataset[:]:
        data_list.append(float(data))

    sdnn, rmssd, sdsd, pnn5, pnn10, pnn50, pnn100 = time_domain_features(data_list)

    wavelet_a1, wavelet_d1 = wavelet(data_list)
    wavelet_a2, wavelet_d2 = wavelet(wavelet_a1)
    wavelet_a3, wavelet_d3 = wavelet(wavelet_a2)

    mean_abs_a3, power_a3, std_a3 = wavelet_transformed_features(wavelet_a3)
    mean_abs_d3, power_d3, std_d3 = wavelet_transformed_features(wavelet_d3)
    mean_abs_d2, power_d2, std_d2 = wavelet_transformed_features(wavelet_d2)

    sd1, sd2 = poincare_transformed_features(data_list)

    # label: AF - 1, Normal - 0
    wr.writerow([sdnn, rmssd, sdsd, pnn5, pnn10, pnn50, pnn100, mean_abs_a3, mean_abs_d3, mean_abs_d2,
                 power_a3, power_d3, power_d2, std_a3, std_d3, std_d2, sd1, sd2, 1])

f.close()


