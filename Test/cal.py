import csv
import numpy as np
import os


def feature_cal(path, num, label):
    f1 = open(path + str(num) + '_P/' + str(num) + '_COVAREP_features.csv', 'r')
    f2 = open(path + 'COVAREP_features_set.csv', 'a', newline='')
    data = csv.reader(f1)
    wr1 = csv.writer(f2)

    F0 = []
    NAQ = []
    QOQ = []
    H1H2 = []
    PSP = []
    MDQ = []
    peakSlope = []
    Rd = []
    Rd_conf = []

    for line in data:
        F0.append(line[0])
        NAQ.append(line[1])
        QOQ.append(line[2])
        H1H2.append(line[3])
        PSP.append(line[4])
        MDQ.append(line[5])
        peakSlope.append(line[6])
        Rd.append(line[7])
        Rd_conf.append(line[8])

    F0_max = np.max(np.array(F0, dtype=float))
    F0_min = np.min(np.array(F0, dtype=float))
    F0_mean = np.mean(np.array(F0, dtype=float))
    F0_median = np.median(np.array(F0, dtype=float))
    F0_range = F0_max - F0_min
    F0_std = np.std(np.array(F0, dtype=float))
    F0_var = np.var(np.array(F0, dtype=float))

    NAQ_max = np.max(np.array(NAQ, dtype=float))
    NAQ_min = np.min(np.array(NAQ, dtype=float))
    NAQ_mean = np.mean(np.array(NAQ, dtype=float))
    NAQ_median = np.median(np.array(NAQ, dtype=float))
    NAQ_range = NAQ_max - NAQ_min
    NAQ_std = np.std(np.array(NAQ, dtype=float))
    NAQ_var = np.var(np.array(NAQ, dtype=float))

    QOQ_max = np.max(np.array(QOQ, dtype=float))
    QOQ_min = np.min(np.array(QOQ, dtype=float))
    QOQ_mean = np.mean(np.array(QOQ, dtype=float))
    QOQ_median = np.median(np.array(QOQ, dtype=float))
    QOQ_range = QOQ_max - QOQ_min
    QOQ_std = np.std(np.array(QOQ, dtype=float))
    QOQ_var = np.var(np.array(QOQ, dtype=float))

    H1H2_max = np.max(np.array(H1H2, dtype=float))
    H1H2_min = np.min(np.array(H1H2, dtype=float))
    H1H2_mean = np.mean(np.array(H1H2, dtype=float))
    H1H2_median = np.median(np.array(H1H2, dtype=float))
    H1H2_range = H1H2_max - H1H2_min
    H1H2_std = np.std(np.array(H1H2, dtype=float))
    H1H2_var = np.var(np.array(H1H2, dtype=float))

    PSP_max = np.max(np.array(PSP, dtype=float))
    PSP_min = np.min(np.array(PSP, dtype=float))
    PSP_mean = np.mean(np.array(PSP, dtype=float))
    PSP_median = np.median(np.array(PSP, dtype=float))
    PSP_range = PSP_max - PSP_min
    PSP_std = np.std(np.array(PSP, dtype=float))
    PSP_var = np.var(np.array(PSP, dtype=float))

    MDQ_max = np.max(np.array(MDQ, dtype=float))
    MDQ_min = np.min(np.array(MDQ, dtype=float))
    MDQ_mean = np.mean(np.array(MDQ, dtype=float))
    MDQ_median = np.median(np.array(MDQ, dtype=float))
    MDQ_range = MDQ_max - MDQ_min
    MDQ_std = np.std(np.array(MDQ, dtype=float))
    MDQ_var = np.var(np.array(MDQ, dtype=float))

    peakSlope_max = np.max(np.array(peakSlope, dtype=float))
    peakSlope_min = np.min(np.array(peakSlope, dtype=float))
    peakSlope_mean = np.mean(np.array(peakSlope, dtype=float))
    peakSlope_median = np.median(np.array(peakSlope, dtype=float))
    peakSlope_range = peakSlope_max - peakSlope_min
    peakSlope_std = np.std(np.array(peakSlope, dtype=float))
    peakSlope_var = np.var(np.array(peakSlope, dtype=float))

    Rd_max = np.max(np.array(Rd, dtype=float))
    Rd_min = np.min(np.array(Rd, dtype=float))
    Rd_mean = np.mean(np.array(Rd, dtype=float))
    Rd_median = np.median(np.array(Rd, dtype=float))
    Rd_range = Rd_max - Rd_min
    Rd_std = np.std(np.array(Rd, dtype=float))
    Rd_var = np.var(np.array(Rd, dtype=float))

    Rd_conf_max = np.max(np.array(Rd_conf, dtype=float))
    Rd_conf_min = np.min(np.array(Rd_conf, dtype=float))
    Rd_conf_mean = np.mean(np.array(Rd_conf, dtype=float))
    Rd_conf_median = np.median(np.array(Rd_conf, dtype=float))
    Rd_conf_range = Rd_conf_max - Rd_conf_min
    Rd_conf_std = np.std(np.array(Rd_conf, dtype=float))
    Rd_conf_var = np.var(np.array(Rd_conf, dtype=float))

    wr1.writerow([F0_max, F0_min, F0_mean, F0_median, F0_range, F0_std, F0_var,
                  NAQ_max, NAQ_min, NAQ_mean, NAQ_median, NAQ_range, NAQ_std, NAQ_var,
                  QOQ_max, QOQ_min, QOQ_mean, QOQ_median, QOQ_range, QOQ_std, QOQ_var,
                  H1H2_max, H1H2_min, H1H2_mean, H1H2_median, H1H2_range, H1H2_std, H1H2_var,
                  PSP_max, PSP_min, PSP_mean, PSP_median, PSP_range, PSP_std, PSP_var,
                  MDQ_max, MDQ_min, MDQ_mean, MDQ_median, MDQ_range, MDQ_std, MDQ_var,
                  peakSlope_max, peakSlope_min, peakSlope_mean, peakSlope_median, peakSlope_range, peakSlope_std, peakSlope_var,
                  Rd_max, Rd_min, Rd_mean, Rd_median, Rd_range, Rd_std, Rd_var,
                  Rd_conf_max, Rd_conf_min, Rd_conf_mean, Rd_conf_median, Rd_conf_range, Rd_conf_std, Rd_conf_var,
                  label])

    f1.close()
    f2.close()

    print(num + ' finish!')


file_path = 'E:/DAIC_WOZ_Data/Data/Data/dev/PHQ-8_binary_0/man/'
file_list = os.listdir(file_path)

for file_name in file_list:
    if file_name.find('wav') is not -1:
        num = file_name[:3]
        feature_cal(file_path, num, 0)
