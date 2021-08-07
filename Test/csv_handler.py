import csv
import os


def func(path, num, folder, gender):
    f1 = open(path + str(num) + '.csv', 'r')
    data_list = list(csv.reader(f1))

    save_path = 'D:/DAIC_WOZ_Data/Data/SoX/Features_for_DNN/'
    f2 = open(save_path + folder + '/' + folder + '_Praat_Total_Feature.csv', 'a', newline='')
    f3 = open(save_path + folder + '/' + folder + '_Praat_Total_Feature_' + gender + '.csv', 'a', newline='')
    wr1 = csv.writer(f2)
    wr2 = csv.writer(f3)

    for line in data_list[1:]:
        wr1.writerow(line)
        wr2.writerow(line)

    f1.close()
    f2.close()
    f3.close()

    print(num + ' finish!')


folder = "train"
binary = "PHQ-8_binary_1"
gen = "woman"

print(folder + " " + binary + " " + gen + " start")

file_path = 'D:/DAIC_WOZ_Data/Data/SoX/Features_for_DNN/' + folder + '/' + binary + '/' + gen + '/'
file_list = os.listdir(file_path)

for file_name in file_list:
    num = file_name[:3]
    func(file_path, num, folder, gen)
