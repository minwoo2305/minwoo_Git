import numpy as np
import math
from scipy import stats
import csv
import os


def pearson_coef(feature, label):
    # correlation_coefficient[0] = correlation
    # correlation_coefficient[1] = p-value
    correlation_coefficient = stats.pearsonr(feature, label)

    coef = correlation_coefficient[0]

    if math.isnan(coef):
        return 0

    return abs(coef)


def point_biserial_coef(feature, label):
    # correlation_coefficient[0] = correlation
    # correlation_coefficient[1] = p-value
    point_biserial = stats.pointbiserialr(feature, label)

    coef = point_biserial[0]

    if math.isnan(coef):
        return 0

    return abs(coef)

'''
for f_num in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]:
    for step in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']:
            # file_path = 'C:/Users/minwo/바탕 화면/save_features/2_class/SCI_vs_MCI&AD/' + step
            file_path = 'C:/Users/minwo/바탕 화면/save_features/2_class/MCI_vs_AD/' + step
            file_list = os.listdir(file_path)

            for file_name in file_list:
                if file_name.find('pearson') is -1:
                    if file_name.find('csv') is not -1:
                        img_path = file_path + '/' + file_name

                        dataset = np.genfromtxt(img_path, delimiter=',', encoding='UTF8')
                        dataset = dataset[1:]
                        print('len_of_original_features : ', len(dataset[0]) - 1)

                        dataset = np.array(dataset)

                        scores = []
                        for index in range(len(dataset[0]) - 1):
                          score = pearson_coef(dataset[:, index], dataset[:, -1])
                          # score = point_biserial_coef(dataset[:, index], dataset[:, -1])
                          temp = [score, index]
                          scores.append(temp)

                        sorted_list = sorted(scores, key=lambda x: x[0], reverse=True)
                        sorted_list = np.array(sorted_list)
                        pearson_rank_index = sorted_list[:, 1]

                        feature_name = np.genfromtxt(img_path, delimiter=',', encoding='UTF8', dtype=str)
                        feature_name = feature_name[0]
                        temp = []

                        original_dataset = np.genfromtxt(img_path, delimiter=',', encoding='UTF8', skip_header=1)
                        labels = original_dataset[:, -1]
                        labels = np.expand_dims(labels, axis=1)
                        dataset = original_dataset[:, :-1]
                        dataset = np.transpose(dataset)
                        feature_name = np.transpose(feature_name)

                        subset = []
                        f_name = []
                        for index in pearson_rank_index[:f_num]:
                          subset.append(dataset[int(index)])
                          f_name.append(feature_name[int(index)])

                        f_name.append('AA')
                        print('len_of_pearson_features : ', len(subset))

                        subset = np.transpose(subset)
                        f_name = np.transpose(f_name)
                        f_name = np.expand_dims(f_name, axis=0)

                        dataset = np.concatenate((subset, labels), axis=1)
                        pearson_dataset = np.concatenate((f_name, dataset), axis=0)

                        save_path = file_path + '/pearson'
                        f = open(save_path + '/pearson_' + str(len(subset[0])) + '_' + file_name, 'w', encoding='utf-8', newline='')
                        wr = csv.writer(f)
                        for line in pearson_dataset:
                          wr.writerow(line)
                        f.close()
'''

for f_num in range(2000, 6001, 100):
    dataset = np.genfromtxt('S189-O216-210215.csv', delimiter=',', encoding='UTF8')
    dataset = dataset[1:]
    print('len_of_original_features : ', len(dataset[0]) - 1)

    dataset = np.array(dataset)

    scores = []
    for index in range(len(dataset[0]) - 1):
      score = pearson_coef(dataset[:, index], dataset[:, -1])
      # score = point_biserial_coef(dataset[:, index], dataset[:, -1])
      temp = [score, index]
      scores.append(temp)

    sorted_list = sorted(scores, key=lambda x: x[0], reverse=True)
    sorted_list = np.array(sorted_list)
    pearson_rank_index = sorted_list[:, 1]

    feature_name = np.genfromtxt('S189-O216-210215.csv', delimiter=',', encoding='UTF8', dtype=str)
    feature_name = feature_name[0]
    temp = []

    original_dataset = np.genfromtxt('S189-O216-210215.csv', delimiter=',', encoding='UTF8', skip_header=1)
    labels = original_dataset[:, -1]
    labels = np.expand_dims(labels, axis=1)
    dataset = original_dataset[:, :-1]
    dataset = np.transpose(dataset)
    feature_name = np.transpose(feature_name)

    subset = []
    f_name = []
    for index in pearson_rank_index[:f_num]:
      subset.append(dataset[int(index)])
      f_name.append(feature_name[int(index)])

    f_name.append('AA')
    print('len_of_pearson_features : ', len(subset))

    subset = np.transpose(subset)
    f_name = np.transpose(f_name)
    f_name = np.expand_dims(f_name, axis=0)

    dataset = np.concatenate((subset, labels), axis=1)
    pearson_dataset = np.concatenate((f_name, dataset), axis=0)

    save_path = 'C:/Users/minwo/바탕 화면/csv_files'
    f = open(save_path + '/pearson_' + str(len(subset[0])) + '_S189-O216-210215.csv', 'w', encoding='utf-8', newline='')
    wr = csv.writer(f)
    for line in pearson_dataset:
      wr.writerow(line)
    f.close()
