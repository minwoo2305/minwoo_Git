import os
import numpy as np
import math
import csv
from scipy import stats


def point_biserial_coef(feature, label):
    # correlation_coefficient[0] = correlation
    # correlation_coefficient[1] = p-value

    feature = np.array(feature)
    label = np.array(label)

    feature = feature.astype('float64')
    label = label.astype('float64')

    correlation_coefficient = stats.pointbiserialr(feature, label)

    coef = correlation_coefficient[0]

    if math.isnan(coef):
        return 0

    return coef ** 2


def pearson_coef(feature, label):
    # correlation_coefficient[0] = correlation
    # correlation_coefficient[1] = p-value

    feature = np.array(feature)
    label = np.array(label)

    feature = feature.astype('float64')
    label = label.astype('float64')

    correlation_coefficient = stats.pearsonr(feature, label)

    coef = correlation_coefficient[0]

    if math.isnan(coef):
        return 0

    return abs(coef)

'''
total_dataset = []
total_dataset_name = []

for step in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']:

    data_path = 'C:/Users/minwo/바탕 화면/save_features/2_class/MCI_vs_AD/' + step
    data_list = os.listdir(data_path)

    for data_name in data_list:
        if data_name.find('csv') is not -1:
            data_path = data_path + '/' + data_name
            dataset = np.genfromtxt(data_path, delimiter=',', encoding='UTF8', dtype='float64')
            dataset = dataset[1:, :]
            dataset_name = np.genfromtxt(data_path, delimiter=',', encoding='UTF8', dtype=str)
            dataset_name = dataset_name[0]

            if step is not '11':
                dataset = dataset[:, :-1]
                dataset_name = dataset_name[:-1]

            if step is '1':
                total_dataset = dataset
                total_dataset_name = dataset_name
            else:
                total_dataset = np.concatenate((total_dataset, dataset), axis=1)
                total_dataset_name = np.concatenate((total_dataset_name, dataset_name), axis=0)

total_dataset_name = np.expand_dims(total_dataset_name, axis=0)
total_dataset = np.concatenate((total_dataset_name, total_dataset), axis=0)
print(total_dataset[0])
'''

total_dataset = np.genfromtxt('C:/Users/minwo/바탕 화면/img_features_86.73.csv', delimiter=',', encoding='UTF8', dtype='float64')
total_dataset = total_dataset[1:, :]
total_dataset_name = np.genfromtxt('C:/Users/minwo/바탕 화면/img_features_86.73.csv', delimiter=',', encoding='UTF8', dtype=str)
total_dataset_name = total_dataset_name[0]

for f_num in [200]:

    # dataset = total_dataset[1:]
    dataset = total_dataset
    print('len_of_original_features : ', len(dataset[0]) - 1)
    dataset = np.array(dataset)

    scores = []
    for index in range(len(dataset[0]) - 1):
        # score = pearson_coef(dataset[:, index], dataset[:, -1])
        score = point_biserial_coef(dataset[:, index], dataset[:, -1])
        temp = [score, index]
        scores.append(temp)

    sorted_list = sorted(scores, key=lambda x: x[0], reverse=True)

    # feature_name = total_dataset[0]
    feature_name = total_dataset_name

    # original_dataset = total_dataset[1:, :]
    original_dataset = total_dataset
    labels = original_dataset[:, -1]
    labels = np.expand_dims(labels, axis=1)
    dataset = original_dataset[:, :-1]
    dataset = np.transpose(dataset)
    feature_name = np.transpose(feature_name)

    subset = []
    f_name = []
    coef_list = []

    for coef, index in sorted_list[:f_num]:
        subset.append(dataset[int(index)])
        f_name.append(feature_name[int(index)])
        coef_list.append(coef)

    f_name.append('AA')
    coef_list.append('None')
    print('len_of_pearson_features : ', len(subset))

    subset = np.transpose(subset)
    f_name = np.transpose(f_name)
    f_name = np.expand_dims(f_name, axis=0)

    coef_list = np.transpose(coef_list)
    coef_list = np.expand_dims(coef_list, axis=0)

    dataset = np.concatenate((subset, labels), axis=1)
    temp = np.concatenate((f_name, coef_list), axis=0)
    pearson_dataset = np.concatenate((temp, dataset), axis=0)

    file_path = 'C:/Users/minwo/바탕 화면'

    # f = open(file_path + '/pearson_' + str(len(subset[0])) + '_total_img_features_MCI_vs_AD.csv', 'w', encoding='utf-8', newline='')
    f = open(file_path + '/point_biserial_file.csv', 'w', encoding='utf-8',
             newline='')
    wr = csv.writer(f)
    for line in pearson_dataset:
        wr.writerow(line)
    f.close()

