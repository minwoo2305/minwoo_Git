import os
import numpy as np
import csv

total_dataset = []
total_dataset_name = []

base_path = 'E:/features/SCI_vs_AD/'

for step in ['sorted_block_features_3']:

    data_path = base_path + step
    data_list = os.listdir(data_path)

    for data_name in data_list:
        if data_name.find('csv') is not -1:
            data = data_path + '/' + data_name
            dataset = np.genfromtxt(data, delimiter=',', encoding='UTF8', dtype='float64')
            dataset = dataset[1:, :]
            dataset_name = np.genfromtxt(data, delimiter=',', encoding='UTF8', dtype=str)
            dataset_name = dataset_name[0]

            if data_name is not data_list[-1]:
                dataset = dataset[:, :-1]
                dataset_name = dataset_name[:-1]

            if data_name is data_list[0]:
                total_dataset = dataset
                total_dataset_name = dataset_name
            else:
                total_dataset = np.concatenate((total_dataset, dataset), axis=1)
                total_dataset_name = np.concatenate((total_dataset_name, dataset_name), axis=0)

total_dataset_name = np.expand_dims(total_dataset_name, axis=0)
total_dataset = np.concatenate((total_dataset_name, total_dataset), axis=0)
print("The Number of Total Features : " + str(len(total_dataset[0]) - 1))

'''
f = open(base_path + 'block_features_2/block_total_features.csv', 'w', encoding='utf-8', newline='')
wr = csv.writer(f)
for line in total_dataset:
    wr.writerow(line)
f.close()
'''

len_features = len(total_dataset[0]) - 1
features_of_block = 8000
stride = 8000

num_of_blocks = 1
features_count = features_of_block

while features_count < len_features:
    features_count += stride
    num_of_blocks += 1

print("The Number of Blocks : " + str(num_of_blocks))

dataset = total_dataset[:, :-1]
label = total_dataset[:, -1]
label = np.expand_dims(label, axis=1)

for block_num in range(num_of_blocks):

    subset = dataset[:, (block_num * stride):(block_num * stride) + features_of_block]
    # subset = dataset[:, :100 * block_num + 100]
    block_dataset = np.concatenate((subset, label), axis=1)

    f = open(base_path + 'block_features_4/block_' + str(block_num + 1) + '_features.csv', 'w', encoding='utf-8', newline='')
    wr = csv.writer(f)
    for line in block_dataset:
        wr.writerow(line)
    f.close()

    print("block " + str(block_num + 1) + " has finished")
