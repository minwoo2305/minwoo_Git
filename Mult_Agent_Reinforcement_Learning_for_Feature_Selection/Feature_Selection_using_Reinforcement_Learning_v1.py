import random
import numpy as np
import pandas as pd
from sklearn import neural_network
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import cross_val_score

import warnings
warnings.filterwarnings('ignore')


# file_name = 'wisconsin_breast_cancer'
# file_name = 'forest_cover_type'
# file_name = 'spambase'
# file_name = 'the_insurance_company_benchmark'
# file_name = 'musk'
# file_name = 'arcene' # -> 정규화 이후 결측치 발생
# file_name = 'madelon'
file_name = 'colon_cancer'

data = pd.read_csv(file_name + '.csv')
data = pd.DataFrame(data)
data = shuffle(data)
total_x_data = data.iloc[:, :-1]
y_data = data.iloc[:, -1]
total_normalization_x_data = (total_x_data - np.mean(total_x_data)) / np.std(total_x_data)
total_normalization_x_data = pd.DataFrame(total_normalization_x_data)
y_data = pd.DataFrame(y_data)


def get_reward(features_list):
    if len(features_list) == 0:
        return 0
    else:
        x_data = total_normalization_x_data.iloc[:, features_list]
        x_train, x_test = x_data.iloc[:train_test_ratio, :], x_data.iloc[train_test_ratio:, :]
        y_train, y_test = y_data.iloc[:train_test_ratio], y_data.iloc[train_test_ratio:]

        clf = neural_network.MLPClassifier(hidden_layer_sizes=(5, 2), activation='relu', learning_rate_init=0.01,
                                           max_iter=10)
        clf.fit(x_train, y_train)
        acc = clf.score(x_test, y_test)

    return round(acc, 4)


def get_reward_k_fold(features_list, k):
    if len(features_list) == 0:
        return 0
    else:
        data = total_normalization_x_data.iloc[:, features_list]
        clf = neural_network.MLPClassifier(hidden_layer_sizes=(5, 2), activation='relu', learning_rate_init=0.01,
                                           max_iter=10)
        kfold = StratifiedKFold(n_splits=k, shuffle=True)
        # kfold = KFold(n_splits=k, shuffle=True)
        results = cross_val_score(clf, data, y_data, cv=kfold)

    return round((sum(results) / k), 4)


epsilon = 0.5
alpha = 0.2
epsilon_decay_rate = 0.995
train_test_ratio = int(len(total_x_data) * 0.5)

num_agents = len(total_x_data.iloc[0])

Q_values = []
for i in range(num_agents):
    Q_values.append([random.uniform(0, 1), random.uniform(0, 0.5)])

all_rewards_and_features = []
num_episodes = 1000

actions = [0] * num_agents
for episode in range(num_episodes):
    for agent in range(num_agents):
        rand_number = random.uniform(0, 1)
        if rand_number > epsilon:
            # Exploit
            actions[agent] = np.argmax(Q_values[agent])
        else:
            # Explore
            actions[agent] = random.choice([0, 0, 0, 0, 1])

    features = []
    for i, act in enumerate(actions):
        if act == 1:
            features.append(i)
    # R = get_reward(features)
    R = get_reward_k_fold(features, 5)

    all_rewards_and_features.append([R, features])

    Counter_features = []
    for i, act in enumerate(actions):
        if act == 0:
            Counter_features.append(i)
    # Counter_R = get_reward(Counter_features)
    Counter_R = get_reward_k_fold(Counter_features, 5)

    if Counter_R <= R:
        for agent in range(num_agents):
            Q_values[agent][actions[agent]] += (alpha * R)
    else:
        for agent in range(num_agents):
            Q_values[agent][actions[agent]] -= (alpha * R)

    epsilon = epsilon * epsilon_decay_rate

    all_rewards_and_features = sorted(all_rewards_and_features, key=lambda x: x[0], reverse=True)

    if episode % 100 == 0:
        # print("Episode " + str(episode) + " - Q values : ")
        # print(Q_values)
        print("Episode " + str(episode) + " - Actions : ")
        print(actions)
        print("-------------------------------------------------------------------------")
        print("Episode " + str(episode) + " - Number of Features : " + str(len(features)))
        print("Episode " + str(episode) + " - Selected Features : " + str(features))
        print("Episode " + str(episode) + " - Current Reward : " + str(R))
        print("-------------------------------------------------------------------------")
        print("Maximum Reward : " + str(all_rewards_and_features[0][0]))
        print("Number of Maximum Reward Features : " + str(len(all_rewards_and_features[0][1])))
        print("Features of Maximum Reward : " + str(all_rewards_and_features[0][1]))
        print("=======================================================================================")

total_features = []
for i, act in enumerate(actions):
    total_features.append(i)
total_R = get_reward(total_features)
print(total_R)

'''
# Save Selected Feature csv file
selected_data = total_x_data.iloc[:, all_rewards_and_features[0][1]]
selected_data = pd.DataFrame(selected_data)
save_data = pd.concat([selected_data, y_data], axis=1)
save_data.to_csv('Selected_data_of_' + file_name + '_' + str(all_rewards_and_features[0][0]) + '.csv', index=False)
print('Save selected features of data')
print('Number of selected feature : ' + str(len(all_rewards_and_features[0][1])))
'''