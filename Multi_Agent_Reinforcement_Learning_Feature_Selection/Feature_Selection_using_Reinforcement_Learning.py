import random
import numpy as np
import pandas as pd
from sklearn import neural_network
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

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
# total_normalization_x_data = pd.DataFrame(total_x_data)
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


# Parameters
epsilon = 0.5
alpha = 0.01
epsilon_decay_rate = 0.9995
replay_rate = 0.01
num_agents = len(total_x_data.iloc[0])
num_episodes = 10000
train_test_ratio = int(len(total_x_data) * 0.7)

# initialization of Q values
Q_values = []
for i in range(num_agents):
    Q_values.append([random.uniform(0, 1), random.uniform(0, 0.05)])

History_of_Q_values = np.array([Q_values])

all_rewards_and_features = []

main_actions = [0] * num_agents
guide_actions = [0] * num_agents

for episode in range(1, num_episodes + 1):
    for main_agent_idx in range(num_agents):
        rand_number = random.uniform(0, 1)
        if rand_number > epsilon:
            # Exploit
            main_actions[main_agent_idx] = np.argmax(Q_values[main_agent_idx])
        else:
            # Explore
            main_actions[main_agent_idx] = random.choice([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

    main_features_list = []
    for i, main_act in enumerate(main_actions):
        if main_act == 1:
            main_features_list.append(i)
    main_R = get_reward(main_features_list)
    # main_R = get_reward_k_fold(main_features_list, 5)

    # History
    all_rewards_and_features.append([main_R, main_features_list])
    all_rewards_and_features = sorted(all_rewards_and_features, key=lambda x: len(x[1]))
    all_rewards_and_features = sorted(all_rewards_and_features, key=lambda x: x[0], reverse=True)

    for guide_agent_idx in range(num_agents):
        guide_actions[guide_agent_idx] = random.choice([0, 1])

    guide_features_list = []
    for i, guide_act in enumerate(guide_actions):
        if guide_act == 1:
            guide_features_list.append(i)
    guide_R = get_reward(guide_features_list)
    # guide_R = get_reward_k_fold(guide_features_list, 5)

    '''
    if random.uniform(0, 1) > replay_rate:
        # Reward for Training
        train_R = main_R - guide_R
        # Training
        for agent_idx in range(num_agents):
            if main_actions[agent_idx] != guide_actions[agent_idx]:
                Q_values[agent_idx][main_actions[agent_idx]] += (alpha * train_R)
    else:
        best_actions = [0] * num_agents
        for idx in all_rewards_and_features[0][1]:
            best_actions[idx] = 1

        train_R = main_R - all_rewards_and_features[0][0]
        for agent_idx in range(num_agents):
            if main_actions[agent_idx] != best_actions[agent_idx]:
                Q_values[agent_idx][main_actions[agent_idx]] += (alpha * train_R)
    '''

    # Reward for Training
    train_R = main_R - guide_R
    # Training
    if train_R > 0:
        for agent_idx in range(num_agents):
            if main_actions[agent_idx] != guide_actions[agent_idx]:
                Q_values[agent_idx][main_actions[agent_idx]] += (alpha * train_R)

    History_of_Q_values = np.append(History_of_Q_values, np.array([Q_values]), axis=0)

    epsilon = epsilon * epsilon_decay_rate

    if episode % 100 == 0:
        # print("Episode " + str(episode) + " - Q values : ")
        # print(Q_values)
        print("Episode " + str(episode) + " - Actions : ")
        print(main_actions)
        print("-------------------------------------------------------------------------")
        print("Episode " + str(episode) + " - Number of Original Features : " + str(num_agents))
        print("Episode " + str(episode) + " - Number of Features : " + str(len(main_features_list)))
        print("Episode " + str(episode) + " - Selected Features : " + str(main_features_list))
        print("Episode " + str(episode) + " - Current Accuracy : " + str(main_R))
        print("-------------------------------------------------------------------------")
        print("Maximum Accuracy : " + str(all_rewards_and_features[0][0]))
        print("Number of Maximum Accuracy Features : " + str(len(all_rewards_and_features[0][1])))
        print("Features of Maximum Accuracy : " + str(all_rewards_and_features[0][1]))
        print("=======================================================================================")

'''
# # Save Selected Feature csv file
selected_data = total_x_data.iloc[:, all_rewards_and_features[0][1]]
selected_data = pd.DataFrame(selected_data)
save_data = pd.concat([selected_data, y_data], axis=1)
save_data.to_csv('Selected_data_of_' + file_name + '_' + str(all_rewards_and_features[0][0]) + '.csv', index=False)
print('Save selected features of data')
print('Number of selected feature : ' + str(len(all_rewards_and_features[0][1])))
'''

# History of Q values[:, 0, 0] -> [Total, Agent number, Action]
# plt.plot(range(num_episodes + 1), History_of_Q_values[:, 0, 0])
# plt.xlabel('Episodes')
# plt.ylabel('Q-value')
# plt.show()
# plt.plot(range(num_episodes + 1), History_of_Q_values[:, 0, 1])
# plt.xlabel('Episodes')
# plt.ylabel('Q-value')
# plt.show()
