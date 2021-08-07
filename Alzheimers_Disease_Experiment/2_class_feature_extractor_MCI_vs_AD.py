from keras.models import load_model, Model
from keras.preprocessing import image
from keras import models
import numpy as np
import pandas as pd
import os
from keras import backend as K


def feature_extract(data_path, model_name, save_path, model_path, step_num):
    model = load_model(model_path + '/' + model_name)

    flatten = model.get_layer('flatten')
    test_model = Model(inputs=model.input, outputs=flatten.output)

    ad_img_dir = data_path + '/' + 'AD'
    mci_img_dir = data_path + '/' + 'MCI'

    ad_img = []
    for img_name in os.listdir(ad_img_dir):
        img_path = os.path.join(ad_img_dir, img_name)
        img = image.load_img(img_path, target_size=(100, 100))
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor /= 255.
        ad_img.append(img_tensor)

    mci_img = []
    for img_name in os.listdir(mci_img_dir):
        img_path = os.path.join(mci_img_dir, img_name)
        img = image.load_img(img_path, target_size=(100, 100))
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor /= 255.
        mci_img.append(img_tensor)

    ad_labels = np.array([0 for i in range(len(ad_img))])
    mci_labels = np.array([1 for i in range(len(mci_img))])

    total_labels = np.concatenate((ad_labels, mci_labels), axis=0)

    img_features = []
    for img in ad_img:
        img_features.append(test_model.predict(img)[0])

    for img in mci_img:
        img_features.append(test_model.predict(img)[0])

    # print(len(img_features))

    # length of image features
    # print(len(img_features[0]))

    # Dataset + Label
    feature_set = []
    for i in range(len(total_labels)):
        temp = np.append(img_features[i], total_labels[i])
        feature_set.append(temp)

    # Feature column name
    features_column = []
    for i in range(len(img_features[0])):
        features_column.append('img_f' + str(i) + '_' + str(step_num))

    features_column.append('AA')

    df = pd.DataFrame(feature_set, columns=features_column)
    if model_name[-7:-3] == '00.0':
        df.to_csv(save_path + '/img_features_' + model_name[-8:-3] + '.csv', index=False)
    else:
        df.to_csv(save_path + '/img_features_' + model_name[-7:-3] + '.csv', index=False)


for step_no in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:

    save_path = 'C:/Users/minwo/바탕 화면/AD_Dataset/save_features/2_class/MCI_vs_AD/' + str(step_no)
    # save_path = 'C:/Users/minwo/바탕 화면/Data/save_feature/2_class/MCI_vs_AD/' + str(step_no)
    model_path = 'E:/save_ver.4/save_model/2_class/MCI_vs_AD/' + str(step_no)
    data_path = 'C:/Users/minwo/바탕 화면/Speech Data/data/backup2/data/' + str(step_no) + '/img_data'
    # data_path = 'C:/Users/minwo/바탕 화면/Data/' + str(step_no) + '/img_data'

    model_list = os.listdir(model_path)

    for model_name in model_list:
        if model_name.find('h5') is not -1:
            feature_extract(data_path, model_name, save_path, model_path, step_no)
            print(str(step_no) + ' is done')
            K.clear_session()
