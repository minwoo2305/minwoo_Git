from keras import models
from keras import layers
from keras.optimizers import Adam
import numpy
from keras import backend as K
from keras.callbacks import EarlyStopping
import random

for step in range(100):
    for step_no in ['2_class/MCI_vs_AD']:

        f_num = 100

        data_path = 'C:/Users/minwo/바탕 화면/save_features/' + step_no + '/total/pearson'
        dataset = numpy.loadtxt(data_path + '/pearson_' + str(f_num) + '_total_img_features.csv', delimiter=",",
                                skiprows=1)

        random.shuffle(dataset)

        train_index = len(dataset) * 0.8
        train_index = int(train_index)

        train_data = dataset[:train_index, :-1]
        train_labels = dataset[:train_index, -1]

        test_data = dataset[train_index:, :-1]
        test_labels = dataset[train_index:, -1]

        network = models.Sequential()
        network.add(layers.Dense(5, activation='relu', input_dim=f_num))
        network.add(layers.Dense(5, activation='relu'))
        network.add(layers.Dense(1, activation='sigmoid'))

        network.compile(optimizer=Adam(lr=0.0000165), loss='binary_crossentropy', metrics=['accuracy'])
        # early_stopping = EarlyStopping(monitor='acc', patience=5, mode='auto')
        network.fit(train_data, train_labels, epochs=300, verbose=0)

        test_loss, test_acc = network.evaluate(test_data, test_labels)
        test_acc = round(test_acc, 3)

        print(step_no + ' test_acc :', test_acc)

        network.save('./save_model/' + step_no + '/save_model_' + str(test_acc) + '.h5')

        K.clear_session()
