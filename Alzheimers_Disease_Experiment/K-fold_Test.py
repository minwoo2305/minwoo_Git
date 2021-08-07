from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import numpy
import os

for class_num, category in [(2, 'SCI_vs_AD')]:

    data_path = 'E:/features/' + category + '/final_block_features'
    data_list = os.listdir(data_path)

    for data_name in data_list:
        if data_name.find('csv') is not -1:
            data = data_path + '/' + data_name

            result_list = []
            iterations = 10

            dataset = numpy.loadtxt(data, delimiter=",", skiprows=1)
            features_num = len(dataset[0, :-1])
            print(features_num)


            def create_model():
                # create model
                model = Sequential()
                model.add(Dense(50, input_dim=features_num, activation='relu'))
                model.add(Dense(10, activation='relu'))
                model.add(Dense(class_num, activation='softmax'))
                # Compile model
                model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

                return model


            for i in range(iterations):
                # f = open("result_file_block_" + category + ".txt", 'a')

                X = dataset[:, 0:features_num]
                Y = dataset[:, features_num]

                model = KerasClassifier(build_fn=create_model, epochs=100, verbose=0)

                kfold = StratifiedKFold(n_splits=5, shuffle=True)
                results = cross_val_score(model, X, Y, cv=kfold)

                print(str(i + 1) + " step : " + str(numpy.average(results)))
                # f.write(str(i + 1) + " step : " + str(numpy.average(results)) + "\n")
                # f.write(str(i + 1) + " step result list : " + str(results) + "\n")
                # f.write("==========================================================" + "\n")

                result_list.append(numpy.average(results))

                K.clear_session()
                # f.close()

            f = open("result_file_final_block_features_" + category + ".txt", 'a')

            print(data_name + " average accuracy : " + str(sum(result_list) / iterations))
            print(data_name + " maximum accuracy : " + str(max(result_list)))
            print(data_name + " minimum accuracy : " + str(min(result_list)))
            print("========================================================================")

            f.write(data_name + " average accuracy : " + str(sum(result_list) / iterations) + "\n")
            f.write(data_name + " maximum accuracy : " + str(max(result_list)) + "\n")
            f.write(data_name + " minimum accuracy : " + str(min(result_list)) + "\n")
            f.write("========================================================================\n")

            f.close()
