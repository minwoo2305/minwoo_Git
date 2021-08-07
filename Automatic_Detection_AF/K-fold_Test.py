from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import numpy
import time

result_list = []
iterations = 100
features_num = 18


def create_model():
    # create model
    model = Sequential()
    model.add(Dense(10, input_dim=features_num, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


start_time = time.time()

dataset = numpy.loadtxt('AF_features_features_18.csv', delimiter=",", skiprows=1)

for i in range(iterations):
    # f = open("result_file_AF.txt", 'a')

    X = dataset[:, 0:features_num]
    Y = dataset[:, features_num]

    model = KerasClassifier(build_fn=create_model, epochs=300, verbose=0)

    kfold = StratifiedKFold(n_splits=5, shuffle=True)
    results = cross_val_score(model, X, Y, cv=kfold)

    print(str(i + 1) + " step : " + str(numpy.average(results)))
    # f.write(str(i + 1) + " step : " + str(numpy.average(results)) + "\n")
    # f.write(str(i + 1) + " step result list : " + str(results) + "\n")
    # f.write("==========================================================" + "\n")

    result_list.append(numpy.average(results))

    K.clear_session()
    # f.close()

# f = open("result_file_AF.txt", 'a')

print("average accuracy : " + str(sum(result_list) / iterations))
print("maximum accuracy : " + str(max(result_list)))
print("minimum accuracy : " + str(min(result_list)))

# f.write("average accuracy : " + str(sum(result_list) / iterations) + "\n")
# f.write("maximum accuracy : " + str(max(result_list)) + "\n")
# f.write("minimum accuracy : " + str(min(result_list)) + "\n")

'''
end_time = time.time()
total_time = end_time - start_time
minute = total_time // 60
total_time = total_time % 60
hour = minute // 60
minute = minute % 60

print("running time : " + str(hour) + "h " + str(minute) + "m " + str(total_time) + "s")
print("==========================================================")

f.write("running time : " + str(hour) + "h " + str(minute) + "m " + str(total_time) + "s" + "\n")
f.write("==========================================================" + "\n")
'''

# f.close()
