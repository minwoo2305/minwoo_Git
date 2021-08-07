import numpy
import random
from keras import models
from keras import layers
from keras.utils import to_categorical
import matplotlib.pyplot as plt

features_num = 18

dataset = numpy.loadtxt('AF_features_features_18.csv', delimiter=",", skiprows=1)

random.shuffle(dataset)

length = len(dataset)
idx = int(length * 0.8)

X_train = dataset[:idx, 0:features_num]
Y_train = dataset[:idx, features_num]

X_test = dataset[:idx, 0:features_num]
Y_test = dataset[:idx, features_num]

Y_train = to_categorical(Y_train, 2)
Y_test = to_categorical(Y_test, 2)

network = models.Sequential()
network.add(layers.Dense(10, activation='relu', input_shape=(features_num,)))
network.add(layers.Dense(10, activation='relu'))
network.add(layers.Dense(2, activation='softmax'))

network.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


history = network.fit(X_train, Y_train, epochs=100)

test_loss, test_acc = network.evaluate(X_test, Y_test)

print('test_acc:', test_acc)
network.save('AF_model_18_softmax.h5')

acc = history.history['acc']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc)
plt.title('Training Acc')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.show()

print(network.predict(X_test))
