import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from keras.models import load_model
import feature_extraction

# file_path = 'C:/Users/minwo/바탕 화면/AF_Database/MIT-BIH_Normal_Sinus_Rhythm_Database(nsrdb)'
file_path = 'C:/Users/minwo/바탕 화면/AF_Database/MIT-BIH_Atrial_Fibrillation_Database(afdb)'
file_name = '04126rr.txt'
f = open(file_path + '/' + file_name, 'r')

data_list = []
dataset = f.read()
dataset = dataset.split()

for data in dataset[:]:
    data_list.append(float(data))

f.close()

# wavelet transform -> [0] : A Level, [1] : D Level
# data_list = feature_extraction.wavelet(data_list)[0]

lst_max = max(data_list)
lst_min = min(data_list)

model = load_model('AF_model_18_softmax.h5')
# print(model.summary())

# start_point : x_max
start_point = 60
# interval : speed
interval = 100

length = np.linspace(start_point, len(data_list) + start_point, len(data_list))

test_list = []


class Scope(object):
    def __init__(self, ax, fn, xmax=10, ymax=10, xstart=0, ystart=0, title='Heart Rate Variability(HRV)',
                 xlabel='Number of RR', ylabel='RR interval(s)'):
        self.xmax = xmax  # x축 길이
        self.xstart = xstart  # x축 시작점
        self.ymax = ymax  # y축 길이
        self.ystart = ystart  # y축 시작점
        self.idx = 0
        self.time_sum = 0

        # 그래프 설정
        self.ax = ax
        self.ax.set_xlim((self.xstart, self.xmax))
        self.ax.set_ylim((self.ystart, self.ymax))
        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.set_xticks([])

        self.x = []  # x축 정보
        self.y = []  # y축 정보
        self.y_value = 0  # y축 값
        self.x_value = 0  # x축 값
        self.fn = fn
        self.line, = ax.plot([], [])

    # 그래프 설정
    def update(self, i):

        # 값 넣기
        self.x_value, self.y_value, self.time_sum = self.fn(self.idx, self.time_sum)  # y값 함수 불러오기
        self.y.append(self.y_value)  # y값 넣기
        self.x.append(self.x_value)  # x값 넣기
        self.line.set_data(self.x, self.y)

        self.idx += 1

        if self.x[-1] >= self.xstart + self.xmax:
            self.xstart = self.xstart + self.xmax / start_point
            self.ax.set_xlim(self.xstart, self.xstart + self.xmax)
            self.ax.figure.canvas.draw()

        return (self.line,)


fig, ax = plt.subplots(1, figsize=(5, 6))


def insert(idx, time_sum):
    y_value = data_list[idx]
    x_value = length[idx]

    if len(test_list) == start_point:
        del(test_list[0])
        test_list.append(y_value)
        time_sum += y_value
        features = feature_extraction.get_features(test_list)
        features = np.expand_dims(features, axis=0)

        test_acc = model.predict(features, verbose=0)[0][1]
        test_acc = round(test_acc * 100, 3)

        print("AF Probability : " + str(test_acc) + "%,    Current Time : " + str(int(time_sum)) + "(s)")

    else:
        test_list.append(y_value)
        time_sum += y_value

    return x_value, y_value, time_sum


scope = Scope(ax, insert, xmax=start_point, ymax=lst_max, xstart=0, ystart=lst_min)
ani = animation.FuncAnimation(fig, scope.update, interval=interval, blit=True)
plt.show()
