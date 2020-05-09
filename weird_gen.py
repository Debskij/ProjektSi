import math
import random
import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np


def weird_gen(stopien_wielomianu: int, maksymalny_mnoznik: float, data_sample: int, okres: list):
    some_fun = []
    while len(some_fun) < data_sample * 2:
        polinomial = [maksymalny_mnoznik * (2.1 * random.random() - 1) for x in range(stopien_wielomianu)]
        period = (random.random() * (okres[1] - okres[0]) + okres[0]) * 10
        for i in range(int(period)):
            some_fun.append(abs(
                sum([val * i ** idx * math.pi / 10 for idx, val in enumerate(polinomial)]) * math.sin(i / 2 * math.pi)))
        if random.random() < 1:
            for j in range(int(period), 0, -1):
                some_fun.append(abs(
                    sum([val * j ** idx * math.pi / 10 for idx, val in enumerate(polinomial)]) * math.sin(
                        j / 2 * math.pi)))
    return [x for x in some_fun if x > 1e-5]


def weird_gen_anomaly(stopien_wielomianu: int, maksymalny_mnoznik: float, data_sample: int, okres: list):
    some_fun = []
    while len(some_fun) < data_sample * 2:
        noice = bool(random.random() < 0.3)
        polinomial = [maksymalny_mnoznik * (2.1 * random.random() - 1) for x in range(stopien_wielomianu)]
        period = (random.random() * (okres[1] - okres[0]) + okres[0]) * 10
        for i in range(int(period)):
            first = abs(
                sum([val * i ** idx * math.pi / 10 for idx, val in enumerate(polinomial)]) * math.sin(i / 2 * math.pi))
            if noice:
                some_fun.append((random.random() * 0.4 + 0.8) * first)
            else:
                some_fun.append(first)
        if random.random() < 0.8:
            for j in range(int(period), 0, -1):
                first = abs(sum([val * j ** idx * math.pi / 10 for idx, val in enumerate(polinomial)]) * math.sin(
                    j / 2 * math.pi))
                if noice:
                    some_fun.append((random.random() * 0.8 + 0.6) * first)
                else:
                    some_fun.append(first)
    return [x for x in some_fun if x > 1e-5]


# val = weird_gen(3, 4, 1000, [1, 20])
#
# plt.plot(val)
# plt.savefig('test.png')
if __name__ == '__main__':

    x_train = []
    y_train = []
    values = weird_gen_anomaly(3, 4, 200000, [5, 20])

    TIME_STEPS = 30
    max_value = max([x for x in values])
    min_value = min([x for x in values])

    for i in range(len(values) - TIME_STEPS):
        x_list = []
        for x1 in values[i:i + TIME_STEPS]:
            x_list.append([(x1 - min_value) / (max_value - min_value)])
        x_train.append(x_list)
        y_train.append([(values[i + TIME_STEPS] - min_value) / (max_value - min_value)])

    x_train = np.array(x_train)

    model = keras.Sequential()
    model.add(keras.layers.LSTM(
        units=64,
        input_shape=(x_train.shape[1], x_train.shape[2])
    ))
    model.add(keras.layers.Dropout(rate=0.2))
    model.add(keras.layers.RepeatVector(n=x_train.shape[1]))
    model.add(keras.layers.LSTM(units=32, return_sequences=True))
    model.add(keras.layers.Dropout(rate=0.2))
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(units=x_train.shape[2])))
    model.compile(loss='mae', optimizer='adam')
    model.summary()

    history = model.fit(
        x_train, np.array(y_train),
        epochs=10,
        batch_size=32,
        validation_split=0.1,
        shuffle=False
    )

    model.save("funmodel.h5")
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
