import math
import random
import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np

def singenAnomal(mnoznik, wielkosc_datasetu, czestosc_anomali):
    values = []
    anomalies = []
    while len(values) < int(wielkosc_datasetu):
        multip = random.uniform(float(1/mnoznik), float(mnoznik))
        step = 1 / 50 * math.pi * multip
        iter_per_sin = 2 * math.pi / step
        itx = 0
        while itx < iter_per_sin:
            values.append([ math.sin(itx * step) + 2])
            itx += 1
            if random.random() < float(czestosc_anomali):
                anomalies.append(len(values))
                break
            if random.random() > 1-float(czestosc_anomali):
                values[-1][0] = values[-1][0] * random.uniform(float(1/mnoznik), float(mnoznik))
    return values

def singen(mnoznik, wielkosc_datasetu):
    values = []
    while len(values) < int(wielkosc_datasetu):
        multip = random.uniform(float(1/mnoznik), float(mnoznik))
        step = 1 / 50 * math.pi * multip
        iter_per_sin = 2 * math.pi / step
        itx = 0
        while itx < iter_per_sin:
            values.append([math.sin(itx * step) + 2])
            itx += 1
    return values

if __name__ == '__main__':

    x_train = []
    y_train = []
    values = singen(4, 100000)

    TIME_STEPS = 30
    max_value = max([x[0] for x in values])
    min_value = min([x[0] for x in values])

    for i in range(len(values) - TIME_STEPS):
        x_list = []
        for x1 in values[i:i + TIME_STEPS]:
            x_list.append([(x1[0] - min_value) / (max_value - min_value)])
        x_train.append(x_list)
        y_train.append([(values[i + TIME_STEPS][0] - min_value) / (max_value - min_value)])

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
        epochs=5,
        batch_size=32,
        validation_split=0.1,
        shuffle=False
    )

    model.save("sinmodel.h5")
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
