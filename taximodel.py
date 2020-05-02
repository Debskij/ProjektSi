import math
import random
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np

if __name__ == '__main__':

    x_train = []
    y_train = []
    values = pd.read_csv('nyc_taxi.csv')
    rows = [x for x in values]
    values = list(values[rows[1]])

    TIME_STEPS = 30
    max_value = max(values)
    min_value = min(values)

    for i in range(len(values) - TIME_STEPS):
        x_list = []
        for x1 in values[i:i + TIME_STEPS]:
            x_list.append([(x1 - min_value) / (max_value - min_value)])
        x_train.append(x_list)
        y_train.append([(values[i + TIME_STEPS] - min_value) / (max_value - min_value)])

    starting_idx = int(len(values)/10)
    x_test = x_train[:starting_idx]
    x_train = x_train[starting_idx:]
    y_test = y_train[:starting_idx]
    y_train = y_train[starting_idx:]
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

    model.save("taxi.h5")
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
