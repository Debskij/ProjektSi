import math
import random
import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np

def singen(mnoznik, wielkosc_datasetu, czestosc_anomali):
    values = []
    anomalies = []
    while len(values) < int(wielkosc_datasetu):
        multip = random.uniform(float(1/mnoznik), float(mnoznik))
        step = 1 / 50 * math.pi * multip
        iter_per_sin = 2 * math.pi / step
        itx = 0
        while itx < iter_per_sin:
            values.append(math.sin(itx * step))
            itx += 1
            if random.random() < float(czestosc_anomali):
                anomalies.append(len(values))
                break
            if random.random() > 1-float(czestosc_anomali):
                values[-1] = values[-1] * random.uniform(float(1/mnoznik), float(mnoznik))
    return [values, anomalies]

x_train = []
y_train = []
values, anomalies = singen(4, 100000, 0.01)

TIME_STEPS = 30
max_value = max(values)
min_value = min(values)

for i in range(len(values) - TIME_STEPS):
    x_list = []
    for x1 in values[i:i + TIME_STEPS]:
        x_list.append([(x1 - min_value) / (max_value - min_value)])
    x_train.append(x_list)
    y_train.append([(values[i + TIME_STEPS] - min_value) / (max_value - min_value)])

x_test = x_train[0:10000]
x_train = x_train[10000:]
y_test = y_train[0:10000]
y_train = y_train[10000:]
x_train = np.array(x_train)
# print(len(x_train[0][0]))
# len(x_train[0])
# close_value[0:TIME_STEPS]


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
    epochs=15,
    batch_size=32,
    validation_split=0.1,
    shuffle=False
)

model.save("sinmodel.h5")
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
