# !nvidia - smi
#
# !pip
# install
# gdown
# !pip
# install
# tensorflow - gpu

import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from pandas.plotting import register_matplotlib_converters

# % matplotlib
# inline
# % config
# InlineBackend.figure_format = 'retina'

register_matplotlib_converters()
sns.set(style='whitegrid', palette='muted', font_scale=1.5)

rcParams['figure.figsize'] = 22, 10

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

import os

len(os.listdir("data1/"))
# for file in os.listdir("data1/"):
#   print(file)

import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt

import os

x_train = []
y_train = []
TIME_STEPS = 30
count = 0
for file in os.listdir("data1/"):
    print(file)
    if count < 1000:
        count += 1
    else:
        break
    data = open(f"data1/{file}")
    close_value = []
    pom = 0
    for line in data:

        if pom == 0 or line == '\n':
            pom = 1
            continue
        line = line.replace('\n', '')
        close_value.append([float(line.split(',')[2])])

    max_value = max(close_value)[0]
    min_value = min(close_value)[0]

    for i in range(len(close_value) - TIME_STEPS):
        x_list = []
        for x1 in close_value[i:i + TIME_STEPS]:
            x_list.append([(x1[0] - min_value) / (max_value - min_value)])
        x_train.append(x_list)
        y_train.append([(close_value[i + TIME_STEPS][0] - min_value) / (max_value - min_value)])

x_test = x_train[0:1000]
x_train = x_train[1000:]
y_test = y_train[0:1000]
y_train = y_train[1000:]
x_train = np.array(x_train)
# print(len(x_train[0][0]))
# len(x_train[0])
x_train
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

# model.save("1000files_64.h5")
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()

