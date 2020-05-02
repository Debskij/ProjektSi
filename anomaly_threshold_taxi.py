# %%

import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt
import statistics
from scipy.signal import lfilter

# %%

def signal_smoothing(n, a, signal):
    b = [1.0 / n] * n
    return lfilter(b, a, signal)


def gauss_smoothing(percentage, signal):
    return sum(sorted(signal[int(percentage * 0.01 * len(signal)):-int(percentage * 0.01 * len(signal))])) \
           / (len(signal) * (1 - 2 * percentage * 0.01))


def percent_treshold(test_mae_loss):
    pom = list(test_mae_loss)
    pom.sort()
    TRESHOLD = pom[int(len(pom) * 0.95)]
    return TRESHOLD[0]


def normalised_treshold(test_mae_loss):
    # pom = list(test_mae_loss)
    # pom = signal_smoothing(8, 3, pom)
    TRESHOLD = gauss_smoothing(5, test_mae_loss) * 2.5
    return TRESHOLD[0]


def stdev_treshold(test_mae_loss):
    test_mae_loss = test_mae_loss.reshape(len(test_mae_loss))
    return statistics.stdev(test_mae_loss)


# %%

file_name = "taxi"
model = load_model("taxi.h5")

# %%

values = pd.read_csv('nyc_taxi.csv')
rows = [x for x in values]
values = list(values[rows[1]])
x_train = []
y_train = []
TIME_STEPS = 30
max_value = max(values)
min_value = min(values)
for i in range(len(values) - TIME_STEPS):
    x_list = []
    for x1 in values[i:i + TIME_STEPS]:
        x_list.append([(x1 - min_value) / (max_value - min_value)])
    x_train.append(x_list)
    y_train.append([(values[i + TIME_STEPS] - min_value) / (max_value - min_value)])

starting_idx = int(len(values) / 10)
x_test = x_train[:starting_idx]
y_test = y_train[:starting_idx]



X_test_pred = model.predict(x_test)

test_mae_loss = np.mean(np.abs((X_test_pred - x_test)), axis=1)
test_mae_loss = [i / j for i, j in zip(test_mae_loss, values)]
test_mae_loss = signal_smoothing(20, 2, test_mae_loss)
label_x = list(np.arange(len(test_mae_loss)))
THRESHOLD = normalised_treshold(test_mae_loss)

# %%

test_score_df = pd.DataFrame(label_x)
test_score_df['loss'] = test_mae_loss
test_score_df['threshold'] = THRESHOLD
test_score_df['anomaly'] = test_score_df.loss > test_score_df.threshold

# %%

plt.plot(test_score_df.index, test_score_df.loss, label='loss')
plt.plot(test_score_df.index, test_score_df.threshold, label='threshold')
plt.xticks(rotation=25)
plt.legend()

# %%

anomalies = test_score_df[test_score_df.anomaly == True]
anomalies.head()

# %%

pred_plot = []
y_plot = []
for y1 in y_test:
    y_plot.append(y1[0] * (max_value - min_value) + min_value)

for ys in X_test_pred:
    pred_plot.append(ys[0] * (max_value - min_value) + min_value)


plt.figure(figsize=(20, 10))
plt.plot(pred_plot, '-rD', markevery=list(anomalies.index), markeredgewidth=0.02, markerfacecolor="green")
plt.plot(y_plot, '-bD', markevery=list(anomalies.index), markeredgewidth=0.02, markerfacecolor="cyan")
plt.savefig(f'{file_name}_normalised.png')

# %%


plt.figure(figsize=(20, 10))
plt.plot(y_plot, '-gD', markevery=list(anomalies.index), markeredgewidth=0.02, markerfacecolor="red")
plt.savefig(f'{file_name}_anomaly.png')

# %%

plt.figure(figsize=(20, 10))
plt.plot(pred_plot)
plt.plot(y_plot)
plt.savefig(f'{file_name}_difference.png')

# %%
