# %%

import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt
import statistics
from scipy.signal import lfilter
from singen import singenAnomal
from singen import singen
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
    return TRESHOLD


def stdev_treshold(test_mae_loss):
    test_mae_loss = test_mae_loss.reshape(len(test_mae_loss))
    return statistics.stdev(test_mae_loss)


# %%

file_name = "sin"
model = load_model("sinmodel.h5")
TIME_STEPS = 30

# %%

data = singenAnomal(4, 1000,0.01)
x_test = []
y_test = []
pom = 0

max_value = max(data)[0]
min_value = min(data)[0]

for i in range(len(data) - TIME_STEPS):
    x_list = []
    for x1 in data[i:i + TIME_STEPS]:
        x_list.append([(x1[0] - min_value) / (max_value - min_value)])
    x_test.append([value for value in x_list if value])
    y_test.append([(data[i + TIME_STEPS][0] - min_value) / (max_value - min_value)])

# %%

X_test_pred = model.predict(x_test)

test_mae_loss = np.mean(np.abs((X_test_pred - x_test)), axis=1)
test_mae_loss = [i / j for i, j in zip(test_mae_loss, data)]
test_mae_loss = signal_smoothing(20, 2, test_mae_loss)
label_x = list(np.arange(len(test_mae_loss)))
THRESHOLD = normalised_treshold(test_mae_loss)
# %%

test_score_df = pd.DataFrame(label_x)
test_score_df['loss'] = test_mae_loss
test_score_df['threshold'] = [THRESHOLD] *len(test_mae_loss)
test_score_df['anomaly'] = test_score_df.loss > test_score_df.threshold

# %%

plt.plot(test_score_df.index, test_score_df.loss, label='loss')
plt.plot(test_score_df.index, test_score_df.threshold, label='threshold')
plt.xticks(rotation=25)
plt.legend();
plt.savefig(f'{file_name}_treshold.png')
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

for i in range(len(x_test)):
    if i not in anomalies.index:
        pred_plot[i] = (y_test[i][0] * (max_value - min_value)) + min_value
    else:
        pred_plot[i] = (pred_plot[i] + data[i] + data[i]) / 3

plt.figure(figsize=(20, 10))
plt.plot(pred_plot, '-rD', markevery=list(anomalies.index), markeredgewidth=0.02, markerfacecolor="green")
plt.plot(y_plot, '-bD', markevery=list(anomalies.index), markeredgewidth=0.02, markerfacecolor="cyan")
plt.savefig(f'{file_name}_normalised.png')

plt.figure(figsize=(20, 10))
plt.plot(y_plot, '-gD', markevery=list(anomalies.index), markeredgewidth=0.02, markerfacecolor="red")
plt.savefig(f'{file_name}_anomaly.png')

# %%

plt.figure(figsize=(20, 10))
plt.plot(pred_plot)
plt.plot(y_plot)



