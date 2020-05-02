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

file_name = "PATK"
model = load_model("1000files_64.h5")
TIME_STEPS = 30

# %%

data = open(f"data1/{file_name}.csv")
close_value = []
x_test = []
y_test = []
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
    x_test.append([value for value in x_list if value])
    y_test.append([(close_value[i + TIME_STEPS][0] - min_value) / (max_value - min_value)])

# %%

X_test_pred = model.predict(x_test)

test_mae_loss = np.mean(np.abs((X_test_pred - x_test)), axis=1)
test_mae_loss = [i / j for i, j in zip(test_mae_loss, close_value)]
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
plt.legend();

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
plt.plot(y_plot, '-bD', markevery=list(anomalies.index), markeredgewidth=0.02, markerfacecolor="cyan")
plt.plot(pred_plot, '-rD', markevery=list(anomalies.index), markeredgewidth=0.02, markerfacecolor="green")
plt.savefig(f'{file_name}_normalised.png')

# %%


plt.figure(figsize=(20, 10))
plt.plot(y_plot, '-gD', markevery=list(anomalies.index), markeredgewidth=0.02, markerfacecolor="red")
plt.savefig(f'{file_name}_anomaly.png')

# %%

plt.figure(figsize=(20, 10))
plt.plot(pred_plot)
plt.plot(y_plot)
plt.savefig(f'{file_name}_diffrence.png')
# %%
