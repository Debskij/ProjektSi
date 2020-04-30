import math
import random
import matplotlib.pyplot as plt

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

values, anomalies = singen(4, 1000, 0.01)
plt.plot(values)
plt.show()
