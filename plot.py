import numpy as np
import matplotlib.pyplot as plt

trace = np.load("dataset_v3_50k/test/x_00001.npy")
heatmap = np.load("dataset_v3_50k/test/heatmap_00001.npy")

plt.figure(figsize=(10,4))
plt.plot(trace, label="Trace")
plt.plot(heatmap, label="Heatmap")
plt.legend()
plt.show()