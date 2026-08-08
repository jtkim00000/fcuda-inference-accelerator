import matplotlib.pyplot as plt
import numpy as np
import math

x = [
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512
]

cpu_fp = [
    3.60554,
    7.15396,
    14.2533,
    28.479,
    56.917,
    113.984,
    227.214,
    452.931,
    905.066,
    1812.73
]
cpu_tr = [
    7.49775,
    11.5388,
    19.3284,
    34.9724,
    66.0001,
    118.096,
    231.759,
    459.377,
    914.755,
    1831.53
]

gpu_fp = [
    10.9104,
    11.0716,
    10.9357,
    11.3285,
    11.621,
    20.9189,
    20.8712,
    21.3788,
    22.2217,
    24.6953,
]
gpu_tr = [
    20.7304,
    21.0016,
    20.888,
    21.2163,
    21.3183,
    32.1786,
    32.2172,
    34.143,
    37.3844,
    45.6336
]

speedup_fp = []
speedup_tr = []

x_label = "# of Inferences Running in Parallel (log2(X))"
y_label = "Runtime Speedup"
plot_title = "CUDA Forward Propagation Accelerator Speedup"

#calculate the throughput
for i in range(len(cpu_fp)):
    speedup_fp.append(cpu_fp[i] / gpu_fp[i])
    speedup_tr.append(cpu_tr[i] / gpu_tr[i])

# plot as logarithmic scale
for i in range(len(x)):
    x[i] = math.log2(x[i])

plt.plot(x, speedup_fp, marker='o', color='#1f77b4', linestyle='-', linewidth=2, markersize=8, label='Forward Propagation Speedup')
plt.plot(x, speedup_tr, marker='o', color='#ff7f0e', linestyle='-', linewidth=2, markersize=8, label='Total Runtime Speedup')

# Formatting
plt.xlabel(x_label, fontsize=12)
plt.ylabel(y_label, fontsize=12)
plt.title(plot_title, fontsize=14, pad=12)
plt.legend(loc='best', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)

# Open interactive viewer
plt.show()