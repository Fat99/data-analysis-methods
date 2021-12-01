import numpy as np
import matplotlib.pyplot as plt


data_to_numpy = np.genfromtxt('custom_dataset.csv',delimiter=',')

size_1, size_2 = np.shape(data_to_numpy)
clear_data = np.zeros((size_1 - 1, size_2 - 1))
clear_data[:, :]  = data_to_numpy[1:, 1:]

colors = {0:'red', 1:'green'}

fig, ax = plt.subplots(1, 1, figsize=(15, 8))

for i in range(size_1 - 1):
    if clear_data[i, 2] == 0:
        ax.scatter(clear_data[i, 0], clear_data[i, 1], c=colors[0])
    else:
        ax.scatter(clear_data[i, 0], clear_data[i, 1], c=colors[1])

plt.show()


