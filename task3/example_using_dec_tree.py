import numpy as np
from decision_tree import MyDecisionTree
import matplotlib.pyplot as plt

my_tree = MyDecisionTree(2, 1)

dataset1 = np.array([[2.771244718, 1.784783929],
                     [1.728571309, 1.169761413],
                     [3.678319846, 2.81281357],
                     [3.961043357, 2.61995032],
                     [2.999208922, 2.209014212],
                     [7.497545867, 3.162953546],
                     [9.00220326, 3.339047188],
                     [7.444542326, 0.476683375],
                     [10.12493903, 3.234550982],
                     [6.642287351, 3.319983761]])

fig, ax = plt.subplots()
ax.scatter(dataset1[:5, 0], dataset1[:5, 1], label='class 0')
ax.scatter(dataset1[5:, 0], dataset1[5:, 1], label='class 1')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Наши данные для обучения решающего дерева')
plt.show()

classes1 = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

dataset2 = np.array([[2.771244718, 1.784783929],
                     [6.642287351, 3.319983761]])
classes2 = np.array([0, 1])

print(my_tree.fit_data(dataset1, classes1))

print(my_tree.predict_data(np.array([[5, 2], [10, 3]])))
