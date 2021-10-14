import numpy as np
import torch as T
from log_regression import MyOwnLogisticRegression


device = 'cpu'

# Модельные данные для обучения
train_x = np.array([
    [-10, -11], [-10, -9], [-9, -10], [-9, -11], [-10, -10],
    [10, 11], [10, 9], [9, 10], [9, 11], [10, 10]], dtype=np.float32)
train_y = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=np.long)

# тестовые данные для обучения
test_x = np.array([[10, 11], [10, 9]])

train_x = T.tensor(train_x, dtype=T.float32).to(device)
train_y = T.tensor(train_y, dtype=T.long).to(device)

test_x = T.tensor(test_x, dtype=T.float32).to(device)

times = 30

num_of_features = train_x.size()[1]
log_regression = MyOwnLogisticRegression(num_of_features)

log_regression.fit(times, train_x, train_y)

# Вывожу на экран предикт
print(log_regression.predictions(test_x))