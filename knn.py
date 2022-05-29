import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
warnings.filterwarnings('ignore')

print("===== KNN =====")

# Load data
x_point2 = np.load('C:/Users/gyals/PycharmProjects/team/x_point2.npy')
y_point = np.load('C:/Users/gyals/PycharmProjects/team/y_point.npy')

x_load = np.load('C:/Users/gyals/PycharmProjects/team/x_load.npy')
y_load = np.load('C:/Users/gyals/PycharmProjects/team/y_load.npy')

x_sample = np.load('C:/Users/gyals/PycharmProjects/team/x_sample.npy')
y_sample = np.load('C:/Users/gyals/PycharmProjects/team/y_sample.npy')

# Flatten data
x_point2 = x_point2.reshape(1803, 21*2)
x_load = x_load.reshape(2059, 100*100*3)
x_sample = x_sample.reshape(150, 100*100*3)


# Split data
x_train2, x_valid2, y_train2, y_valid2 = \
    train_test_split(x_point2, y_point, test_size=0.2, shuffle=True, stratify=y_point, random_state=34)
x_train, x_valid, y_train, y_valid = \
    train_test_split(x_load, y_load, test_size=0.2, shuffle=True, stratify=y_load, random_state=34)

# KNN
# k_list = range(1, 40, 2)
# accuracies = []
# for k in k_list:
#   classifier = KNeighborsClassifier(n_neighbors = k)
#   classifier.fit(x_train2, y_train2)
#   accuracies.append(classifier.score(x_valid2, y_valid2))
# plt.plot(k_list, accuracies, marker='o', linewidth=2, label='point')

# k_list = range(1, 40, 2)
# accuracies = []
# for k in k_list:
#   classifier = KNeighborsClassifier(n_neighbors = k)
#   classifier.fit(x_load, y_load)
#   accuracies.append(classifier.score(x_sample, y_sample))
# plt.plot(k_list, accuracies, marker='o', linewidth=2, label='sample img')

k_list = range(1, 40, 2)
accuracies = []
for k in k_list:
  classifier = KNeighborsClassifier(n_neighbors = k)
  classifier.fit(x_train, y_train)
  accuracies.append(classifier.score(x_valid, y_valid))
plt.plot(k_list, accuracies, marker='o', linewidth=2, label='original img')

plt.xlabel("k")
plt.ylabel("Accuracy")
plt.legend()
plt.title("KNN Accuracy")
plt.show()