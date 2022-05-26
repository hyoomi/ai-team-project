import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
warnings.filterwarnings('ignore')

print("===== KNN =====")

# Load data
x_point = np.load('C:/Users/gyals/PycharmProjects/team/x_point2.npy')
y_point = np.load('C:/Users/gyals/PycharmProjects/team/y_point.npy')

# Flatten data
x_point = x_point.reshape(1803, 21*2)

# Split data
x_train, x_valid, y_train, y_valid = \
    train_test_split(x_point, y_point, test_size=0.2, shuffle=True, stratify=y_point, random_state=34)

# Analyze data
num_y_train = np.zeros(10, dtype=int)
for idx, key in enumerate(y_train):
    num_y_train[key] += 1

num_y_test = np.zeros(10, dtype=int)
for idx, key in enumerate(y_valid):
    num_y_test[key] += 1

# Visualize analyzed data
y_value = np.arange(10)
plt.plot(y_value, num_y_train, marker='o', linewidth= 2, label='y_train')
plt.plot(y_value, num_y_test, marker='o', linewidth= 2, label='y_test')
plt.xlabel("label")
plt.ylabel("number")
plt.xticks(y_value)
plt.legend()
plt.title("number of each label")
plt.show()

# KNN
k_list = range(1, 25, 2)
accuracies = []
for k in k_list:
  classifier = KNeighborsClassifier(n_neighbors = k)
  classifier.fit(x_train, y_train)
  accuracies.append(classifier.score(x_valid, y_valid))
plt.plot(k_list, accuracies)
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.title("KNN Accuracy")
plt.show()