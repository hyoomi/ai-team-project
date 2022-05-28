from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
import os
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("===== CNN =====")

# Load data
x_load = np.load('C:/Users/gyals/PycharmProjects/team/x_load.npy')
y_load = np.load('C:/Users/gyals/PycharmProjects/team/y_load.npy')

# Print data images
# for i in range(y_load.size):
#     plt.title(y_load[i])
#     plt.imshow(x_load[i])
#     plt.show()

# Split data
x_train, x_valid, y_train, y_valid = train_test_split(x_load, y_load, test_size=0.1, shuffle=True, random_state=34)
# (1853, 100, 100, 3) / (206, 100, 100, 3)

# Normalize data
x_train = x_train.astype('float32') / 255.
x_valid = x_valid.astype('float32') / 255.

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_valid.shape[0], 'test samples')

# [Note]
# 합성곱 연산: input data에 filter를 적용한 것
# 패딩: 합성곱 연산 수행 전 데이터 주변을 특정값으로 채워 늘리는 것
# stride: input data에 filter를 적용하는 위치의 간격

# Create Model
model = keras.Sequential()
model.add(keras.layers.Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=(100, 100, 3)))
model.add(keras.layers.MaxPooling2D(2, 2))
model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Conv2D(64, kernel_size=(5, 5), activation='relu'))
model.add(keras.layers.MaxPooling2D(2, 2))
model.add(keras.layers.Conv2D(64, kernel_size=(5, 5), activation='relu'))
model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

# Model train
model.summary()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=10)

# Evaluate model
loss, accuracy = model.evaluate(x_valid, y_valid)
print("loss = ", loss)
print("accuracy = ", accuracy)

# Visualize accuracy and loss
plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



