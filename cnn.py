from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
import os
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("===== CNN =====")

# Load data ==================================================
x_load = np.load('C:/Users/gyals/PycharmProjects/team/x_load.npy')
y_load = np.load('C:/Users/gyals/PycharmProjects/team/y_load.npy')
x_sample = np.load('C:/Users/gyals/PycharmProjects/team/x_sample.npy')
y_sample = np.load('C:/Users/gyals/PycharmProjects/team/y_sample.npy')

# Print data images
# for i in range(y_sample.size):
#     plt.title(y_sample[i])
#     plt.imshow(x_sample[i])
#     plt.show()

# Preprocessing ================================================
# Split data (1853, 100, 100, 3) / (206, 100, 100, 3)
x_train, x_valid, y_train, y_valid = train_test_split(x_load, y_load, test_size=0.1, shuffle=True, random_state=34)

# Normalize data
x_train = x_train.astype('float32') / 255.
x_valid = x_valid.astype('float32') / 255.
x_sample = x_sample.astype('float32') / 255.

print('x_train shape:', x_train.shape)
print('x_valid shape:', x_valid.shape)
print('x_sample shape:', x_sample.shape)

# Analyze data ===============================================
# num_y_train = np.zeros(10, dtype=int)
# for idx, key in enumerate(y_train):
#     num_y_train[key] += 1
#
# num_y_test = np.zeros(10, dtype=int)
# for idx, key in enumerate(y_valid):
#     num_y_test[key] += 1
#
# # Visualize analyzed data
# y_value = np.arange(10)
# plt.plot(y_value, num_y_train, marker='o', linewidth= 2, label='y_train')
# plt.plot(y_value, num_y_test, marker='o', linewidth= 2, label='y_test')
# plt.xlabel("label")
# plt.ylabel("number")
# plt.xticks(y_value)
# plt.legend()
# plt.title("number of each label")
# plt.show()

# CNN ===================================================
# [Note]
# 합성곱 연산: input data에 filter를 적용한 것
# 패딩: 합성곱 연산 수행 전 데이터 주변을 특정값으로 채워 늘리는 것
# stride: input data에 filter를 적용하는 위치의 간격

# Create Model
model = keras.Sequential()
model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(keras.layers.MaxPooling2D(2, 2))
model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D(2, 2))
model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

# Model train
model.summary()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_sample, y_sample))



# Evaluate model
loss, accuracy = model.evaluate(x_valid, y_valid)
print("loss = ", loss)
print("accuracy = ", accuracy)

# predict = model.predict(x_sample, batch_size=32)
# print(predict)

# Visualize accuracy and loss
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# Save model and history
model.save('cnn1_model.h5')
np.save('cnn1_history.npy', history.history)