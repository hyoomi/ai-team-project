from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import warnings
import os
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("===== NN =====")

# Load data
x_load = np.load('C:/Users/gyals/PycharmProjects/team/x_load.npy')
y_load = np.load('C:/Users/gyals/PycharmProjects/team/y_load.npy')

# Split data
x_train, x_valid, y_train, y_valid = train_test_split(x_load, y_load, test_size=0.1, shuffle=True, random_state=34)
# (1853, 100, 100, 3) / (206, 100, 100, 3)

# 데이터 전처리
x_train = x_train.reshape(1853,30000).astype('float32')/255.0
x_valid = x_valid.reshape(206,30000).astype('float32')/255.0

# 모델 구성
model = keras.Sequential()
model.add(keras.layers.Dense(units = 64, input_dim= 30000, activation = 'relu'))
model.add(keras.layers.Dense(units = 10, activation = 'softmax'))

# 모델 학습과정 설정
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'sgd', metrics = ['accuracy'])

# 모델 학습 batch_size : 몇개의 샘플로 가중치를 갱신할 것인가 ?
hist = model.fit(x_train, y_train, epochs = 30, batch_size = 32, verbose=1)

