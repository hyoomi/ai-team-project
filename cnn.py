from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
import os

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class CNN:
    # 파라미터: conv2D layer에 적용할 kernel_size, 첫번째 Dense layer의 출력 뉴런 수, 에포크 수, 직접 촬영한 데이터 사용 유무
    def __init__(self, conv_kernel_size=3, dense_units=256, epoch=10, use_sample=True):
        self.use_sample = use_sample
        # Load data: x_load (2059, 100, 100, 3) / x_sample (150, 100, 100, 3)
        self.x_load = np.load('C:/Users/gyals/PycharmProjects/team/x_load.npy')  # 기존 이미지 셋
        self.y_load = np.load('C:/Users/gyals/PycharmProjects/team/y_load.npy')
        self.x_sample = np.load('C:/Users/gyals/PycharmProjects/team/x_sample.npy')  # 직접 촬영한 이미지 셋
        self.y_sample = np.load('C:/Users/gyals/PycharmProjects/team/y_sample.npy')

        # Split data: x_train (1853, 100, 100, 3) / x_valid(206, 100, 100, 3)
        # load data를 split하여 trian data와 valid data 정의
        self.x_train, self.x_valid, self.y_train, self.y_valid = \
            train_test_split(self.x_load, self.y_load, test_size=0.1, shuffle=True, random_state=34)

        # Normalize data
        self.x_train = self.x_train.astype('float32') / 255.
        self.x_valid = self.x_valid.astype('float32') / 255.
        self.x_sample = self.x_sample.astype('float32') / 255.

        # Create Model
        model = keras.Sequential()  # model layer 쌓기
        model.add(keras.layers.Conv2D(32, kernel_size=(conv_kernel_size, conv_kernel_size), activation='relu',
                                      input_shape=(100, 100, 3)))  # out_channels 수 = 32
        model.add(keras.layers.MaxPooling2D(2, 2))  # 절반으로 축소하는 layer
        model.add(keras.layers.Dropout(0.25))

        # convolution layer의 out_channels 수 = 64
        model.add(keras.layers.Conv2D(64, kernel_size=(conv_kernel_size, conv_kernel_size), activation='relu'))
        model.add(keras.layers.MaxPooling2D(2, 2))
        model.add(keras.layers.Dropout(0.25))

        model.add(keras.layers.Flatten())  # Dense적용 위해 차원 flatten
        model.add(keras.layers.Dense(dense_units, activation='relu'))  # 출력뉴런수 = dense_units
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(10, activation='softmax'))  # 출력층. label=10, 활성화 함수=softmax

        if use_sample:
            # Train Model: 기존 데이터 셋 10%를 validation 데이터 셋으로 사용하여 학습
            model.summary()  # layer와 output shape, param # 정보 확인
            # 원샷인코딩 안했기 때문에 spares_categorical_crossentropy사용
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            self.his = model.fit(self.x_train, self.y_train, epochs=epoch, validation_data=(self.x_valid, self.y_valid))

            # Evaluate model: 촬영한 데이터로 모델 평가
            loss, accuracy = model.evaluate(self.x_sample, self.y_sample)
            print("Sample Data: ", "loss = ", loss, ", accuracy = ", accuracy)
        else:
            # Train Model: 기존 데이터 셋의 90%를 학습
            model.summary()
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            self.his = model.fit(self.x_train, self.y_train, epochs=epoch)

            # Evaluate model: 기존 데이터 셋의 10%센트로 모델 평가
            loss, accuracy = model.evaluate(self.x_valid, self.y_valid)
            print("Test Data: ", "loss = ", loss, ", accuracy = ", accuracy)

    def show_all_load_images(self):
        # Show x_load image
        for i in range(self.x_load.size):
            plt.title(self.y_load[i])
            plt.imshow(self.x_load[i])
            plt.show()

    def show_all_sample_images(self):
        # Show x_sample image
        for i in range(self.x_sample.size):
            plt.title(self.y_sample[i])
            plt.imshow(self.x_sample[i])
            plt.show()

    def visualize_label_distribution(self):
        # Analyze label distribution
        num_y_train = np.zeros(10, dtype=int)
        for idx, key in enumerate(self.y_train):
            num_y_train[key] += 1
        num_y_test = np.zeros(10, dtype=int)
        for idx, key in enumerate(self.y_valid):
            num_y_test[key] += 1

        # Visualize analyzed data
        y_value = np.arange(10)
        plt.plot(y_value, num_y_train, marker='o', linewidth=2, label='y_train')
        plt.plot(y_value, num_y_test, marker='o', linewidth=2, label='y_test')
        plt.title("number of each label")
        plt.xlabel("label")
        plt.ylabel("number")
        plt.xticks(y_value)
        plt.legend()
        plt.show()

    def visualize_accuracy_according_to_epoch(self):
        # Visualize accuracy: 에포크에 따른 정확도 추이
        plt.plot(self.his.history['accuracy'], label='train')
        if self.use_sample:
            plt.plot(self.his.history['val_accuracy'], label='test')
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend()
        plt.show()

    def visualize_loss_according_to_epoch(self):
        # Visualize loss: 에포크에 따른 loss 추이
        plt.plot(self.his.history['loss'], label='train')
        if self.use_sample:
            plt.plot(self.his.history['val_loss'], label='test')
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend()
        plt.show()
