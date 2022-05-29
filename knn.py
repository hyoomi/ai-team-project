import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
warnings.filterwarnings('ignore')

class KNN:
    #  파라미터: data_type - 0 (x,y)관절 좌표 데이터 사용 / 1 (x,y,z)관절 좌표 데이터 사용 / else 원본 이미지 데이터 사용
    def __init__(self, data_type=0):
        self.data_type = data_type
        # Load data
        self.x_point2 = np.load('x_point2.npy')  # (x, y) 관절 좌표 데이터 로드
        self.y_point = np.load('y_point.npy')

        self.x_point3 = np.load('x_point3.npy')  # (x, y, z) 관절 좌표 데이터 로드

        self.x_load = np.load('x_load.npy') # 원본 이미지 데이터 로드
        self.y_load = np.load('y_load.npy')

        # Flatten data
        self.x_point2 = self.x_point2.reshape(1803, 21*2)
        self.x_point3 = self.x_point3.reshape(1803, 21*3)
        self.x_load = self.x_load.reshape(2059, 100*100*3)

        # Split data
        self.x_train2, self.x_valid2, self.y_train2, self.y_valid2 = \
            train_test_split(self.x_point2, self.y_point, test_size=0.2, shuffle=True, stratify=self.y_point, random_state=34)
        self.x_train3, self.x_valid3, self.y_train3, self.y_valid3 = \
            train_test_split(self.x_point3, self.y_point, test_size=0.2, shuffle=True, stratify=self.y_point, random_state=34)
        self.x_train, self.x_valid, self.y_train, self.y_valid = \
            train_test_split(self.x_load, self.y_load, test_size=0.2, shuffle=True, stratify=self.y_load, random_state=34)

        # KNN Model
        k_list = range(3, 40, 2)
        accuracies = []
        label = ' '
        for k in k_list:
            classifier = KNeighborsClassifier(n_neighbors=k)  # KNN 모델
            if data_type == 0:
                classifier.fit(self.x_train2, self.y_train2)  # KNN 모델 학습
                accuracies.append(classifier.score(self.x_valid2, self.y_valid2))  # KNN 모델 평가
                label = '(x,y)'
            elif data_type == 1:
                classifier.fit(self.x_train3, self.y_train3)
                accuracies.append(classifier.score(self.x_valid3, self.y_valid3))
                label = '(x,y,z)'
            else:
                classifier.fit(self.x_load, self.y_load)
                accuracies.append(classifier.score(self.x_valid, self.y_valid))
                label = 'original img'
        # K 변화에 따른 정확도 그래프
        plt.plot(k_list, accuracies, marker='o', linewidth=2, label=label)
        plt.xlabel("k")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title("KNN Accuracy")
        plt.show()


