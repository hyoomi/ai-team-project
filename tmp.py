import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

label = []  # 각 label별 data수를 저장
x_load = []  # (100,100,3) data가 저장
y_load = []  # class 저장 (0 ~ 9)

# 세영님은 아래처럼 경로 지정
# C:/Users/seyou/Downloads/Sign-Language-Digits-Dataset-master/Sign-Language-Digits-Dataset-master/Dataset/
for i in range(10):
    path = f'C:/Users/gyals/PycharmProjects/Digit/{i}/'  # 경로 입력
    os.chdir(path)  # 경로 폴더로 들어감
    files = os.listdir(path)  # 폴더 안에 있는 파일의 이름을 가져옴

    label.append(len(files))  # 각 label별 개수를 저장

    # 데이터 저장하기
    for file in files:
        f = cv2.imread(file, cv2.IMREAD_COLOR)  # 하나의 파일을 np형태로 읽는다 (옵션은 COLOR)
        x_load.append(f)  # 읽은 파일 추가
        y_load.append(i)  # label 기록

# [테스트] 각 class별로 한개씩 데이터 이미지 출력
tmp = 0
for idx, key in enumerate(label):
    plt.title(y_load[tmp])
    plt.imshow(x_load[tmp])
    plt.show()
    tmp += key

print(len(x_load))

# img = cv2.imread(files[0], cv2.IMREAD_COLOR)
# print(img.shape)
# print(img)
# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

