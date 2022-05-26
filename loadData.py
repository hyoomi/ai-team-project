import os
import numpy as np
import cv2
import mediapipe as mp

print("===== LoadData =====")

# Data
label = []  # 각 label별 data수를 저장
x_load = []  # (100,100,3) data가 저장
y_load = []  # class 저장 (0 ~ 9)

# Load data
for i in range(10):
    # set path
    path = f'C:/Users/gyals/PycharmProjects/Digit/{i}/'  # 경로 입력
    os.chdir(path)  # 경로 폴더로 들어감
    files = os.listdir(path)  # 폴더 안에 있는 파일의 이름을 가져옴
    print(files)
    # save data
    for file in files:
        f = cv2.imread(file, cv2.IMREAD_COLOR)  # 하나의 파일을 np형태로 읽는다 (옵션은 COLOR)
        if f.shape != (100, 100, 3):
            continue
        x_load.append(f)  # 읽은 파일 추가
        y_load.append(i)  # label 기록
    label.append(len(files))  # 각 label별 개수를 저장

# Save data
np.save('C:/Users/gyals/PycharmProjects/team/x_load.npy', x_load)
np.save('C:/Users/gyals/PycharmProjects/team/y_load.npy', y_load)


# First step is to initialize the Hands class an store it in a variable
mp_hands = mp.solutions.hands
# Now second step is to set the hands function which will hold the landmarks points
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3)
# Last step is to set up the drawing function of hands landmarks on the image
mp_drawing = mp.solutions.drawing_utils

# Data preprocessing
p_hands = []
y_point = np.array(1, dtype=int)  # 전처리 성공한 데이터의 label값
wrong_index = []  # 전처리 실패한 데이터의 index 값
wrong_label = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 전처리 실패한 label별 개수
for idx, x in enumerate(x_load):
    results = hands.process(cv2.cvtColor(x, cv2.COLOR_BGR2RGB))  # 관절 탐지
    p_hand = []
    if results.multi_hand_landmarks:  # 탐지 성공했다면
        for hand_no, hand_landmarks in enumerate(results.multi_hand_landmarks):
            for i in range(len(mp_hands.HandLandmark)):  # p: [x, y, z]
                p = []
                p.append(hand_landmarks.landmark[mp_hands.HandLandmark(i).value].x)
                p.append(hand_landmarks.landmark[mp_hands.HandLandmark(i).value].y)
                #p.append(hand_landmarks.landmark[mp_hands.HandLandmark(i).value].z)
                p_hand.append(p)  # p_hand: [[x, y, z] * 21]
        p_hands.append(p_hand)  # p_hands: [[[x, y, z] * 21] * 1803]
        y_point = np.append(y_point, y_load[idx])
    else:  # 실패 데이터 기록
        wrong_index.append(idx)
        wrong_label[y_load[idx]] += 1
x_point = np.array(p_hands)  # 전처리 성공한 데이터
y_point = np.delete(y_point, 0)  # 맨 앞에 1 넣어준 것 삭제(39째줄 때문)

# Save data
np.save('C:/Users/gyals/PycharmProjects/team/x_point2.npy', x_point)
np.save('C:/Users/gyals/PycharmProjects/team/y_point.npy', y_point)

# Print shape
print("shape of x_load", np.array(x_load).shape)  # load된 데이터의 shape
print("shape of x_point", x_point.shape)  # 관절을 뽑아낸 데이터의 shape
print("num of hand landmarks", len(mp_hands.HandLandmark))  # 관절 수
print("shape of y_point", y_point.shape) # 관절을 뽑아낸 데이터의 label shape
print("num of each label", label)  # 각 label별 개수
print("wrong index", wrong_index)  # 잘못된 데이터의 인덱스
print("wrong num of each label", wrong_label)  # 잘못된 라벨별 수