import cv2
import mediapipe as mp
import sys
import pyautogui as pg
import os
import datetime
from keras.models import load_model
import time
import numpy as np

canvas = np.zeros((480, 640, 3), np.uint8)
col = [0, 0, 255]  # 기본 색상(빨강)
xp, yp = 0, 0

def save_canvas():
    global canvas
    timestamp = int(time.time())
    filename = f'static/canvas_{timestamp}.png'
    cv2.imwrite(filename, canvas)
    return f"Canvas saved as {filename}"

# 처음꺼
# def web_gen():
#     # 웹캠 비디오 캡처 초기화
#     cap = cv2.VideoCapture(0)
#     global canvas, xp, yp, col
#     cap.set(3, 640)  # 너비 설정
#     cap.set(4, 1000)  # 높이 설정
#     cap.set(10, 150)  # 밝기 설정
#
#     # 손 추적을 위한 mediapipe Hands 객체 초기화
#     mpHands = mp.solutions.hands
#     hands = mpHands.Hands()
#     mpdraw = mp.solutions.drawing_utils
#
#     pasttime = 0
#
#     #색상 이미지가 저장된 폴더 정의
#     folder = 'colors'
#     mylist = os.listdir(folder)
#     overlist = []
#     #
#     # 폴더에서 이미지 로드 및 리스트에 추가
#     for i in mylist:
#         image = cv2.imread(f'{folder}/{i}')
#         print(image.shape)
#         overlist.append(image)
#     #
#     # # 초기 헤더 이미지 설정
#     header = overlist[0]
#
#     while True:
#         # 웹캠에서 프레임 읽고 수평으로 뒤집기
#         success, frame = cap.read()
#         frame = cv2.flip(frame, 1)
#
#         # 손 추적을 위해 프레임을 RGB 색상 공간으로 변환
#         img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#         # 프레임 처리하여 손의 랜드마크 감지
#         results = hands.process(img)
#         lanmark = []
#
#         if results.multi_hand_landmarks:
#             for hn in results.multi_hand_landmarks:
#                 for id, lm in enumerate(hn.landmark):
#                     h, w, c = frame.shape
#                     cx, cy = int(lm.x * w), int(lm.y * h)
#                     lanmark.append([id, cx, cy])
#                 mpdraw.draw_landmarks(frame, hn, mpHands.HAND_CONNECTIONS)
#
#         if len(lanmark) != 0:
#             # 손이 "선택 모드" 또는 "그리기 모드"에 있는지 확인
#             x1, y1 = lanmark[8][1], lanmark[8][2]
#             x2, y2 = lanmark[12][1], lanmark[12][2]
#
#             if lanmark[8][2] < lanmark[6][2] and lanmark[12][2] < lanmark[10][2]:
#                 xp, yp = 0, 0
#                 print('선택 모드')
#
#                 # 손 위치에 따라 선택된 색상 감지
#                 if y1 < 100:
#                     if 71 < x1 < 142:
#                         # header = overlist[7]
#                         col = (0, 0, 0)
#                     if 142 < x1 < 213:
#                         # header = overlist[6]
#                         col = (226, 43, 138)
#                     # ... 다른 색상 조건 ...
#
#                 # 선택된 색상을 나타내는 사각형 그리기
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), col, cv2.FILLED)
#
#             elif lanmark[8][2] < lanmark[6][2]:
#                 if xp == 0 and yp == 0:
#                     xp, yp = x1, y1
#
#                 # "그리기 모드"일 때 캔버스에 선 그리기
#                 if col == (0, 0, 0):
#                     cv2.line(frame, (xp, yp), (x1, y1), col, 10, cv2.FILLED)
#                     cv2.line(canvas, (xp, yp), (x1, y1), col, 10, cv2.FILLED)
#                 cv2.line(frame, (xp, yp), (x1, y1), col, 5, cv2.FILLED)
#                 cv2.line(canvas, (xp, yp), (x1, y1), col, 5, cv2.FILLED)
#                 print('그리기 모드')
#                 xp, yp = x1, y1
#
#         # 프레임과 캔버스를 합성하기 위해 캔버스 준비
#         imgGray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
#         _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
#         imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
#
#         # 비트 연산을 사용하여 프레임과 캔버스 합성
#         frame = cv2.bitwise_and(frame, imgInv)
#         frame = cv2.bitwise_or(frame, canvas)
#
#         # 프레임 위에 헤더(색상 선택) 추가
#         frame[0:100, 0:640] = header
#
#         # 프레임당 초당 프레임 수(FPS) 계산 및 표시
#         ctime = time.time()
#         fps = 1 / (ctime - pasttime)
#         pasttime = ctime
#
#         ret, buffer = cv2.imencode('.jpg', frame)
#         frame = buffer.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
#

def webcam_feed():
    global canvas, xp, yp, col
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)  # 너비 설정
    cap.set(4, 480)  # 높이 설정
    cap.set(10, 150)  # 밝기 설정

    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    mpdraw = mp.solutions.drawing_utils

    pasttime = 0

    folder = 'colors'
    mylist = os.listdir(folder)
    overlist = []

    for i in mylist:
        image = cv2.imread(f'{folder}/{i}')
        overlist.append(image)

    header = overlist[0]

    while True:
        success, frame = cap.read()
        frame = cv2.flip(frame, 1)

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img)
        lanmark = []

        if results.multi_hand_landmarks:
            for hn in results.multi_hand_landmarks:
                for id, lm in enumerate(hn.landmark):
                    h, w, c = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lanmark.append([id, cx, cy])
                mpdraw.draw_landmarks(frame, hn, mpHands.HAND_CONNECTIONS)

        if len(lanmark) != 0:
            x1, y1 = lanmark[8][1], lanmark[8][2]
            x2, y2 = lanmark[12][1], lanmark[12][2]

            if lanmark[8][2] < lanmark[6][2] and lanmark[12][2] < lanmark[10][2]:
                xp, yp = 0, 0
                if y1 < 100:
                    if 71 < x1 < 142:
                        col = (0, 0, 0)
                    if 142 < x1 < 213:
                        col = (226, 43, 138)
                    # 다른 색상 조건 추가

                cv2.rectangle(frame, (x1, y1), (x2, y2), col, cv2.FILLED)

            elif lanmark[8][2] < lanmark[6][2]:
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1

                if col == (0, 0, 0):
                    cv2.line(frame, (xp, yp), (x1, y1), col, 10, cv2.FILLED)
                    cv2.line(canvas, (xp, yp), (x1, y1), col, 10, cv2.FILLED)
                cv2.line(frame, (xp, yp), (x1, y1), col, 5, cv2.FILLED)
                cv2.line(canvas, (xp, yp), (x1, y1), col, 5, cv2.FILLED)
                xp, yp = x1, y1

        frame[0:100, 0:640] = header

        ctime = time.time()
        fps = 1 / (ctime - pasttime)
        pasttime = ctime

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def canvas_feed():
    global canvas
    while True:
        ret, buffer = cv2.imencode('.jpg', canvas)
        canvas_frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + canvas_frame + b'\r\n')