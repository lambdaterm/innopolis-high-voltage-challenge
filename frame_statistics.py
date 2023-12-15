import copy
import time
import numpy as np
import math
import matplotlib.pyplot as plt
import cv2
from typing import Tuple, Union
from typing import Tuple

ModelResult = Tuple[int, str, float]


if __name__ == '__main__':

    ###
    print('Models initialization ...')
    ###
    # Забираем видео из видео потока
    cap = cv2.VideoCapture('IMG_3728.MOV')
    # Забираем видео с мобильного телефона
    # cap = cv2.VideoCapture('http://172.16.74.154:8080/video', cv2.CAP_ANY)
    # cap.set(3, 900)
    # cap.set(4, 900)
    # Забираем видео с web камеры
    # web_cam_screen_size = (800, 600)
    # cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # cap.set(3, web_cam_screen_size[0])
    # cap.set(4, web_cam_screen_size[0])
    ###
    frames = 0
    fps = 0
    frame_time = time.time()
    frame_time_in_sec = 0
    while cap.isOpened():

        ###
        frame_time = time.time() - frame_time
        frame_time_in_sec = frame_time_in_sec + frame_time
        frames += 1
        if frame_time_in_sec > 1:
            frame_time_in_sec = 0
            fps = frames
            frames = 0
        if fps != 0:
            print(f'fps = {fps} Frame performing time = {str(int(frame_time * 1000))} ms')
        frame_time = time.time()

        ###
        ret, frame = cap.read()
        original_frame = frame.copy()
        print(f'Frame_size = {frame.shape}')
        # frame = cv2.resize(frame, (900, 900))

        annotated_image = frame
        cv2.imshow("Camera", annotated_image)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()
