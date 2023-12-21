import copy
import time
import numpy as np
import math
import matplotlib.pyplot as plt
import cv2
from typing import Tuple, Union
from typing import Tuple
from pathlib import Path
import sys
from Libs.pipeline import Pipeline

ModelResult = Tuple[int, str, float]

OUTPUT_DIR = r'D:\Data\Result'
IMG_PATH = Path(r'data\InnopolisTestImages\DJI_0032.JPG')

if __name__ == '__main__':

    pipeline = Pipeline(
        golden_model_path=Path(r"Models\insulator_gold.pt"),
        base_model_path=Path(r"Models\insulator_base.pt"),
        broken_model_path=Path(
            r"Models\insulator_broken_gold.pt"),
        # broken_model_path=Path(r"D:\ML\ResultModels\Insulators\InsulatorModel\yolo_broken_insulators_m_gold\train5\weights\best.pt"),
        conf_supreme=0.5,
        iou_supreme=0.5,
        conf_broken=0.5,
        iou_broken=0.1,
    )

    # insulator, broken = pipeline.predict(
    #     IMG_PATH,
    #     img_sizes_insulators=(1500, 2500),
    #     img_sizes_broken=(640, 960),
    # )

    ###
    print('Models initialization ...')
    ###
    # Забираем видео из видео потока
    cap = cv2.VideoCapture(r'D:\PyProjects\innopolis-high-voltage-challenge\data\video\500_vertical_1[1920x1090_60fps].MP4')
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
    cnt = 0
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

        if cnt % 40 == 0:
            file_name = 'real_video_11_frame_' + str(cnt)+'.jpg'
            output_full_path = Path(OUTPUT_DIR, file_name)
            print(output_full_path)
            cv2.imwrite(str(output_full_path), original_frame)

        cnt += 1
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
