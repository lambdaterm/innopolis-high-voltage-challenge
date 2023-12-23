import copy
import time
import numpy as np
import math
import matplotlib.pyplot as plt
import cv2
from typing import Tuple, Union
from typing import Tuple
from pathlib import Path
from Libs import utils
import sys
from Libs.pipeline import Pipeline

ModelResult = Tuple[int, str, float]

OUTPUT_DIR = r'D:\Data\Result'
IMG_PATH = Path(r'data\InnopolisTestImages\DJI_0032.JPG')

if __name__ == '__main__':

    print('Models initialization ...')
    pipeline = Pipeline(
        golden_model_path=Path(r"Models\insulator_gold.pt"),
        base_model_path=Path(r"Models\insulator_base.pt"),
        broken_model_path=Path(
            r"Models\insulator_broken_gold.pt"),
        # broken_model_path=Path(r"D:\ML\ResultModels\Insulators\InsulatorModel\yolo_broken_insulators_m_gold\train5\weights\best.pt"),
        conf_supreme=0.5,
        iou_supreme=0.5,
        conf_broken=0.4,
        iou_broken=0.1,
    )
    insulator, broken = pipeline.predict(
        IMG_PATH,
        img_sizes_insulators=(1080, 1920),
        img_sizes_broken=(640, 960),
        # img_sizes_broken=(480, 640),
    )

    ###
    print('Done')
    ###
    # Забираем видео из видео потока
    cap = cv2.VideoCapture(r'D:\PyProjects\innopolis-high-voltage-challenge\data\video\Dron_FLY_demonstration.mp4')
    # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('output_video_from_file.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20, (1920, 1080))

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
    insulator = [0, 0, 0, 0]
    broken = [0, 0, 0, 0]
    #################################################################################
    while cap.isOpened():

        # cnt += 1
        # if cnt % 2:
        #     continue

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
        annotated_frame = original_frame

        insulator, broken = pipeline.predict(
            original_frame,
            # img_sizes_insulators=(1500, 2500),
            img_sizes_insulators=(1080, 1920),
            # img_sizes_broken=(640, 960),
            img_sizes_broken=(480, 640),
        )

        if len(insulator) > 0:
            coords_insulators = insulator[..., :4].astype(int)

            for block in coords_insulators:
                annotated_frame = cv2.rectangle(original_frame, block[:2], block[-2:], color=(255, 0, 0), thickness=3)

        if len(broken) > 0:
            coords_damages = broken[..., :4].astype(int)
            for block in coords_damages:
                annotated_frame = cv2.rectangle(original_frame, block[:2], block[-2:], color=(0, 0, 255), thickness=3)

        out.write(annotated_frame)
        cv2.imshow("Camera", annotated_frame)

        print(f'Frame_size = {frame.shape}')

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    #####################################################################

    cv2.waitKey(0)
    out.release()
    cap.release()
    cv2.destroyAllWindows()
