import torch
from pathlib import Path
import numpy as np
import cv2
from typing import Union
from .preprocessing import YoloPreprocessing
from .postprocessing import YoloDetectorNMS, BasePostprocessing, YoloDetectorNME
from ultralytics import YOLO

class YoloWrap(torch.nn.Module):

    def __init__(self, yolo_model):
        super().__init__()
        self.model = yolo_model.model

    def forward(self, tensor):
        result = self.model(tensor)
        return torch.permute(result[0], [0, 2, 1])

        # if len(result) == 2:
        #     return torch.permute(result[0], result[1][-1]
        # else:
        #     return result[0], torch.nan


class SupremeYOLO(torch.nn.Module):

    def __init__(self, golden_model, yolo_model):
        super().__init__()
        self.golden_model = YoloWrap(golden_model)
        self.yolo_model = YoloWrap(yolo_model)


    def forward(self, tensor):
        with torch.no_grad():
            detection_yolo = self.yolo_model(tensor)
            detection_gold = self.golden_model(tensor)
            # detection_yolo = torch.permute(detections_yolo, [0,2,1])
            # detection_gold = torch.permute(detections_gold, [0,2,1])
            detection = torch.cat((detection_gold, detection_yolo), dim=1)
            return detection



class SupremeYOLOInference:

    def __init__(self, golden_model_path: Path, base_model_path: Path, labels: list, conf=0.5, iou=0.1):

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


        self.model = SupremeYOLO(YOLO(golden_model_path), YOLO(base_model_path))
        self.model.to(self.device)
        self.model.eval()
        self.preprocess = YoloPreprocessing()
        self.postprocess = YoloDetectorNME(labels=labels, cls=0.05, iou=iou, verbose=False)
        self.conf = conf

    def _predict(self, img_path: Union[np.ndarray, Path], image_size=640):

        img_prepared, pad_ratio, pad_extra, pad_to_size, _ = self.preprocess(img_path, image_size=image_size)
        padding_meta = {
            'pad_to_size': pad_to_size,
            'pad_extra': pad_extra,
            'ratio': pad_ratio,
        }
        # print(img_prepared.shape)
        img_prepared = np.transpose(img_prepared.copy(), [0, 3, 1, 2])/255
        tensor = torch.tensor(img_prepared, dtype=torch.float32, device=self.device)

        result = self.model(tensor).cpu().numpy()
        result = result[0]
        detections = self.postprocess(result, padding_meta=padding_meta, resize=True, numpy=True)

        return detections


    def predict(self, img_path: Union[Path, np.ndarray], image_sizes=(640, 960, 1500, 2000, 2500, 3000)):

        det = []
        for image_size in image_sizes:

            result = self._predict(img_path, image_size=image_size)
            if len(result)>0:
                det.append(result)
        det = np.concatenate(det)
        # print(det)
        # det = det[self.postprocess.nme(det[..., :4],  det[..., 4:5], det[..., 5:6], self.postprocess.iou)]
        det = self.postprocess.nme(det[..., :4],  det[..., 4:5], det[..., 5:6], self.postprocess.iou)
        det = det[np.where(det[..., 4] > self.conf)]
        det = self.postprocess.nme(det[..., :4],  det[..., 4:5], det[..., 5:6], self.postprocess.iou)
        if len(det)>0:
            det = det[self.filter_by_area(det[..., :4], area_threshold=4000)]
        return det


    def filter_by_area(self, coords: np.ndarray, area_threshold=1000):
        area = (coords[:, 2] - coords[:, 0]) * (coords[:, 3] - coords[:, 1])
        return np.where(area > area_threshold)




class YOLOInference:

    def __init__(self, base_model_path: Union[list,Path], labels: list, conf=0.5, iou=0.1):

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if isinstance(base_model_path, list):
            gold_model_path, original_model_path = base_model_path
            gold_model, original_model = YOLO(gold_model_path), YOLO(original_model_path)
            self.model = SupremeYOLO(golden_model=gold_model, yolo_model=original_model)
        else:
            self.model = YoloWrap(YOLO(base_model_path))
        self.model.to(self.device)
        self.model.eval()
        self.preprocess = YoloPreprocessing(padding_size=(0, 0))
        self.preprocess = YoloPreprocessing()
        self.postprocess = YoloDetectorNMS(labels=labels, cls=conf, iou=iou, verbose=False)


    def _predict(self, img: Union[Path, np.ndarray], image_size=640):



        img_prepared, pad_ratio, pad_extra, pad_to_size, _ = self.preprocess(img,
                                                                             image_size=image_size,
                                                                             auto=False,
                                                                             scaleup=True)
        padding_meta = {
            'pad_to_size': pad_to_size,
            'pad_extra': pad_extra,
            'ratio': pad_ratio,
        }
        # print(img_prepared.shape)
        img_prepared = np.transpose(img_prepared.copy(), [0, 3, 1, 2])/255
        tensor = torch.tensor(img_prepared, dtype=torch.float32, device=self.device)
        result = self.model(tensor)[0].cpu().numpy()
        result = np.squeeze(result)
        detections = self.postprocess(result, padding_meta=padding_meta, resize=True, numpy=True)

        return detections


    def predict(self, img_path: Union[Path, np.ndarray], image_sizes=(640, 960, 1500), tta=False):

        det = []
        for image_size in image_sizes:

            result = self._predict(img_path, image_size=image_size)
            if len(result) > 0:
                det.append(result)

            if tta:
                img_90 = cv2.rotate(img_path.copy(), cv2.ROTATE_90_CLOCKWISE)
                img_180 = cv2.rotate(img_path.copy(), cv2.ROTATE_180)
                img_270 = cv2.rotate(img_path.copy(), cv2.ROTATE_90_COUNTERCLOCKWISE)

                # 90 degrees
                result = self._predict(img_90, image_size=image_size)
                if len(result) > 0:
                    result[:, :4] = self.rotate90Degcounter(result[:, :4], img_90.shape[1])
                    det.append(result)



                # 180 degrees
                h, w = img_180.shape[:2]
                c_x, c_y = w // 2, h // 2
                result = self._predict(img_180, image_size=image_size)
                if len(result) > 0:
                    result[:, :4] = np.hstack((self.rotate(result[:, 2:4], (c_x, c_y), degrees=180),
                                               self.rotate(result[:, :2], (c_x, c_y), degrees=180))).reshape(-1, 4)
                    det.append(result)

                # -90 degrees
                result = self._predict(img_270, image_size=image_size)
                if len(result) > 0:
                    result[:, :4] = self.rotate90Deg(result[:, :4], img_270.shape[0])
                    det.append(result)





        if len(det) > 0:
            det = np.concatenate(det)
            print(det)
            det = det[self.postprocess.nms(det[..., :4],  det[..., 4:5], self.postprocess.iou)]
            print(det)
        return det

    @staticmethod
    def rotate(p, origin=(0, 0), degrees=0):
        angle = np.deg2rad(degrees)
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]])
        o = np.atleast_2d(origin)
        p = np.atleast_2d(p)
        return np.squeeze((R @ (p.T - o.T) + o.T).T)

    @staticmethod
    def rotate90Degcounter(bndbox, img_width):
        """Rotates bbox by 90 degrees counterclockwise"""
        x_min, y_min, x_max, y_max = bndbox[:, 0:1], bndbox[:, 1:2], bndbox[:, 2:3], bndbox[:, 3:4]
        new_xmin = y_min
        new_ymin = img_width - x_max
        new_xmax = y_max
        new_ymax = img_width - x_min
        return np.hstack((new_xmin, new_ymin, new_xmax, new_ymax))

    @staticmethod
    def rotate90Deg(bndbox, img_height):
        """Rotates bbox by 90 degrees clockwise"""
        x_min, y_min, x_max, y_max = bndbox[:, 0:1], bndbox[:, 1:2], bndbox[:, 2:3], bndbox[:, 3:4]
        new_xmin = img_height - y_max  # Reflection about center X-line
        new_ymin = x_min
        new_xmax = img_height - y_min  # Reflection about center X-line
        new_ymax = x_max
        return np.hstack((new_xmin, new_ymin, new_xmax, new_ymax))