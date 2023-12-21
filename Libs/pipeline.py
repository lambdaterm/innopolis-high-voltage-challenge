import numpy as np
import cv2
from typing import Union
from .inference import YOLOInference, SupremeYOLOInference
from pathlib import Path


class Pipeline:

    def __init__(self, golden_model_path: Path,
                 base_model_path: Path,
                 broken_model_path: Path,
                 iou_supreme=0.1,
                 conf_supreme=0.5,
                 iou_broken=0.2,
                 conf_broken=0.7):

        self.supreme_yolo = SupremeYOLOInference(
            golden_model_path,
            base_model_path,
            ['isolator',],
            conf=conf_supreme,
            iou=iou_supreme,
        )

        self.broken_yolo = YOLOInference(
            broken_model_path,
            ['broken_iso',],
            conf=conf_broken,
            iou=iou_broken
        )



    def predict(self,
                img_path: Union[Path, np.ndarray, str],
                img_sizes_insulators=(640, 960, 1500, 2000, 2500, 3000),
                img_sizes_broken=(640, 960),
                ):

        if isinstance(img_path, Path):
            img = cv2.imread(img_path.as_posix())
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(img_path, str):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(img_path, np.ndarray):
            img = img_path
        else:
            raise TypeError('Unsupported image type. Should be Path or np.ndarray or str')



        result_isolator = self.supreme_yolo.predict(img, image_sizes=img_sizes_insulators)

        result_broken = []
        if len(result_isolator) == 0:
            return result_isolator, result_broken

        for box in result_isolator[..., :4]:

            x_u, y_u, x_l, y_l = box.astype(int)
            img_patch = img[y_u:y_l, x_u:x_l]
            broken = self.broken_yolo.predict(img_patch, image_sizes=img_sizes_broken)
            if len(broken)>0:
                broken[..., [0, 2]] += x_u
                broken[..., [1, 3]] += y_u
                result_broken.append(broken)
        if len(result_broken) > 0:
            result_broken = np.vstack(result_broken)
            result_broken = result_broken[self.broken_yolo.postprocess.nms(result_broken[..., :4], result_broken[..., 4:5], self.broken_yolo.postprocess.iou)]

        return result_isolator, result_broken