import numpy as np
import cv2
from typing import Union
from .inference import YOLOInference, SupremeYOLOInference
from pathlib import Path
from .utils import xywh2xyxy, xyxy2xywh

class Pipeline:

    def __init__(self, golden_model_path: Path,
                 base_model_path: Path,
                 broken_model_path: Union[Path, list],
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
                img_sizes_insulators=(640, 960, 1500, 2500, 3500),
                img_sizes_broken=(640, 960),
                broken_coords_fix=(1.7, 1.3),
                tta=True
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



        max_size = max(img.shape[:2])

        img_sizes_broken = [img_size for img_size in img_sizes_insulators if img_size <= max_size]


        result_isolator = self.supreme_yolo.predict(img, image_sizes=img_sizes_insulators)

        # fixing coords to get wider
        w, h = result_isolator[..., 2] - result_isolator[..., 0], result_isolator[..., 3] - result_isolator[..., 1]
        sizes = np.vstack((w,h)).transpose()
        result_isolator[..., 0:2] = result_isolator[..., 0:2] - sizes * 0.1
        result_isolator[..., 0:2] = np.where(result_isolator[..., 0:2] < 0, 0, result_isolator[..., 0:2])
        result_isolator[..., 2:4] = result_isolator[..., 2:4] + sizes * 0.1



        result_broken = []
        if len(result_isolator) == 0:
            return result_isolator, result_broken

        for box in result_isolator[..., :4]:

            x_u, y_u, x_l, y_l = box.astype(int)
            img_patch = img[y_u:y_l, x_u:x_l]
            # img_patch = cv2.cvtColor(img_patch, cv2.COLOR_RGB2BGR)

            broken = self.broken_yolo.predict(img_patch, image_sizes=img_sizes_broken, tta=True)
            if len(broken)>0:
                broken[..., [0, 2]] += x_u
                broken[..., [1, 3]] += y_u
                result_broken.append(broken)
        if len(result_broken) > 0:
            result_broken = np.vstack(result_broken)
            result_broken = result_broken[self.broken_yolo.postprocess.nms(result_broken[..., :4], result_broken[..., 4:5], self.broken_yolo.postprocess.iou)]


            # fixing croods. КОСТЫЛЬ под метрику размеры метрики

            coords_xywh = xyxy2xywh(result_broken[..., :4])
            coords_xywh = np.where(coords_xywh == np.amax(coords_xywh[:, 2:4], axis=1)[:, None], coords_xywh * broken_coords_fix[0], coords_xywh)
            coords_xywh = np.where(coords_xywh == np.amin(coords_xywh[:, 2:4], axis=1)[:, None], coords_xywh * broken_coords_fix[1], coords_xywh)
            result_broken[..., :4] = xywh2xyxy(coords_xywh)

        return result_isolator, result_broken