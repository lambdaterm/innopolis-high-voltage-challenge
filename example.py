import shutil
import os

shutil.copy('conv.py',
        os.path.join('venv', 'lib', 'site-packages', 'ultralytics', 'nn', 'modules', 'conv.py'))

import sys
from src.utils.pipeline import Pipeline
from pathlib import Path

IMG_PATH = Path(r'..\data\InnopolisTestImages\DJI_0032.JPG')

def main():

    pipeline = Pipeline(
        golden_model_path=Path(r"src\model\insulator_gold.pt"),
        base_model_path=Path(r"src\model\insulator_base.pt"),
        broken_model_path=[
            Path(r"src\model\insulator_broken_gold.pt"),
            Path(r"src\model\insulator_broken_original.pt"),
            ],
        # broken_model_path=Path(r"D:\ML\ResultModels\Insulators\InsulatorModel\yolo_broken_insulators_m_gold\train5\weights\best.pt"),
        conf_supreme=0.5,
        iou_supreme=0.5,
        conf_broken=0.5,
        iou_broken=0.1,
    )

    insulator, broken = pipeline.predict(
        IMG_PATH,
        img_sizes_insulators=(1500, 2500),
        img_sizes_broken=(640, 960),
    )

    print("Coords for insulator:")
    print(insulator)

    print("Coords for broken insulators:")
    print(broken)


if __name__ == '__main__':
    main()


