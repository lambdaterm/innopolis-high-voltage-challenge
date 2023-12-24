import shutil
import os

shutil.copy(os.path.join('utils', 'conv.py'),
        os.path.join('..', 'venv', 'lib', 'site-packages', 'ultralytics', 'nn', 'modules', 'conv.py'))

from utils.utils import xyxy2xywh
from utils.pipeline import Pipeline
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union
from tqdm import tqdm
import cv2
from argparse import ArgumentParser


class Statistics:

    def __init__(self, conf_supreme=0.2, iou_supreme=0.3, conf_broken=0.5, iou_broken=0.1):
        """
        initializing pipeline with such parameters
        """

        self.pipeline = Pipeline(
            golden_model_path=Path(r"model\insulator_gold.pt".replace('\\', os.sep)),
            base_model_path=Path(r"model\insulator_base.pt".replace('\\', os.sep)),
            broken_model_path=[
                Path(r"model\insulator_broken_gold.pt".replace('\\', os.sep)),
                Path(r"model\insulator_broken_original.pt".replace('\\', os.sep))
            ],
            conf_supreme=conf_supreme,
            iou_supreme=iou_supreme,
            conf_broken=conf_broken,
            iou_broken=iou_broken,
        )

        print("[+] Pipeline loaded")


    def process_folder(self, folder_from: Union[Path,str], folder_to:Union[Path,str], plot=True):
        """
        Function runs through the folder and detects broken insulators.
        Results are written in xywh format into pandas Dataframe

        folder_from - where to look for images,
        folder_to - where to save images,
        plot - should we plot marks on images, if False, then nothing will be written into folder_to
        """

        if isinstance(folder_from, str):
            folder_from = Path(folder_from)

        if isinstance(folder_to, str):
            folder_to = Path(folder_to)


        result_array = np.empty((0, 7))

        print('[+] Detecting broken ISOs', flush=True)
        for img_file in tqdm(folder_from.iterdir()):

            img = cv2.imread(img_file.as_posix())
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]

            _, broken_iso = self.pipeline.predict(
                img_path=img,
                img_sizes_insulators=(2500, 3500, 4500, 5500, 6500),
                img_sizes_broken=(640, ),
                tta=True
            )

            if len(broken_iso) == 0:
                broken_iso = np.array([0., 0., 0., 0., 0.001, 0]).reshape(-1, 6)

            else:
                # plotting detection
                if plot:
                    for box in broken_iso[..., :4]:
                        box = box[:4].astype(int)
                        img = cv2.rectangle(img, box[0:2], box[2:4], color=(255, 0, 0), thickness=3)

                    folder_to.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(
                        folder_to.joinpath(img_file.name).as_posix(),
                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    )


                # collecting info and converting to xywh format
                broken_iso[..., :4] = broken_iso[..., :4] / ([w, h] * 2)
                broken_iso[..., :4] = xyxy2xywh(broken_iso[..., :4])
                # broken_iso[..., :4] = broken_iso[..., :4].round(5)

            filename = np.array([f"{img_file.stem}" for i in range(len(broken_iso))])[..., None]
            result = np.hstack((filename, broken_iso))
            result_array = np.vstack((result_array, result))


        self.result_df = pd.DataFrame(
            columns=['file_name', 'x', 'y', 'w', 'h', 'probability', 'obj_class'],
            data=result_array
        )

        self.result_df[self.result_df.columns[1:7]] = self.result_df[self.result_df.columns[1:7]].astype(np.float32)
        print('[+] Done calculating ISOs')

    def to_csv(self, filename: Path):
        """
        Generates .csv file accorind to submit format
        header - file_name,rbbox,probability


        """

        if not hasattr(self, 'result_df'):
            print('[!] Dataframe with results not yet generated. Pls call process_folder function')

        df_grouped = self.result_df.groupby('file_name').apply(
            lambda x: (x.iloc[:, 1:5].values.tolist(), x.iloc[:, 5:6].values.reshape(-1).tolist()), )
        df_grouped = pd.DataFrame(
            data=df_grouped.tolist(),
            index=df_grouped.index,
            columns=['rbbox', 'probability'],
        )
        df_grouped.to_csv(filename, quoting=0, index=True)

        # Возможно тут надо побороться с ошибками ковычек

        print('[+] Done!')


if __name__ == '__main__':
    parser = ArgumentParser(description='innopolis-high-voltage-challenge')
    parser.add_argument('--indir', type=str, help='Input dir with images')
    args = parser.parse_args()

    s = Statistics()
    s.process_folder(str(args.indir).replace('\\', os.sep), '.\OUTPUT'.replace('\\', os.sep))
    s.to_csv('result.csv')