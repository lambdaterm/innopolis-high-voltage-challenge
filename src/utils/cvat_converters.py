from pathlib import Path
import pandas as pd
import numpy as np
from shutil import make_archive, rmtree, unpack_archive
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET
from tqdm import tqdm
import cv2
tqdm.pandas()

class BaseConverter:

    def __init__(self, format_cvat: str, format_target: str):
        '''
        Указываем форматы конвертации
        '''

        self.format_cvat = format_cvat
        self.format_target = format_target

    def to_cvat(self, folder_to: Path, folder_from: Path, *args, **kwargs):
        '''
        Конвертилка из неизвестного формата в определенный поддерживаемый CVAT формат
        :param folder_to:
        :param folder_from:
        :param kwargs:
        :return:
        '''
        pass

    def from_cvat(self, folder_to: Path, folder_from: Path, *args, **kwargs):
        '''
        Метод конвертации из формата CVAT в определенный формат
        :param folder_to:
        :param folder_from:
        :param args:
        :param kwargs:
        :return:
        '''
        pass



class YOLOConverter(BaseConverter):

    def __init__(self):
        super().__init__('YOLO1.1', 'YOLO')


    def from_cvat(self, folder_to: Path, folder_from: Path, *args, **kwargs):
        def copy_files(x: pd.Series, folder_to: Path):
            img_folder = folder_to.joinpath(x['RelPath'], 'images')
            img_folder.mkdir(parents=True, exist_ok=True)
            lbl_folder = folder_to.joinpath(x['RelPath'], 'labels')
            lbl_folder.mkdir(parents=True, exist_ok=True)

            if not pd.isna(x['Image']):
                img_folder.joinpath(x['Image'].name).write_bytes(x['Image'].read_bytes())
            lbl_folder.joinpath(x['Label'].name).write_text(x['Label'].read_text())


        folder_to.mkdir(parents=True, exist_ok=True)


        if folder_from.is_file():
            with TemporaryDirectory() as tmp:
                working_dir = Path(tmp)
                unpack_archive(folder_from, working_dir, format='zip')

                labels = [label for label in working_dir.joinpath('obj_train_data').glob('**/*.txt')]
                df_labels = pd.DataFrame(data=labels, columns=['Label'])
                df_labels['FileStem'] = df_labels.Label.map(lambda x: x.stem)

                imgs = [img for img in working_dir.joinpath('obj_train_data').glob('**/*.*') if img.suffix != '.txt']
                df_images = pd.DataFrame(data=imgs, columns=['Image'])
                df_images['FileStem'] = df_images.Image.map(lambda x: x.stem)

                df = pd.merge(df_images, df_labels, left_on='FileStem', right_on='FileStem', how='outer')


                #check for duplicates
                duplicates = df.groupby('FileStem')['Image'].count().loc[lambda x: x > 1].sort_index()
                if len(duplicates) > 0:
                    print('[!] Found duplicates!!! Please clean them first')
                    return duplicates


                df['RelPath'] = df['Label'].map(lambda x: x.relative_to(working_dir.joinpath('obj_train_data', 'obj_train_data')).parent)


                # return df

                df.progress_apply(copy_files, axis=1, folder_to=folder_to)



                folder_to.joinpath('labels.txt').write_text(working_dir.joinpath('obj.names').read_text())







    def to_cvat(self, folder_to: Path, folder_from: Path, *args, **kwargs):

        arch = kwargs.get('archive', False)
        split_mode = kwargs.get('split_mode', False)

        labels = folder_from.joinpath('labels.txt').read_text().splitlines()

        images = [file for file in folder_from.glob('**/images/*.*') if file.is_file()]
        df = pd.DataFrame(
            data=images,
            columns=['Image']
        )
        df['Label'] = df.Image.map(lambda x: x.parents[1].joinpath('labels', x.stem + '.txt'))
        df['Type'] = df.Image.map(lambda x: res if (res := x.relative_to(folder_from).parts[0]) != 'images' else '')
        df['ImageRel'] = df.Image.map(lambda x: x.relative_to(folder_from))


        if split_mode == 'type':
            for group_name, group in df.groupby('Type'):
                group_folder = folder_to.joinpath(group_name)
                group_folder.mkdir(exist_ok=True, parents=True)
                self.__copy_cvat_files(group, group_folder, labels)
                if arch:
                    make_archive(folder_to.joinpath(group_name).as_posix(), 'zip', group_folder)
                    rmtree(group_folder, ignore_errors=True)

        elif split_mode == 'parts':
            quantity = kwargs.get('size', 3)
            splitted_array = np.array_split(df, quantity)
            for group_name, group in enumerate(splitted_array):
                group_folder = folder_to.joinpath(str(group_name+1))
                group_folder.mkdir(exist_ok=True, parents=True)
                self.__copy_cvat_files(group, group_folder, labels)
                if arch:
                    make_archive(folder_to.joinpath(str(group_name+1)).as_posix(), 'zip', group_folder)
                    rmtree(group_folder, ignore_errors=True)

        elif split_mode == 'all':
            self.__copy_cvat_files(df, folder_to, labels)
            if arch:
                make_archive(folder_to.parent.joinpath('CVAT').as_posix(), 'zip', folder_to)
                # rmtree(folder_to, ignore_errors=True)


    def __copy_cvat_files(self, df: pd.DataFrame, folder_to: Path, labels: list):
        def copy_files(x: pd.Series, copy_to: Path):
            copy_to_series = copy_to.joinpath(x['Type']) if x['Type'] != '' else copy_to
            copy_to_series.mkdir(exist_ok=True)
            copy_to_series.joinpath(x['Image'].name).write_bytes(x['Image'].read_bytes())

            if x['Label'].exists():
                copy_to_series.joinpath(x['Label'].name).write_bytes(x['Label'].read_bytes())
            else:
                copy_to_series.joinpath(x['Label'].name).touch(mode=0o777)

        train_list = df.ImageRel.map(lambda x: Path('data', 'obj_train_data',
                                                    x.parts[0] if x.parts[0] != 'images' else '',
                                                    x.parts[-1]).as_posix()
                                     ).tolist()

        self.__make_cvat_files(folder_to, labels, train_list)

        files_folder = folder_to.joinpath('obj_train_data')
        files_folder.mkdir(parents=True, exist_ok=True)

        df.progress_apply(copy_files, axis=1, copy_to=files_folder)


    def __make_cvat_files(self, folder_to: Path, labels:list, train_list: list):
        folder_to.joinpath('obj.names').write_text('\n'.join(labels))  # obj.names

        text_to_write = f"classes = {len(labels)}\n"
        text_to_write += "train = data/train.txt\n"
        text_to_write += "names = data/obj.names\n"
        text_to_write += "backup = backup/\n"

        folder_to.joinpath('obj.data').write_text(text_to_write)
        folder_to.joinpath('train.txt').write_text('\n'.join(train_list)+'\n')




class YOLOSegCVATConverter(BaseConverter):

    def __init__(self):
        super().__init__('CVAT 1.1', 'YOLO_Segm')


    def from_cvat(self, folder_to: Path, folder_from: Path, *args, **kwargs):

        if kwargs['empty_dir']:
            rmtree(folder_to, ignore_errors=True)

        folder_to.mkdir(parents=True, exist_ok=True)

        if folder_from.is_file():
            with TemporaryDirectory() as tmp:
                working_dir = Path(tmp)
                unpack_archive(folder_from, working_dir, format='zip')
                self._from_cvat(folder_to, working_dir)
        else:
            self._from_cvat(folder_to, folder_from)



    def _from_cvat(self, folder_to: Path, folder_from: Path, *args, **kwargs):


        tree = ET.parse(folder_from.joinpath('annotations.xml').as_posix())
        root = tree.getroot()

        # get dict of labels to index
        labels = {name.text: str(i) for i, label in enumerate(root.iter('label')) for name in label.iter('name')}

        folder_to.joinpath('labels.txt').write_text('\n'.join(list(labels.keys())))

        for child in tqdm(root.iter('image')):

            # getting filename
            h, w = int(child.attrib['height']), int(child.attrib['width'])
            rel_pic_loc = Path(child.attrib['name'])
            if len(rel_pic_loc.parts)<=2:
                doctype, filename = '', rel_pic_loc.parts[-1]
            else:
                other, doctype, filename = '\\'.join(rel_pic_loc.parts[:-2]), rel_pic_loc.parts[-2], rel_pic_loc.parts[-1]

            coords = []
            for polygon in child:

                # getting text_label
                label = polygon.attrib['label']

                # getting points
                points = polygon.attrib['points']
                points = points.replace(',', ' ').replace(';', ' ').split(' ')
                points = list(map(lambda x: float(x), points))
                points[::2] = list(map(lambda x: str(round(x/w,5)), points[::2]))
                points[1::2] = list(map(lambda x: str(round(x/h,5)), points[1::2]))

                points.insert(0, labels[label])
                coords.append(' '.join(points))


            # saving imgs
            img_file = folder_to.joinpath(other, doctype, 'images', filename)
            img_file.parent.mkdir(parents=True, exist_ok=True)
            if folder_from.joinpath('images').exists():
                pic_loc = folder_from.joinpath('images', child.attrib['name'])
            else:
                pic_loc = folder_from.joinpath(child.attrib['name'])
            img_file.write_bytes(pic_loc.read_bytes())

            # saving labels
            label_file = folder_to.joinpath(other, doctype, 'labels', filename.rsplit('.', maxsplit=1)[0] + '.txt')
            label_file.parent.mkdir(parents=True, exist_ok=True)
            label_file.write_text('\n'.join(coords))


    def to_cvat(self, folder_to: Path, folder_from: Path, *args, **kwargs):

        assert folder_from.joinpath('labels.txt').exists(), 'file labels.txt not found in folder_from'

        limit_files = kwargs.get('limit_files', -1)
        split_mode = kwargs.get('split_mode', 'all')
        split_size = kwargs.get('split_size', 3)


        folder_to.mkdir(parents=True, exist_ok=True)

        labels = folder_from.joinpath('labels.txt').read_text().splitlines()



        files = []
        for folder in folder_from.glob('**/images'):
            for i, img_file in enumerate(folder.iterdir()):
                files.append((img_file, lbl if (lbl:=img_file.parents[1].joinpath('labels', img_file.stem + '.txt')).exists()
                else None))

                if i >= limit_files - 1 and limit_files != -1:
                    break




        if split_mode == 'all':
            self.__make_tree_and_copy(folder_to=folder_to,
                                      folder_from=folder_from,
                                      files=files,
                                      labels=labels,
                                      )
        elif split_mode == 'parts':
            splitted_files = np.array_split(files, split_size)
            for i, files in enumerate(splitted_files):
                print(f'Splitting part {i+1}')
                self.__make_tree_and_copy(folder_to=folder_to.joinpath(f'part{i+1}'),
                                          folder_from=folder_from,
                                          files=files,
                                          labels=labels,)



    def __make_tree_and_copy(self, folder_to, folder_from, files, labels):

        root = ET.Element('annotations')

        for i, (img_file, lbl_file) in enumerate(tqdm(files)):
            img = cv2.imread(img_file.as_posix())

            #building relative path
            path_parts = list(img_file.relative_to(folder_from).parts)
            path_parts[:-1] = list(filter(lambda x: 'images' not in x, path_parts[:-1]))
            new_image_rel_path = Path('/'.join(path_parts))

            h,w = img.shape[:2]

            # Tree making

            img_elem = ET.SubElement(
                root,
                'image',
                 {
                     'id': str(i),
                     'name': new_image_rel_path.as_posix(),
                     'width': str(w),
                     'height': str(h),
                 }
            )

            if lbl_file:
                lbls = lbl_file.read_text().splitlines()
                for lbl in lbls:
                    lbl_index, coords = lbl.split(' ', maxsplit=1)


                    coords = np.array(coords.split(' ')).reshape(-1, 2).astype(float)
                    coords = (coords*[w,h]).round(2).astype('str')
                    new_coords = []
                    for coord in coords.tolist():
                        new_coords.append(','.join(coord))
                    new_coords = ';'.join(new_coords)
                    lbl_dict = {
                        'label': labels[int(lbl_index)],
                        'source': 'manual',
                        'occluded': '0',
                        'points': new_coords,
                        'z_order': '0',
                    }
                    ET.SubElement(
                        img_elem,
                        'polygon',
                        lbl_dict
                    )

            # copying img

            new_image_path = folder_to.joinpath(new_image_rel_path)
            new_image_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                new_image_path.write_bytes(img_file.read_bytes())
            except:
                print(new_image_path)
                print(img_file)
                raise PermissionError


        tree = ET.ElementTree(root)
        tree.write(folder_to.joinpath('annotations.xml'))

        # print(ET.tostring(root, encoding='utf8').decode('utf8'))

