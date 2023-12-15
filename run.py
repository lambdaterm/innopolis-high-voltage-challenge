import os
import cv2

DATA_DIR = 'data'

if __name__ == '__main__':

    # model =
    print('Start model with small tensors testing ...')
    for i in os.listdir(DATA_DIR):
        image_from_file = cv2.imread(os.path.join(DATA_DIR, i))
    print('Done')





