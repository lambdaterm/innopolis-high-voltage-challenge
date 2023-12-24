import numpy as np


def xyxy2xywh(x):
    y = np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # top left x
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # top left y
    y[:, 2] = x[:, 2] - x[:, 0]   # bottom right x
    y[:, 3] = x[:, 3] - x[:, 1]   # bottom right y
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y