import numpy as np


class BasePostprocessing:

    def __init__(self, verbose: bool = False, **kwargs):
        if verbose:
            print(f"[+] {self.__class__.__name__} loaded")

    def __call__(self, *args, **kwargs):
        pass



class YoloDetectorNMS(BasePostprocessing):

    def __init__(self, labels:list, iou=0.2, cls=0.5, verbose=False):
        super().__init__(verbose)
        self.iou = iou
        self.cls = cls
        self.labels = labels


    def __call__(self,
                 vector: np.ndarray,
                 **kwargs,
                 ):
        '''
        Function translates predictions from yolo detector
        :param prediction: Tensor from yolo exit
        :param imgsize: size of image, h,w
        :param conf_thresh: class confidence threshold
        :param iou_thresh:  intersection over union threshold
        :return: return list of lists, which contains - x_top, y_top, x_bottom, y_bottom, conf, label index, label
        '''



        if kwargs.get('padding_meta') is None:
            padding_meta = {
                'pad_to_size': (0,0),
                'pad_extra': (0,0),
                'ratio': (1, 1),

            }
        else:
            padding_meta = kwargs['padding_meta']

        detect_res = []
        if vector.shape[0] == 0:
            return detect_res

        n_classes = len(self.labels)
        vector = vector[vector[..., 4:4+n_classes].max(axis=1) > self.cls]

        if len(vector) == 0:
            return []

        box, det, seg = np.split(vector, [4, 4+n_classes], axis=1)
        box = self.xywh2xyxy(box)  # creating from x,y height, width -> xy xy coords of box

        conf, j = det.max(axis=1, keepdims=True), det.argmax(axis=1, keepdims=True)
        i = self.nms(box, conf, self.iou)  # calculating non maximum suppression
        detect_res = np.concatenate((box, conf, j, seg), axis=1)[i]

        ind = np.lexsort((detect_res[...,0], detect_res[..., 1]))
        detect_res = detect_res[ind]

        if kwargs.get('resize'): # should we resize to original size
        ### resize to original pic
            detect_res[..., :4:2] = (detect_res[..., :4:2] - padding_meta['pad_to_size'][0]) \
                                    / padding_meta['ratio'][0] - padding_meta['pad_extra'][0]
            detect_res[..., 1:4:2] = (detect_res[..., 1:4:2] - padding_meta['pad_to_size'][1]) \
                                     / padding_meta['ratio'][1] - padding_meta['pad_extra'][1]
            detect_res[detect_res < 0] = 0
            detect_res[..., :4] = np.round(detect_res[..., :4], 0)
            detect_res[..., 4] = np.round(detect_res[..., 4], 3)

        if kwargs.get('numpy') is not None:  # Return result as numpy array without adding labels
            return detect_res

        if self.labels is not None:
            detection_result = []
            for detected in detect_res:
                obj_detected = list(detected)
                obj_detected[:4] = list(map(lambda x: int(x), obj_detected[:4]))
                obj_detected[5] = int(obj_detected[5])
                try:
                    obj_detected.append(self.labels[obj_detected[5]])
                except:
                    obj_detected.append("Unsupported class")

                detection_result.append(obj_detected)
            return detection_result
        else:
            return detect_res.tolist()

    @staticmethod
    def xywh2xyxy(x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    @staticmethod
    def nms(bounding_boxes: np.array, confidence_score: np.array, threshold: float):
        '''
        Finds best boxes for found objects
        :param bounding_boxes: coords of boxes
        :param confidence_score: np.ndarray with shape (n, 1)
        :param threshold: IoU_threshold
        :return: array of indexes, that corresponds to best found BOXES
        '''

        # If no bounding boxes, return empty list
        if len(bounding_boxes) == 0:
            return [], []

        # Bounding boxes
        boxes = np.array(bounding_boxes)

        # coordinates of bounding boxes
        start_x = boxes[:, 0]
        start_y = boxes[:, 1]
        end_x = boxes[:, 2]
        end_y = boxes[:, 3]

        # Confidence scores of bounding boxes
        score = np.array(confidence_score)

        # Picked bounding boxes
        picked_boxes_index = []

        # Compute areas of bounding boxes
        areas = (end_x - start_x) * (end_y - start_y)

        # Sort by confidence score of bounding boxes
        order = np.argsort(score, axis=0).reshape(-1)

        # Iterate bounding boxes
        while order.size > 0:
            # The index of largest confidence score
            index = order[-1]

            # Pick the bounding box with largest confidence score
            picked_boxes_index.append(index)


            # Compute ordinates of intersection-over-union(IOU)
            x1 = np.maximum(start_x[index], start_x[order[:-1]])
            x2 = np.minimum(end_x[index], end_x[order[:-1]])
            y1 = np.maximum(start_y[index], start_y[order[:-1]])
            y2 = np.minimum(end_y[index], end_y[order[:-1]])




            # Compute areas of intersection-over-union
            w = np.maximum(0.0, x2 - x1)
            h = np.maximum(0.0, y2 - y1)

            intersection = w * h


            # Compute the ratio between intersection and union
            ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

            left = np.where(ratio < threshold)
            order = order[left]

        return picked_boxes_index

class YoloDetectorNME(BasePostprocessing):

    def __init__(self, labels: list, iou=0.2, cls=0.5, verbose=False):
        super().__init__(verbose)
        self.iou = iou
        self.cls = cls
        self.labels = labels

    def __call__(self,
                 vector: np.ndarray,
                 **kwargs,
                 ):
        '''
        Function translates predictions from yolo detector
        :param prediction: Tensor from yolo exit
        :param imgsize: size of image, h,w
        :param conf_thresh: class confidence threshold
        :param iou_thresh:  intersection over union threshold
        :return: return list of lists, which contains - x_top, y_top, x_bottom, y_bottom, conf, label index, label
        '''

        if kwargs.get('padding_meta') is None:
            padding_meta = {
                'pad_to_size': (0, 0),
                'pad_extra': (0, 0),
                'ratio': (1, 1),

            }
        else:
            padding_meta = kwargs['padding_meta']

        detect_res = []
        if vector.shape[0] == 0:
            return detect_res

        n_classes = len(self.labels)
        vector = vector[vector[..., 4:4 + n_classes].max(axis=1) > self.cls]

        if len(vector) == 0:
            return []

        box, det, seg = np.split(vector, [4, 4 + n_classes], axis=1)
        box = self.xywh2xyxy(box)  # creating from x,y height, width -> xy xy coords of box

        conf, j = det.max(axis=1, keepdims=True), det.argmax(axis=1, keepdims=True)
        detect_res = self.nme(box, conf, j, self.iou)  # calculating non maximum suppression

        ind = np.lexsort((detect_res[..., 0], detect_res[..., 1]))
        detect_res = detect_res[ind]

        if kwargs.get('resize'):  # should we resize to original size
            ### resize to original pic
            detect_res[..., :4:2] = (detect_res[..., :4:2] - padding_meta['pad_to_size'][0]) \
                                    / padding_meta['ratio'][0] - padding_meta['pad_extra'][0]
            detect_res[..., 1:4:2] = (detect_res[..., 1:4:2] - padding_meta['pad_to_size'][1]) \
                                     / padding_meta['ratio'][1] - padding_meta['pad_extra'][1]
            detect_res[detect_res < 0] = 0
            detect_res[..., :4] = np.round(detect_res[..., :4], 0)
            detect_res[..., 4] = np.round(detect_res[..., 4], 3)

        if kwargs.get('numpy') is not None:  # Return result as numpy array without adding labels
            return detect_res

        if self.labels is not None:
            detection_result = []
            for detected in detect_res:
                obj_detected = list(detected)
                obj_detected[:4] = list(map(lambda x: int(x), obj_detected[:4]))
                obj_detected[5] = int(obj_detected[5])
                try:
                    obj_detected.append(self.labels[obj_detected[5]])
                except:
                    obj_detected.append("Unsupported class")

                detection_result.append(obj_detected)
            return detection_result
        else:
            return detect_res.tolist()

    @staticmethod
    def xywh2xyxy(x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    @staticmethod
    def nme(bounding_boxes: np.array, confidence_score: np.array, cls_ix: np.array, threshold: float):
        '''
                Finds best boxes for found objects
                :param bounding_boxes: coords of boxes
                :param confidence_score: np.ndarray with shape (n, 1)
                :param threshold: IoU_threshold
                :return: array of indexes, that corresponds to best found BOXES
                '''

        # If no bounding boxes, return empty list
        if len(bounding_boxes) == 0:
            return []

        # Bounding boxes
        boxes = np.array(bounding_boxes)

        # coordinates of bounding boxes
        start_x = boxes[:, 0]
        start_y = boxes[:, 1]
        end_x = boxes[:, 2]
        end_y = boxes[:, 3]

        # Confidence scores of bounding boxes
        score = np.array(confidence_score)

        # Picked bounding boxes
        picked_boxes = []

        # Compute areas of bounding boxes
        areas = (end_x - start_x) * (end_y - start_y)

        # Sort by confidence score of bounding boxes
        order = np.argsort(score, axis=0).reshape(-1)

        # Iterate bounding boxes
        while order.size > 0:
            # The index of largest confidence score
            index = order[-1]

            # Pick the bounding box with largest confidence score


            # Compute ordinates of intersection-over-union(IOU)
            x1 = np.maximum(start_x[index], start_x[order[:-1]])
            x2 = np.minimum(end_x[index], end_x[order[:-1]])
            y1 = np.maximum(start_y[index], start_y[order[:-1]])
            y2 = np.minimum(end_y[index], end_y[order[:-1]])

            # Compute areas of intersection-over-union
            w = np.maximum(0.0, x2 - x1)
            h = np.maximum(0.0, y2 - y1)

            intersection = w * h

            # Compute the ratio between intersection and union
            ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

            similar_ix = order[np.where(ratio >= threshold)]
            similar_ix = np.insert(similar_ix, 0, index)

            sim_boxes = boxes[similar_ix]
            bounding_boxes = sim_boxes[..., 0].min(), sim_boxes[..., 1].min(), sim_boxes[..., 2].max(), sim_boxes[..., 3].max()
            picked_boxes.append(np.concatenate([bounding_boxes, confidence_score[index], cls_ix[index]]))

            left = np.where(ratio < threshold)
            order = order[left]

        picked_boxes = np.vstack(picked_boxes)

        return picked_boxes
