import os
import cv2
import numpy as np
from mindspore import Tensor, context
from yolox_tools.config import config
from yolox_tools.transform import preproc
from yolox_tools.util import load_weights, DetectionEngine

from mindyolo.models.yolox import YOLOx

LABELS = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
          'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
          'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
          'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
          'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
          'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
          'kite', 'baseball bat', 'baseball glove', 'skateboard',
          'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
          'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
          'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
          'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
          'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
          'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
          'refrigerator', 'book', 'clock', 'vase', 'scissors',
          'teddy bear', 'hair drier', 'toothbrush']


context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)


class YoloxPredict:
    def __init__(self, cfg):
        self.config = cfg
        self.detection = DetectionEngine(self.config)
        self.model = self.create_model()
        self.color_lst = self.create_label_color(LABELS)

    @staticmethod
    def create_label_color(labels):
        color_lst = []
        for _ in range(len(labels)):
            color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
            color_lst.append(color)
        return color_lst

    def create_model(self):
        if self.config.backbone == "yolox_darknet53":
            backbone = "yolofpn"
        else:
            backbone = "yolopafpn"
        model = YOLOx(self.config, backbone=backbone)
        if self.config.val_ckpt:
            model = load_weights(model, self.config.val_ckpt)
        return model

    def preprocess(self):
        img = cv2.imread(self.config.img_path)
        padded_img, r = preproc(img, self.config.input_size)
        return img, Tensor(padded_img[None]), r

    def postprocess(self, outputs, scale, img_shape=(640, 640)):
        outputs = self.detection.postprocess(outputs, self.config.num_classes, self.config.conf_thre,
                                             self.config.nms_thre)[0]
        cls = outputs[:, 6]
        scores = outputs[:, 4] * outputs[:, 5]
        bboxes = outputs[:, 0:4]
        bboxes = bboxes / scale
        bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]], 0, img_shape[1])
        bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]], 0, img_shape[0])
        return bboxes, scores, cls

    def draw_bbox(self, img_origin, bboxes, scores, cls, labels, thickness=2):
        length = bboxes.shape[0]
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i in range(length):
            x1, y1, x2, y2 = bboxes[i]
            classes = int(cls[i])
            cv2.rectangle(img_origin, (int(x1), int(y1)), (int(x2), int(y2)), self.color_lst[classes],
                          thickness=thickness)
            x = max(int(x1) - 5, 0)
            y = max(int(y1) - 5, 0)
            text = str(labels[classes]) + ' ' + str(round(scores[i], 3))
            cv2.putText(img_origin, text, (x, y), font, 0.6, (0, 127, 255), 1)
        if not self.config.save_img_path:
            basedir = os.path.dirname(self.config.img_path)
            basename = os.path.basename(self.config.img_path)
            self.config.save_img_path = os.path.join(basedir, 'predict-' + basename)
        cv2.imwrite(self.config.save_img_path, img_origin)

    def predict(self, labels):
        img_origin, padded_img, scale = self.preprocess()
        outputs = self.model(padded_img).asnumpy()
        bboxes, scores, cls = self.postprocess(outputs, scale, img_shape=img_origin.shape[:2])
        self.draw_bbox(img_origin, bboxes, scores, cls, labels)
        print('finished')


if __name__ == '__main__':
    yolox = YoloxPredict(config)
    yolox.predict(LABELS)
