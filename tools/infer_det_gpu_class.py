import copy
import os
import json
import numpy as np
import cv2
import torch
from torchocr.data import create_operators, transform
from torchocr.modeling.architectures import build_model
from torchocr.postprocess import build_post_process
from torchocr.utils.ckpt import load_ckpt
from torchocr.utils.logging import get_logger
from torchocr.utils.visual import draw_det
from tools.infer.utility import get_minarea_rect_crop
from torchocr import Config
from tools.utility import ArgsParser


class OCRDetector:
    def __init__(self, config_path, model_path):
        self.cfg = Config(config_path).cfg
        self.logger = get_logger()
        self.cfg['Global']['pretrained_model'] = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        self.model = build_model(self.cfg['Architecture'])
        load_ckpt(self.model, self.cfg)
        self.model.to(self.device)
        self.model.eval()
        self.post_process_class = build_post_process(self.cfg['PostProcess'])
        self.ops = create_operators(build_det_process(self.cfg), self.cfg['Global'])

    def __call__(self, ndarray_image):
        retval, buffer = cv2.imencode('.jpg', ndarray_image)
        img_bytes = np.array(buffer).tobytes()
        data = {'image': img_bytes}
        batch = transform(data, self.ops)
        images = np.expand_dims(batch[0], axis=0)
        shape_list = np.expand_dims(batch[1], axis=0)
        images = torch.from_numpy(images).to(self.device)
        with torch.no_grad():
            preds = self.model(images)
        post_result = self.post_process_class(preds, [-1, shape_list])
        src_img = ndarray_image
        if isinstance(post_result, dict):
            boxes = [box for sublist in post_result.values() for box in sublist[0]['points']]
        else:
            boxes = post_result[0]['points']

        boxes = sorted_boxes(np.array(boxes))
        img_crop_list = [get_minarea_rect_crop(src_img, box) for box in boxes]

        return img_crop_list, 114514 # 先返回个114514，本来这是时间


def build_det_process(cfg):
    transforms = []
    for op in cfg['Eval']['dataset']['transforms']:
        op_name = list(op)[0]
        if 'Label' in op_name:
            continue
        elif op_name == 'KeepKeys':
            op[op_name]['keep_keys'] = ['image', 'shape']
        transforms.append(op)
    return transforms


def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        for j in range(i, -1, -1):
            if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and \
                    (_boxes[j + 1][0][0] < _boxes[j][0][0]):
                tmp = _boxes[j]
                _boxes[j] = _boxes[j + 1]
                _boxes[j + 1] = tmp
            else:
                break
    return _boxes

if __name__ == '__main__':
    FLAGS = ArgsParser().parse_args()
    ocr_detector = OCRDetector(FLAGS.config)
    FLAGS = vars(FLAGS)
    opt = FLAGS.pop('opt')
    ocr_detector.cfg.merge_dict(FLAGS)
    ocr_detector.cfg.merge_dict(opt)

