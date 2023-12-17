import os
import cv2
import copy
import time
import logging
from .tools_utility import get_minarea_rect_crop, get_rotate_crop_image
from .torchocr.utils.logging import get_logger
from .infer_det_class import TextDetector
from .infer_rec_class import TextRecognizer
from .infer_cls_class import TextClassifier

logger = get_logger()


class InferSystem:
    def __init__(self, **kwargs):
        if not kwargs.get("show_log", None):
            logger.setLevel(logging.WARNING)
        self.use_angle_cls = kwargs.get("use_angle_cls", False)
        self.text_detector = TextDetector(
            kwargs.get("det_model_config"),
            kwargs.get("det_model_name"),
            kwargs.get("devices")
        )
        self.text_recognizer = TextRecognizer(
            kwargs.get("rec_model_config"),
            kwargs.get("rec_model_name"),
            kwargs.get("character_dict_path"),
            kwargs.get("devices")
        )
        if self.use_angle_cls:
            self.text_classifier = TextClassifier(
                kwargs.get("cls_model_config"),
                kwargs.get("cls_model_name"),
                kwargs.get("devices")
            )
        self.drop_score = kwargs.get("drop_score", 0.5)
        self.det_box_type = kwargs.get("det_box_type", "quad")
        self.save_crop_res = kwargs.get("save_crop_res", False)
        self.crop_res_save_dir = kwargs.get("crop_res_save_dir", "./output")
        self.crop_image_res_index = 0

    def __call__(self, input_img, cls=True):
        time_dict = {'det': 0, 'rec': 0, 'cls': 0, 'all': 0}
        if input_img is None:
            logger.debug("no valid image provided")
            return None, None, time_dict
        start = time.time()
        ori_im = input_img.copy()
        dt_boxes, elapse = self.text_detector(input_img)  # 进入位置检测器
        time_dict['det'] = elapse
        if dt_boxes is None:
            logger.debug("no dt_boxes found, elapsed : {}".format(elapse))
            end = time.time()
            time_dict['all'] = end - start
            return None, None, time_dict
        else:
            logger.debug("dt_boxes num : {}, elapsed : {}".format(len(dt_boxes), elapse))
        img_crop_list = []
        dt_boxes = self.sorted_boxes(dt_boxes)
        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            if self.det_box_type == "quad":
                img_crop = get_rotate_crop_image(ori_im, tmp_box)
            else:
                img_crop = get_minarea_rect_crop(ori_im, tmp_box)
            img_crop_list.append(img_crop)
        if self.use_angle_cls and cls:
            img_crop_list, angle_list, elapse = self.text_classifier(img_crop_list)
            time_dict['cls'] = elapse
            logger.debug("cls num  : {}, elapsed : {}".format(
                len(img_crop_list), elapse))
        rec_res, elapse = self.text_recognizer(img_crop_list)  # 进入文字识别器
        time_dict['rec'] = elapse
        logger.debug("rec_res num  : {}, elapsed : {}".format(len(rec_res), elapse))
        if self.save_crop_res:
            self.draw_crop_rec_res(self.crop_res_save_dir, img_crop_list, rec_res)
        filter_boxes, filter_rec_res = [], []
        for box, rec_result in zip(dt_boxes, rec_res):
            text, score = rec_result
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_result)
        end = time.time()
        time_dict['all'] = end - start
        return filter_boxes, filter_rec_res, time_dict

    @staticmethod
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

    def draw_crop_rec_res(self, output_dir, img_crop_list, rec_res):
        os.makedirs(output_dir, exist_ok=True)
        bbox_num = len(img_crop_list)
        for bno in range(bbox_num):
            cv2.imwrite(
                os.path.join(output_dir,
                             f"mg_crop_{bno + self.crop_image_res_index}.jpg"),
                img_crop_list[bno])
            logger.debug(f"{bno}, {rec_res[bno]}")
        self.crop_image_res_index += bbox_num
