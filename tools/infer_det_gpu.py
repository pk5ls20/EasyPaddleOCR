import copy
import os
import sys

import numpy as np
import cv2
import json
import torch

from tools.infer.utility import get_minarea_rect_crop
from torchocr.data import create_operators, transform
from torchocr.modeling.architectures import build_model
from torchocr.postprocess import build_post_process
from torchocr.utils.ckpt import load_ckpt
from torchocr.utils.logging import get_logger
from torchocr.utils.visual import draw_det
from torchocr.utils.utility import get_image_file_list
from tools.utility import ArgsParser
from torchocr import Config

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


def main(cfg):
    logger = get_logger()
    global_config = cfg['Global']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model = build_model(cfg['Architecture'])
    load_ckpt(model, cfg)
    model.to(device)
    model.eval()
    post_process_class = build_post_process(cfg['PostProcess'])
    transforms = build_det_process(cfg)
    ops = create_operators(transforms, global_config)
    save_res_path = global_config.get('output_dir', 'output')
    os.makedirs(save_res_path, exist_ok=True)
    with open(os.path.join(save_res_path, 'predict_det.txt'), "w") as fout:
        for file in get_image_file_list(global_config['infer_img']):
            logger.info("infer_img: {}".format(file))
            with open(file, 'rb') as f:
                img = f.read()
                data = {'image': img}
            batch = transform(data, ops)
            images = np.expand_dims(batch[0], axis=0)
            shape_list = np.expand_dims(batch[1], axis=0)
            images = torch.from_numpy(images).to(device)
            with torch.no_grad():
                preds = model(images)
            post_result = post_process_class(preds, [-1, shape_list])
            src_img = cv2.imread(file)
            dt_boxes_json = []
            if isinstance(post_result, dict):
                det_box_json = {}
                for k in post_result.keys():
                    boxes = post_result[k][0]['points']
                    dt_boxes_list = []
                    for box in boxes:
                        tmp_json = {"transcription": "", "points": np.array(box).tolist()}
                        dt_boxes_list.append(tmp_json)
                    det_box_json[k] = dt_boxes_list
                    save_det_path = f'{save_res_path}/det_results_{os.path.basename(file)}'
                    src_img = draw_det(boxes, src_img)
            else:
                boxes = post_result[0]['points']
                for box in boxes:
                    tmp_json = {"transcription": "", "points": np.array(box).tolist()}
                    dt_boxes_json.append(tmp_json)
                save_det_path = f'{save_res_path}/det_results_{os.path.basename(file)}'
                src_img = draw_det(boxes, src_img)
            img_crop_list = []
            # Here, we find all boxes
            boxes = sorted_boxes(boxes)
            for itm in range(len(boxes)):
                tmp_box = copy.deepcopy(boxes[itm])
                # which is the det_box_type?
                cv2_img = cv2.imread(file)
                img_crop = get_minarea_rect_crop(cv2_img, tmp_box)
                img_crop_list.append(img_crop)
                # is cls enabled? Here pass cls
            # Now, Iterate img_crop_list
            cv2.imwrite(save_det_path, src_img)
            out_str = f'{file}\t{json.dumps(dt_boxes_json)}'
            fout.write(out_str + '\n')
            logger.info(out_str)
            logger.info("The detected Image saved in {}".format(save_det_path))

    logger.info("success!")


if __name__ == '__main__':
    FLAGS = ArgsParser().parse_args()
    cfg = Config(FLAGS.config)
    FLAGS = vars(FLAGS)
    opt = FLAGS.pop('opt')
    cfg.merge_dict(FLAGS)
    cfg.merge_dict(opt)
    main(cfg.cfg)
    # -c
    # configs/det/ch_PP-OCRv4/ch_PP-OCRv4_det_teacher.yml
    # -o
    # Global.pretrained_model=model/ch_PP-OCRv4_det_server_train/best_accuracy.pth