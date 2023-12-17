import cv2
import time
import torch
import numpy as np
from .torchocr import Config
from .torchocr.utils.ckpt import load_ckpt
from .torchocr.utils.logging import get_logger
from .torchocr.postprocess import build_post_process
from .torchocr.data import create_operators, transform
from .torchocr.modeling.architectures import build_model
from .utility import update_rec_head_out_channels


class TextClassifier:
    def __init__(self, config_path, model_path, devices):
        self.cfg = Config(config_path).cfg
        self.logger = get_logger()
        self.device = devices
        self.cfg['Global']['pretrained_model'] = model_path
        self.logger.info(f"Using device: {self.device}")
        self.post_process_class = build_post_process(self.cfg['PostProcess'])
        update_rec_head_out_channels(self.cfg, self.post_process_class)
        self.model = build_model(self.cfg['Architecture'])
        load_ckpt(self.model, self.cfg)
        self.model.to(self.device)
        self.model.eval()
        self.transforms = self.build_cls_process(self.cfg)
        self.ops = create_operators(self.transforms, self.cfg['Global'])

    def __call__(self, image_list):
        start_time = time.time()
        result_list = []
        self.cfg['Global']['infer_mode'] = True
        for i, img in enumerate(image_list):
            retval, buffer = cv2.imencode('.jpg', img)
            img_bytes = np.array(buffer).tobytes()
            data = {'image': img_bytes}
            batch = transform(data, self.ops)
            if batch is None:
                self.logger.info("Error in processing image")
                continue
            images = np.expand_dims(batch[0], axis=0)
            images = torch.from_numpy(images).to(self.device)
            with torch.no_grad():
                preds = self.model(images)
            post_result = self.post_process_class(preds)
            # 根据结果旋转图片
            if int(post_result[0][0]) != 0:
                img = cv2.rotate(img, cv2.ROTATE_180)
                image_list[i] = img
            result_list.append(post_result)
        return image_list, result_list, time.time() - start_time

    @staticmethod
    def build_cls_process(cfg):
        transforms = []
        for op in cfg['Eval']['dataset']['transforms']:
            op_name = list(op)[0]
            if 'Label' in op_name:
                continue
            elif op_name == 'KeepKeys':
                op[op_name]['keep_keys'] = ['image']
            elif op_name == "SSLRotateResize":
                op[op_name]["mode"] = "test"
            transforms.append(op)
        return transforms
