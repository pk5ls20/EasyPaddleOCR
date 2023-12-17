import time
import torch
import numpy as np
from .torchocr import Config
from .torchocr.utils.ckpt import load_ckpt
from .torchocr.utils.logging import get_logger
from .torchocr.postprocess import build_post_process
from .torchocr.data import create_operators, transform
from .torchocr.modeling.architectures import build_model


class TextDetector:
    def __init__(self, config_path, model_path, devices):
        self.cfg = Config(config_path).cfg
        self.logger = get_logger()
        self.cfg['Global']['pretrained_model'] = model_path
        self.device = devices
        self.logger.info(f"Using device: {self.device}")
        self.model = build_model(self.cfg['Architecture'])
        load_ckpt(self.model, self.cfg)
        self.model.to(self.device)
        self.model.eval()
        pre_process_list = [{
            'DetResizeForTest': {
                'limit_side_len': 960,
                'limit_type': "max",
            }
        }, {
            'NormalizeImage': {
                'std': [0.229, 0.224, 0.225],
                'mean': [0.485, 0.456, 0.406],
                'scale': '1./255.',
                'order': 'hwc'
            }
        }, {
            'ToCHWImage': None
        }, {
            'KeepKeys': {
                'keep_keys': ['image', 'shape']
            }
        }]
        self.ops = create_operators(pre_process_list)  # 挪用predict_det的预处理
        self.post_process_class = build_post_process(self.cfg['PostProcess'])

    def __call__(self, ndarray_image):
        start_time = time.time()
        data = {'image': ndarray_image}
        batch = transform(data, self.ops)
        images = np.expand_dims(batch[0], axis=0)
        shape_list = np.expand_dims(batch[1], axis=0)
        images = torch.from_numpy(images).to(self.device)
        with torch.no_grad():
            preds = self.model(images)
        post_result = self.post_process_class(preds, [-1, shape_list])
        # 转换精度， 似乎是int32 -> float32
        if isinstance(post_result, dict):
            boxes = [np.array(box, dtype=np.float32) for sublist in post_result.values() for box in
                     sublist[0]['points']]
        else:
            boxes = np.array(post_result[0]['points'], dtype=np.float32)
        return boxes, time.time() - start_time

    @staticmethod
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
