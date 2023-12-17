import json
import time
import cv2
import torch
import numpy as np
from .torchocr import Config
from .torchocr.utils.ckpt import load_ckpt
from .torchocr.utils.logging import get_logger
from .torchocr.postprocess import build_post_process
from .torchocr.data import create_operators, transform
from .torchocr.modeling.architectures import build_model
from .utility import update_rec_head_out_channels


class TextRecognizer:
    def __init__(self, config_path, model_path, character_dict_path, devices):
        self.cfg = Config(config_path).cfg
        self.cfg['Global']['character_dict_path'] = character_dict_path
        self.cfg['PostProcess']['character_dict_path'] = character_dict_path
        self.logger = get_logger()
        self.cfg['Global']['pretrained_model'] = model_path
        self.device = devices
        self.logger.info(f"Using device: {self.device}")
        self.post_process_class = build_post_process(self.cfg['PostProcess'])
        update_rec_head_out_channels(self.cfg, self.post_process_class)
        self.model = build_model(self.cfg['Architecture'])
        load_ckpt(self.model, self.cfg)
        self.model.to(self.device)
        self.model.eval()
        self.ops = create_operators(self.build_rec_process(self.cfg), self.cfg['Global'])

    def __call__(self, image_array):
        start_time = time.time()
        results = []
        for src_img in image_array:
            retval, buffer = cv2.imencode('.jpg', src_img)
            img_bytes = np.array(buffer).tobytes()
            data = {'image': img_bytes}
            batch = transform(data, self.ops)
            images, others = self.prepare_data(batch)
            preds = self.infer(images, others)
            info = self.format_result(preds)
            info = (info.split('\t')[0], float(info.split('\t')[1]))
            self.logger.info(f"Image result: {info}")
            results.append(info)
        return results, time.time() - start_time

    def prepare_data(self, batch):
        images = np.expand_dims(batch[0], axis=0)
        images = torch.from_numpy(images).to(self.device)

        others = None
        if self.cfg['Architecture']['algorithm'] == "SRN":
            others = [
                torch.from_numpy(np.expand_dims(batch[i], axis=0)).to(self.device)
                for i in range(1, 5)
            ]
        elif self.cfg['Architecture']['algorithm'] == "SAR":
            valid_ratio = torch.from_numpy(np.expand_dims(batch[1], axis=0)).to(self.device)
            others = [valid_ratio]
        elif self.cfg['Architecture']['algorithm'] == "RobustScanner":
            valid_ratio = torch.from_numpy(np.expand_dims(batch[1], axis=0)).to(self.device)
            word_positions = torch.from_numpy(np.expand_dims(batch[2], axis=0)).to(self.device)
            others = [valid_ratio, word_positions]

        return images, others

    def infer(self, images, others):
        with torch.no_grad():
            preds = self.model(images, others)
        return preds

    def format_result(self, preds):
        post_result = self.post_process_class(preds)
        if isinstance(post_result, dict):
            rec_info = {key: {"label": val[0][0], "score": float(val[0][1])}
                        for key, val in post_result.items() if len(val[0]) >= 2}
            info = json.dumps(rec_info, ensure_ascii=False)
        elif isinstance(post_result, list) and isinstance(post_result[0], int):
            info = str(post_result[0])
        else:
            if len(post_result[0]) >= 2:
                info = post_result[0][0] + "\t" + str(post_result[0][1])
        return info

    @staticmethod
    def build_rec_process(cfg):
        transforms = []
        for op in cfg['Eval']['dataset']['transforms']:
            op_name = list(op)[0]
            if 'Label' in op_name:
                continue
            elif op_name in ['RecResizeImg']:
                op[op_name]['infer_mode'] = True
            elif op_name == 'KeepKeys':
                if cfg['Architecture']['algorithm'] == "SRN":
                    op[op_name]['keep_keys'] = [
                        'image', 'encoder_word_pos', 'gsrm_word_pos',
                        'gsrm_slf_attn_bias1', 'gsrm_slf_attn_bias2'
                    ]
                elif cfg['Architecture']['algorithm'] == "SAR":
                    op[op_name]['keep_keys'] = ['image', 'valid_ratio']
                elif cfg['Architecture']['algorithm'] == "RobustScanner":
                    op[op_name][
                        'keep_keys'] = ['image', 'valid_ratio', 'word_positions']
                else:
                    op[op_name]['keep_keys'] = ['image']
            transforms.append(op)
        return transforms
