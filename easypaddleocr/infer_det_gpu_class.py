import numpy as np
import torch
from torchocr.data import create_operators, transform
from torchocr.modeling.architectures import build_model
from torchocr.postprocess import build_post_process
from torchocr.utils.ckpt import load_ckpt
from torchocr.utils.logging import get_logger
from torchocr import Config
from utility import ArgsParser


class TextDetector:
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
        pre_process_list = [{
            'DetResizeForTest': {
                'limit_side_len': 960, # args.det_limit_side_len
                'limit_type': "max",  # args.det_limit_type
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
        self.ops = create_operators(pre_process_list) # 挪用predict_det的预处理
        # self.ops = create_operators(build_det_process(self.cfg), self.cfg['Global'])
        # 之前的create_operators(build_det_process(self.cfg), self.cfg['Global']) 的预处理，只是上面的'KeepKeys'吗？
        self.post_process_class = build_post_process(self.cfg['PostProcess'])

    def __call__(self, ndarray_image):
        # retval, buffer = cv2.imencode('.jpg', ndarray_image)
        # img_bytes = np.array(buffer).tobytes()
        # data = {'image': img_bytes}
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
        print(111)
        return boxes, 114514 # 先返回个114514，本来这是时间

        # boxes = sorted_boxes(np.array(boxes))
        # img_crop_list = [get_minarea_rect_crop(src_img, box) for box in boxes]
        #
        # return img_crop_list, 114514 # 先返回个114514，本来这是时间


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


if __name__ == '__main__':
    FLAGS = ArgsParser().parse_args()
    ocr_detector = TextDetector(FLAGS.config)
    FLAGS = vars(FLAGS)
    opt = FLAGS.pop('opt')
    ocr_detector.cfg.merge_dict(FLAGS)
    ocr_detector.cfg.merge_dict(opt)

