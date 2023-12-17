# from .infer_system import TextSystem
# import types
#
#
# class EasyPaddleOCR(TextSystem):
#     def __init__(self):
#         arg = types.SimpleNamespace()
#         # --det_model_dir=output/det/h_PP-OCRv4_hgnet/export
#         # --det_model_config=C:/Users/pk5ls/Desktop/PytorchOCR/configs/det/ch_PP-OCRv4/ch_PP-OCRv4_det_teacher.yml
#         # --rec_model_dir=output/rec/rec_ppocr_v4_hgnet/export
#         # --rec_model_config=C:/Users/pk5ls/Desktop/PytorchOCR/configs/rec/PP-OCRv4/ch_PP-OCRv4_rec_hgnet.yml
#         # --cls_model_dir=output/cls/mv3/export
#         # --cls_model_config=C:/Users/pk5ls/Desktop/PytorchOCR/configs/cls/cls_mv3.yml
#         # --use_gpu=True
#         # --image_dir="C:/Users/pk5ls/Desktop/Ori"
#         # --use_angle_cls=true
#         arg.det_model_name = 'models/det/ch_PP-OCRv4_det_server_train/best_accuracy.pth'
#         arg.det_model_config = 'configs/det/ch_PP-OCRv4/ch_PP-OCRv4_det_teacher.yml'
#         arg.rec_model_name = 'models/det/ch_PP-OCRv4_rec_server_train/best_accuracy.pth'
#         arg.rec_model_config = 'configs/rec/PP-OCRv4/ch_PP-OCRv4_rec_hgnet.yml'
#         arg.cls_model_name = 'models/ch_ppocr_mobile_v2.0_cls_train/best_accuracy.pth'
#         arg.cls_model_config = 'configs/cls/cls_mv3.yml'
#         arg.use_gpu = True
#         arg.image_dir = 'img/'
#         arg.use_angle_cls = True
#         arg.show_log = None
#
#         super().__init__(arg)
