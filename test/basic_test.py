import unittest
import easypaddleocr
import numpy as np
from PIL import Image


class MyTestCase(unittest.TestCase):
    ocr = easypaddleocr.EasyPaddleOCR(use_angle_cls=True, needWarmUp=True)

    def _fuzzycmp(self, text1, text2):
        text1 = map(lambda t: t.replace(" ", ""), text1)
        text2 = map(lambda t: t.replace(" ", ""), text2)

        text1 = set(text1)
        text2 = set(text2)

        self.assertSetEqual(text1, text2, "OCR result not match")

    def _base(self, img, text: set[str]):
        img = Image.open(img)
        if img.mode != "RGB":
            img = img.convert("RGB")
        _, res, _ = self.ocr.ocr(np.array(img))
        res = set(map(lambda x: x[0], res))
        self._fuzzycmp(res, text)

    def test_basic_ocr(self):
        img = "test_images/basic.jpg"
        text = {"The quick brown fox",
                "jumps over the lazy dog",
                "Where all miracles begins.",
                "我可以吞下玻璃而不伤身体",
                "这是第二句中文"}

        self._base(img, text)

    def test_transparent_ocr(self):
        img = "test_images/transparent.png"
        text = {"This is a transparent PNG image.",
                "这是一段带有描边的文字"}

        self._base(img, text)

    def test_large_ocr(self):
        img = "test_images/large.png"
        text = {"这是一张超级大大大图片",
                "愿得一人心",
                "不离不弃到头白",
                "愿用倾城颜",
                "换爱一生不离散",
                "Was zweie ra na stel zuieg manaf",
                "1145141919810"}

        self._base(img, text)


if __name__ == '__main__':
    unittest.main()
