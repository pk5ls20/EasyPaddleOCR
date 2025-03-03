# EasyPaddleOCR
[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2Fpk5ls20%2FEasyPaddleOCR.svg?type=shield)](https://app.fossa.com/projects/git%2Bgithub.com%2Fpk5ls20%2FEasyPaddleOCR?ref=badge_shield)

A simple, optional tool for **PaddleOCR Detection, direction classification and recognition on CPU and GPU using torch**.

## Usage
```python
import cv2
import numpy as np
from easypaddleocr import EasyPaddleOCR

easyPaddleOCR = EasyPaddleOCR(use_angle_cls=True, needWarmUp=True)
image_path = 'your-picture-filepath'
image_ndarray = np.array(cv2.imread(image_path))
result = easyPaddleOCR.ocr(image_ndarray)
print(result)
```
## Reference
- https://github.com/WenmuZhou/PytorchOCR
- https://github.com/PaddlePaddle/PaddleOCR
- https://github.com/frotms/PaddleOCR2Pytorch

## Additional Notes:
**This repository was originally the Fork of https://github.com/WenmuZhou/PytorchOCR**


## License
[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2Fpk5ls20%2FEasyPaddleOCR.svg?type=large)](https://app.fossa.com/projects/git%2Bgithub.com%2Fpk5ls20%2FEasyPaddleOCR?ref=badge_large)