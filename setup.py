from setuptools import setup, find_packages

setup(
    name='easypaddleocr',
    version='0.1.1',
    packages=find_packages(),
    install_requires=[
        "shapely",
        "scikit-image",
        "imgaug",
        "pyclipper",
        "numpy",
        "opencv-python",
        "Pillow",
        "pyyaml",
        "torch",
        "loguru",
        "huggingface_hub"
    ],
    author="pk5ls20, hv0905"
)
