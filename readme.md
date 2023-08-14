install dependencies

check you have python 3.8

```bash
python -V
```

```bash
sudo apt-get install python3.8-dev
python -m venv venv
pip install --upgrade pip
pip install --upgrade setuptools
pip install cython
pip install torch
pip install torchvision
pip install scikit-build
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
```
Can skip???
git clone https://github.com/open-mmlab/mmdetection.git

clone mmdetection repository and install dependencies
```bash
cd mmdetection
pip install -e .
```

verify installation, should create output image in outputs/vis/demo.jpg
```bash
mim download mmdet --config rtmdet_tiny_8xb32-300e_coco --dest .
python demo/image_demo.py demo/demo.jpg rtmdet_tiny_8xb32-300e_coco.py --weights rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth --device cpu
```
