install dependencies


```bash
pip install torch
pip install torchvision
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
```

clone mmdetection repository and install dependencies
```bash
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -e .
```

verify installation, should create output image in outputs/vis/demo.jpg
```bash
mim download mmdet --config rtmdet_tiny_8xb32-300e_coco --dest .
python demo/image_demo.py demo/demo.jpg rtmdet_tiny_8xb32-300e_coco.py --weights rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth --device cpu
```