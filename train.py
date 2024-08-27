import warnings

import torch

from ultralytics.utils.torch_utils import init_seeds

warnings.filterwarnings('ignore')
from ultralytics import YOLO





if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/LFCA-YOLO.yaml')
    init_seeds(seed=1, deterministic=True)
    model.train(data='dataset/data.yaml',
                cache=False,
                imgsz=640,
                epochs=200,
                batch=32,
                close_mosaic=10,
                workers=8,
                device='0',
                optimizer='SGD',
                project='runs/train',
                name='exp',
                seed=1
                )