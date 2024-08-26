import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':




    model = YOLO('runs/train/exp/weights/best.pt')

    model.val(data='dataset/data.yaml',
              split='val',
              imgsz=640,
              batch=32,
              save_json=True,
              device='0',
              project='runs/val',
              name='exp',
              )