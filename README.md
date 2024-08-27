## LFCA-YOLO: A Local Feature and Context Attention based Insulator Defect Detection Network


## Setup

### Getting Started
Installation (to install pytorch cf. https://pytorch.org/get-started/locally/):
```shell
conda create -n npsr python=3.11
conda activate npsr
pip install torch torchvision torchaudio
```

## Training


usage:
```shell
python trian.py
```

## Visualization

usage:
```shell
python val.py
```
## Datasets

### INSULATOR FAULTSDETECTION  dataset
You can get the INSULATOR FAULTSDETECTION dataset by filling out the form at:
https://universe.roboflow.com/project-vmgqx/insulator-faults-detection

### VOC dataset
vim trian.py:
```shell
    model = YOLO('ultralytics/cfg/models/LFCA-YOLO.yaml')
    init_seeds(seed=1, deterministic=True)
    model.train(data='ultralytics/cfg/VOC.yaml',
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
```

### SFID dataset
Dataset downloadable at:
https://ihepbox.ihep.ac.cn/ihepbox/index.php/s/adTHe1UPu0Vc7vI/download

