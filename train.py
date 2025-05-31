import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/yolov8-HFAM-LFAU-CSRN.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data='data/GTSDB_513.yaml',
                cache=False,
                imgsz=640,
                epochs=400,
                batch=32,
                close_mosaic=0,
                workers=4,
                device='0',
                # optimizer='SGD', # using SGD
                # patience=0, # close earlystop
                # resume=True, 
                # amp=False, # close amp
                # fraction=0.2,
                project='runs',
                name='project_name',
                )
