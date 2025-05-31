import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

# 训练参数官方详解链接：https://docs.ultralytics.com/modes/train/#resuming-interrupted-trainings:~:text=a%20training%20run.-,Train%20Settings,-The%20training%20settings

if __name__ == '__main__':
    model = YOLO('/home/ipmi/DeepLearning/models/ultralytics-20240822/ultralytics-main/ultralytics/cfg/models/yolov8-HFAM-LFAU-CSRN.yaml')
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
                project='runs_light/GTSDB_513/original',
                name='yolov8s-HFAM-LFAU-CSRN',
                )
