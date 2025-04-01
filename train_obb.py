from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v8/yolov8-obb.yaml')
    model.train(data='DOTAv1.yaml', epochs=300, batch=8)



