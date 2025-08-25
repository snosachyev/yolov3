import os
import yaml
import numpy as np

from dotenv import load_dotenv

load_dotenv()


chess_path = './chess_yolo/'
yaml_path = chess_path + 'data.yaml'

with open(yaml_path) as f:
    data_cfg = yaml.safe_load(f)

# Якорные рамки (anchor boxes) и ограничивающие рамки (bounding box)
YOLO_ANCHORS = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                         (59, 119), (116, 90), (156, 198), (373, 326)], np.float32) / 416
YOLO_ANCHOR_MASKS = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

CHECKPOINTS = os.getenv("CHECKPOINTS")
SIZE = int(os.getenv("SIZE"))
WEIGHT_YOLO_V3 = os.getenv("WEIGHT_YOLO_V3")
YOLO_SCORE_THRESHOLD = float(os.getenv("YOLO_SCORE_THRESHOLD"))
YOLO_IOU_THRESHOLD = float(os.getenv("YOLO_IOU_THRESHOLD"))
YOLO_V3_LAYERS = os.getenv("YOLO_V3_LAYERS").split(" ")
EPOCHS = int(os.getenv("EPOCHS"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
IMG_SIZE = int(os.getenv("IMG_SIZE"))
CLASS_NAMES = data_cfg["names"]
NUM_CLASSES = len(CLASS_NAMES)
VAL_PATH = chess_path + data_cfg["val"]
TRAIN_PATH = chess_path + data_cfg["train"]
