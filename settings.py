import os
import yaml
import numpy as np
import tensorflow as tf

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
PREFETCH = int(os.getenv("PREFETCH", tf.data.AUTOTUNE))
IMG_SIZE = int(os.getenv("IMG_SIZE"))
CLASS_NAMES = data_cfg["names"]
COCO_CLASS_NAMES = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana","apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake","chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop",
                "mouse","remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator","book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

NUM_CLASSES = len(CLASS_NAMES)
VAL_PATH = chess_path + data_cfg["val"]
TRAIN_PATH = chess_path + data_cfg["train"]
