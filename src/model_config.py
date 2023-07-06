from  pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt
import pylab

import matplotlib.patches as patches # For bounding boxes
from PIL import Image
from collections import defaultdict
import json
import tqdm
import os 
import urllib.request


class ProblemConfig:

    # Define paths
    dataDir :str
    dataValType :str
    dataTrainType :str
    annValFile  :str
    annTrainFile:str

    IMAGES_FOLDER :str
    ANN_DIR :str
    TRAIN_DIR:str
    VAL_DIR  :str
    TEST_DIR  :str

    YOLO_DIR:str
    YOLO_SETTINGS  :str
    YOLOV3_CFG     :str
    YOLOV3_WEIGHTS :str
    YOLOV4_CFG     :str
    YOLOV4_WEIGHTS :str
    YOLOV5_NANO_ONNX:str  
    YOLOV5_SIZEX_ONNX:str 
    
         

def create_prob_config() -> ProblemConfig:
    prob_config = ProblemConfig()
    

    prob_config.dataDir = "COCOdataset2017"
    prob_config.dataValType   = 'val2017'
    prob_config.dataTrainType = 'train2017'
    prob_config.annValFile   ='{}/annotations/instances_{}.json'.format(prob_config.dataDir, prob_config.dataValType)
    prob_config.annTrainFile ='{}/annotations/instances_{}.json'.format(prob_config.dataDir, prob_config.dataTrainType)
    
    prob_config.IMAGES_FOLDER  = os.path.join(prob_config.dataDir, "images")
    prob_config.ANN_DIR   = os.path.join(prob_config.IMAGES_FOLDER, "annotations")
    prob_config.TRAIN_DIR = os.path.join(prob_config.IMAGES_FOLDER, "train")
    prob_config.VAL_DIR   = os.path.join(prob_config.IMAGES_FOLDER, "val")
    prob_config.TEST_DIR  = os.path.join(prob_config.IMAGES_FOLDER, "test")

    prob_config.YOLO_DIR          = "yolo"
    prob_config.YOLO_SETTINGS     = os.path.join(prob_config.YOLO_DIR, "yolo_settings")
    prob_config.YOLOV3_CFG        = os.path.join(prob_config.YOLO_SETTINGS, "v3/yolov3.cfg")
    prob_config.YOLOV3_WEIGHTS    = os.path.join(prob_config.YOLO_SETTINGS, "v3/yolov3.weights")
    prob_config.YOLOV4_CFG        = os.path.join(prob_config.YOLO_SETTINGS, "v4/yolov4.cfg")
    prob_config.YOLOV4_WEIGHTS    = os.path.join(prob_config.YOLO_SETTINGS, "v4/yolov4.weights")
    prob_config.YOLOV5_NANO_ONNX  = os.path.join(prob_config.YOLO_SETTINGS, "v5/yolov5n.onnx")
    prob_config.YOLOV5_SIZEX_ONNX = os.path.join(prob_config.YOLO_SETTINGS, "v5/yolov5x.onnx")

    return prob_config