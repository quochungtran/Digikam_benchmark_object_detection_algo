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

from  model_config import ProblemConfig, create_prob_config


# Download images 
def download_images(imagesJson, img_folder):

    if (not os.path.exists(img_folder)):
        os.makedirs(img_folder)
    for im in tqdm.tqdm_notebook(imagesJson):
            im_path = os.path.join(img_folder, im['file_name'])
            if not os.path.exists(im_path):
                    urllib.request.urlretrieve(im['coco_url'], im_path)

def getImages(coco, subCats):

    # Specify classes to download
    catIds      = coco.getCatIds(catNms=subCats)

    # get image ids
    imgIds = coco.getImgIds(catIds=catIds)

    # Load images ids
    images = coco.loadImgs(imgIds)

    return catIds, imgIds, images 


if __name__ == "__main__":


    model_config = create_prob_config()

    # init COCO api for instance annotations 

    cocoTrain = COCO(model_config.annTrainFile)
    cocoVal   = COCO(model_config.annValFile)
    
    # todo fix after
    subCats = ['person', 'car', 'bicycle']
    catTrainIds, imgTrainIds, trainImages = getImages(cocoTrain, subCats)
    catValIds, imgValIds, valImages       = getImages(cocoVal, subCats)

    print("----------- Begin to download model ---------------")
    # download img
    download_images(trainImages, model_config.TRAIN_DIR)
    download_images(valImages, model_config.VAL_DIR)

    print("the number of training   data set images : ", len(os.listdir(model_config.TRAIN_DIR)))
    print("the number of validation data set images : ", len(os.listdir(model_config.VAL_DIR)))