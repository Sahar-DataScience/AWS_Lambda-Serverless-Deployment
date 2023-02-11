#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 14:40:50 2022
@author: iovision
"""

#from urllib import response
#from Detector import Detector
import io
#from flask import Flask, render_template, request, send_from_directory, send_file , jsonify 
from PIL import Image
#import cv2 

import os
import numpy as np
import time
import torch 
import json

#detectron2 dependencies
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.config import get_cfg
from detectron2 import model_zoo
import base64
#import re

#from flask_cors import CORS
from base64 import decodestring
#import pdb 

#app = Flask(__name__)
#CORS(app)
#0 init model
#cors = CORS(app, resources={r"/predict": {"origins": "*"}})
#app.config['CORS_HEADERS'] = 'application/json'
#def __init__(self):
# set model and test set
model = 'mask_rcnn_R_50_FPN_3x.yaml'
# obtain detectron2's default config
cfg = get_cfg() 
# load values from a file
# self.cfg.merge_from_file("test.yaml")
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/"+model)) 
# set device to cpu
#cfg.MODEL.DEVICE = "cuda"
cfg.MODEL.DEVICE = "cpu"
# get weights 
cfg.MODEL.WEIGHTS = "./fold1_model_0315229.pth"
#cfg.MODEL.WEIGHTS = "/home/appuser/detectron3_repo/model_final.pth"
# set the testing threshold for this model

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

# build model from weights
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
predictor = DefaultPredictor(cfg)

def img_to_b64(img):
    file_object = io.BytesIO()
    img.save(file_object, format="JPEG")
    file_object.seek(0)
    coded_img = base64.b64encode(file_object.getvalue()).decode()
    return coded_img

def imgb64_to_pil(txt_img):
    txt_img = txt_img[txt_img.find(",")+1:]
    dec = base64.b64decode(txt_img + "===")
    pil_img = Image.open(io.BytesIO(dec))
    return pil_img



def handler(event,context):

    
  
    txt_img = event["body"]

    img = imgb64_to_pil(txt_img) #str 
    #2- predict (without save img)
    img=np.array(img)
    start=time.time()
    outputs = predictor(img)
    end=time.time()
    class_names = ['short_sleeved_shirt', 'long_sleeved_shirt', 'long_sleeved_outwear', 'shorts', 'trousers']
    MetadataCatalog.get("mydataset").thing_classes = class_names
    # visualise
    v = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get("mydataset"), scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # get image
    img_pred =Image.fromarray(np.uint8(v.get_image()[:, :, ::-1])).convert('RGB')
    data_pred ={'items_names': []}
    classes=outputs["instances"].pred_classes.to('cpu').tolist() 
    for i in range(len(classes)):
        data_pred['items_names'].append(class_names[classes[i]])
        im_b64 = img_to_b64(img_pred)
        #3- return predection (json format)
        data_tot={'response':  'data:image/png;base64,'+ im_b64}
        data_tot["pred_itm_names"] = data_pred['items_names']
        end=time.time()
        lat=end-start 
    #return jsonify(data_tot) , print("predicition latency", lat)
    #return txt_img
    return data_tot
