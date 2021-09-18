import os
import argparse
import cv2
import numpy as np
import sys
import glob
import importlib.util
from tflite_runtime.interpreter import Interpreter

from yolo_layer import yolo_decode, img_pre, draw_img

anchor = [[15,21], [18,61], [47,46], [27,108], [97,82]]

MODEL_NAME = 'Sample_TFlite_model'
GRAPH_NAME = 'detect.tflite'
IM_NAME = '4.jpg'

# Get path to current working directory
CWD_PATH = os.getcwd()

PATH_TO_IMAGES = os.path.join(CWD_PATH,IM_NAME)
images = glob.glob(PATH_TO_IMAGES)                                          # 得到图片绝对地址

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Loop over every image and perform detection
for image_path in images:

    input_data, img_raw = img_pre(image_path)                                   # 图像预处理

    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()                                                        # 运行模型

    output_data = interpreter.get_tensor(output_details[0]['index'])            # 获取输出
    print(output_data.shape)

    pred_xywh, objectness, class_scores = yolo_decode(output_data, anchor, num_classes=3,
                                                      input_dims=(160, 160),
                                                      use_softmax=False)        # yolo_decode
    print(pred_xywh.shape, objectness.shape, class_scores.shape)

    i = 0
    max_key = 0
    max = objectness[0][0]
    while(i < 125):
        if((objectness[i][0] > max) and (objectness[i][0] > 0.9)):
            max = objectness[i][0]
            max_key = i
        i = i + 1

    if(max_key == 0):
        print('no object detected')
        break;

    print(max_key, 'th box is the box with biggest objetness', max)

    cla = np.argmax(class_scores[max_key])
    print(cla)
    
    pred_xywh = pred_xywh * 160

    draw_img(pred_xywh[max_key], img_raw)
