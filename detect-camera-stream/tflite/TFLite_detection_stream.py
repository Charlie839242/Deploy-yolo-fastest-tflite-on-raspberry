import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
from tflite_runtime.interpreter import Interpreter

from yolo_layer import yolo_decode, img_pre, draw_img

anchor = [[15,21], [18,61], [47,46], [27,108], [97,82]]

MODEL_NAME = 'Sample_TFlite_model'
GRAPH_NAME = 'detect.tflite'

CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# FPS calculation
FPS = 1
freq = cv2.getTickFrequency()

# 获取摄像头
camera = cv2.VideoCapture(0)

while True:

    # 开始计时
    t1 = cv2.getTickCount()

    # 获取一帧图像
    _,img = camera.read()

    # 图像预处理
    input_data = img_pre(img)

    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()                                                    # 运行模型

    output_data = interpreter.get_tensor(output_details[0]['index'])        # 获取输出
    pred_xywh, objectness, class_scores = yolo_decode(output_data, anchor, num_classes=3,
                                                      input_dims=(160, 160),
                                                      use_softmax=False)    # 输出解码

    i = 0
    max_key = 0
    max = objectness[0][0]
    while (i < 125):
        if ((objectness[i][0] > max) and (objectness[i][0] > 0.9)):
            max = objectness[i][0]
            max_key = i
        i = i + 1
    if(max_key == 0):
        print('no object detected')
        cv2.imshow('results', img)
        
        if cv2.waitKey(1) == ord('q'):
            break
        continue

    print(max_key, 'th box is the box with biggest objetness', max)

    cla = np.argmax(class_scores[max_key])
    print('class = ', cla)

    draw_img(pred_xywh[max_key], img)

    t2 = cv2.getTickCount()
    time = (t2-t1)/freq
    FPS= 1/time
    print('FPS = ',FPS)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
camera.release()
