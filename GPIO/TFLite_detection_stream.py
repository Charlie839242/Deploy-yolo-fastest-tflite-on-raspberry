import os
import RPi.GPIO as GPIO
import argparse
import cv2
import time
import numpy as np
import serial
import sys
import time
from threading import Thread
import importlib.util
from tflite_runtime.interpreter import Interpreter

from yolo_layer import yolo_decode, img_pre, draw_img, find_max_key

GPIO.cleanup()
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(5, GPIO.IN, pull_up_down=GPIO.PUD_UP)

ser = serial.Serial('/dev/ttyAMA0',115200)                             # 设置串口

cost_time = 0                                                          # 设置是否查看每个函数耗时

anchor = [[19,30], [46,37], [23,94], [64,67], [116,96]]

MODEL_NAME = 'Sample_TFlite_model'
GRAPH_NAME = 'detect.tflite'

CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

interpreter = Interpreter(model_path=PATH_TO_CKPT)                      # 载入模型

interpreter.allocate_tensors()

# 获取模型的详细数据
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 计算FPS
FPS = 1
freq = cv2.getTickFrequency()

# 获取摄像头
camera = cv2.VideoCapture(0)

while True:
    x = GPIO.input(5)
    if(x == 1):
        print('camera exit')
        sys.exit(0)

    # 开始计时
    t1 = cv2.getTickCount()

    if(cost_time):
        t_start = cv2.getTickCount()
    _,img = camera.read()                                                                   # 获取一帧图像
    if (cost_time):
        t_end = cv2.getTickCount()
        print('The cost time for the funtion "camera.read" is ', (t_end - t_start) / freq)

    if(cost_time):
        t_start = cv2.getTickCount()
    input_data = img_pre(img)                                                               # 图像预处理
    if(cost_time):
        t_end = cv2.getTickCount()
        print('The cost time for the funtion "img_pre" is ',  (t_end - t_start)/freq)

    if (cost_time):
        t_start = cv2.getTickCount()
    interpreter.set_tensor(input_details[0]['index'],input_data)                            # 设置模型输入
    if (cost_time):
        t_end = cv2.getTickCount()
        print('The cost time for the funtion "interpreter.set_tensor" is ', (t_end - t_start) / freq)

    if (cost_time):
        t_start = cv2.getTickCount()
    interpreter.invoke()                                                                    # 运行模型
    if (cost_time):
        t_end = cv2.getTickCount()
        print('The cost time for the funtion "interpreter.invoke" is ', (t_end - t_start) / freq)

    if (cost_time):
        t_start = cv2.getTickCount()
    output_data = interpreter.get_tensor(output_details[0]['index'])                        # 获取输出
    if (cost_time):
        t_end = cv2.getTickCount()
        print('The cost time for the funtion "interpreter.get_tensor" is ', (t_end - t_start) / freq)

    if (cost_time):
        t_start = cv2.getTickCount()
    pred_xywh, objectness, class_scores = yolo_decode(output_data, anchor, num_classes=3,
                                                      input_dims=(160, 160),
                                                      use_softmax=False)                    # 输出解码
    
    if (cost_time):
        t_end = cv2.getTickCount()
        print('The cost time for the funtion "yolo_decode" is ', (t_end - t_start) / freq)


    if (cost_time):
        t_start = cv2.getTickCount()
    max_key = find_max_key(objectness)
    if (cost_time):
        t_end = cv2.getTickCount()
        print('The cost time for the funtion "find_max_key" is ', (t_end - t_start) / freq)

    if(max_key == 0):
        print('no object detected')
        #cv2.imshow('results', img)
        if cv2.waitKey(1) == ord('q'):
            break
        continue

    cla = np.argmax(class_scores[max_key])
    print('class = ', cla)

    if (cost_time):
        t_start = cv2.getTickCount()
    draw_img(pred_xywh[max_key], img, ser, cla)
    if (cost_time):
        t_end = cv2.getTickCount()
        print('The cost time for the funtion "draw_img" is ', (t_end - t_start) / freq)

    # show FPS
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
