import cv2
import numpy as np
import serial

def yolo_decode(prediction, anchors, num_classes, input_dims, use_softmax=False):
    num_anchors = len(anchors)  # anchor 的数量
    grid_size = prediction.shape[1:3]  # 将一张图片分割成5*5

    # shape: (125, 6)
    prediction = np.reshape(prediction,
                            (grid_size[0] * grid_size[1] * num_anchors, num_classes + 5))

    # generate x_y_offset grid map
    x_y_offset = [[[j, i]] * grid_size[0] for i in range(grid_size[0]) for j in range(grid_size[0])]
    x_y_offset = np.array(x_y_offset).reshape(grid_size[0] * grid_size[1] * num_anchors , 2)

    x_y_tmp = 1 / (1 + np.exp(-prediction[..., :2]))
    box_xy = (x_y_tmp + x_y_offset) / np.array(grid_size)[::-1]

    # Log space transform of the height and width
    anchors2 = np.array(anchors*(grid_size[0] * grid_size[1]))
    box_wh = (np.exp(prediction[..., 2:4]) * anchors2) / np.array(input_dims)[::-1]

    # sigmoid function
    objectness = 1 / (1 + np.exp(-prediction[..., 4:5]))

    # sigmoid function
    if use_softmax:
        class_scores = np.exp(prediction[..., 5:]) / np.sum(np.exp(prediction[..., 5:]))
    else:
        class_scores = 1 / (1 + np.exp(-prediction[..., 5:]))

    return np.concatenate((box_xy, box_wh), axis=-1), objectness, class_scores

def img_pre(read_img):
    img = cv2.cvtColor(read_img, cv2.COLOR_BGR2GRAY)                        # 得到灰度图
    img = cv2.resize(img, (160, 160), interpolation=cv2.INTER_LINEAR)       # 双线性插值得到160*160尺寸
    img = img / 255.0                                                       # 归一化
    input_data = np.asarray(img).astype('float32')
    input_data = np.expand_dims(input_data, axis=0)
    input_data = np.expand_dims(input_data, axis=3)                         # 扩充两个维度，得到1*160*160*1
    return input_data

def draw_img(boxes, img, ser):
    # show img
    x, y, w, h = boxes
    x = x * 640
    y = y * 480
    w = w * 640
    h = h * 480
    ser.write(('#' + str(x) + '*' + str(cla) + '&').encode())
    xmin = int(x - w / 2)
    xmax = int(x + w / 2)
    ymin = int(y - h / 2)
    ymax = int(y + h / 2)
    print(xmin, ymin, xmax, ymax)
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    cv2.imshow('results', img)

def find_max_key(objectness):
    max_key = objectness.argmax(axis = 0)[0]
    if(objectness[max_key][0] > 0.9):
        return max_key
    else:
        return None
