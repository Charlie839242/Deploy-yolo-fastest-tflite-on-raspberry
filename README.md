# Deploy-yolo-fastest-tflite-on-raspberry
**这个项目将[垃圾分类小车](https://github.com/Charlie839242/-Trash-Classification-Car/blob/main/README.md)中的tflite模型移植到了树莓派3b+上面。**  
**该项目主要是为了记录在树莓派部署yolo fastest tflite的流程**  

**(之后有时间会加上nms处理)**

关于如何在linux端运行tflite模型的问题，官方文档中已经给的非常清楚，详见[tflite.API](https://tensorflow.google.cn/lite/api_docs/python/tf/lite/Interpreter)  
  
由于yolo fastest的输出格式和其他版本的yolo不太一样，所以其yolo输出的解码模式和其他版本yolo不同，需要引起注意。若要部署的模型不是yolo fastest tflite而是其他yolo，该项目可能不能直接适用，
但根据能力进行修改即可。

## 实机效果
![image](https://github.com/Charlie839242/Deploy-yolo-fastest-tflite-on-raspberry/blob/main/img/0.gif)  

## 项目内容
本项目包含两个文件夹，***detect-camera-stream***和***detect-single-img***。  
模型文件存在两个文件夹下的tflite/Sample_TFlite_model中。  

***detect-camera-stream***文件可以在树莓派3b+连接USB摄像头的情况下，实时的用yolo-fastest-tflite模型对物体进行检测。帧数可以达到12帧。  
***detect-single-img***文件可以对tflite/下的4.jpg图片进行检测。  

关于运行的命令，存放在***instruction.txt***之中。  



## 从零部署流程  
以***detect-camera-stream***为例。  

&emsp;&emsp;**1. 创建虚拟python环境：**
创建一个tflite文件夹，创建虚拟环境：
```
cd tflite                                     :进入tflite
sudo pip3 install virtualenv                  :创建虚拟环境需要的工具
python3 -m venv tflite-env                    :创建虚拟环境，虚拟环境储存在tflite/tflite-env中
source tflite-env/bin/activate                :进入虚拟环境，每次推出terminal后都要执行此命令以进入虚拟环境
```
&emsp;&emsp;**2. 安装包和依赖：**
在进入虚拟环境后，提取出该项目中的get_pi_requirements.sh，放在tflite文件夹下：
```
bash get_pi_requirements.sh                   :下载包和依赖
```
此时可通过以下代码来测试cv2模块是否安装好(opencv-python模块经常抽风)：
```
python3
import cv2
```  
此时若没有报错则说明opencv-python安装成功，但经常出现以下错误：
```
ImportError: libjasper.so.1: cannot open shared object file: No such file or directory
```
这个报错说明少安装了依赖，执行以下命令即可：
```
sudo apt-get install libjasper-dev
```

&emsp;&emsp;**3. 在tflite文件夹下创建Sample_TFlite_model文件夹，其中存放训练好的tflite模型。**  

&emsp;&emsp;**4. 运行模型**
在tflite文件夹下，运行：
```
python3 TFLite_detection_stream.py
```
即可看到效果
### 注意：若是自己的训练的模型而不是该项目里的，需要到TFLite_detection_stream.py中修改图片分辨率等参数。




