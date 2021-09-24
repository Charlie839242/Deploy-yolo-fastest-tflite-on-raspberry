# Deploy-yolo-fastest-tflite-on-raspberry
## 觉得有用的话可以顺手点个star嗷
**这个项目将[垃圾分类小车](https://github.com/Charlie839242/-Trash-Classification-Car/blob/main/README.md)中的tflite模型移植到了树莓派3b+上面。**  
**该项目主要是为了记录在树莓派部署yolo fastest tflite的流程**  

**(之后有时间会尝试用C++部署来提升性能)**

## 一些问题
### 1. 如何运行tflite文件？  
关于如何在linux端运行tflite模型的问题，官方文档中已经给的非常清楚，详见[tflite.API](https://tensorflow.google.cn/lite/api_docs/python/tf/lite/Interpreter)  
### 2. yolo-fastest的解码问题？  
由于yolo fastest的输出格式和其他版本的yolo不太一样，所以其yolo输出的解码模式和其他版本yolo不同，需要引起注意。若要部署的模型不是yolo fastest tflite而是其他yolo，该项目可能不能直接适用，
但根据能力进行修改即可。
### 3.yolo-fastest的源码：[Yolo-Fastest](https://github.com/dog-qiuqiu/Yolo-Fastest)  
现在yolo fastest的作者推出了V2版本，性能更好。该项目采用的是V1.  
### 4.关于如何在windows上训练yolo fastest模型，详见本人另一个仓库：[Yolo-Fastest-on-Windows](https://github.com/Charlie839242/YOLO-Fastest-on-a-no-gpu-windows-computer)  


## 模型实机效果
![image](https://github.com/Charlie839242/Deploy-yolo-fastest-tflite-on-raspberry/blob/main/img/0.gif)    
该项目在树莓派3b+上可以跑到平均25帧每秒。  
![image](https://github.com/Charlie839242/Deploy-yolo-fastest-tflite-on-raspberry/blob/main/img/FPS_on_3b+.jpg) 

## 小车实机效果  
![image](https://github.com/Charlie839242/Deploy-yolo-fastest-tflite-on-raspberry/blob/main/img/demo_1.gif) 

## 项目内容
本项目包含两个文件夹，***detect-camera-stream***和***detect-single-img***。  
两个文件夹中结构相同，模型文件存在两个文件夹下的tflite/Sample_TFlite_model中，主程序写在TFLite_detection_stream.py和TFLite_detection_img.py里，yolo相关的函数写在yolo_layer.py中。    

***detect-camera-stream***文件可以在树莓派3b+连接USB摄像头的情况下，实时的对视频流进行目标检测。  
***detect-single-img***文件可以对tflite/下的4.jpg图片，即单独一张图片进行检测。  

关于运行的命令，存放在***instruction.txt***之中。  

## 如何直接运行该项目:
&emsp;&emsp;**1. 确保树莓派上有python3.7解释器。**  

&emsp;&emsp;**2. 安装virtualenv：**
```
python3 -m venv tflite-env 
```
&emsp;&emsp;**3. 下载该项目所有文件。**  
&emsp;&emsp;**4. 进入tflite文件夹，进入虚拟python环境：**
```
source tflite-env/bin/activate
bash get_pi_requirements.sh                  :若上一步提示缺少环境则执行这一行
```
&emsp;&emsp;**5. 在tflite文件夹下，运行instruction.txt中的指令：**
```
python3 TFLite_detection_image.py
python3 TFLite_detection_stream.py
```


## 从零部署流程  
&emsp;&emsp;以***detect-camera-stream***为例。  

&emsp;&emsp;**1. 创建虚拟python环境：**  

&emsp;&emsp;创建一个tflite文件夹，创建虚拟环境：
```
cd tflite                                     :进入tflite
sudo pip3 install virtualenv                  :创建虚拟环境需要的工具
python3 -m venv tflite-env                    :创建虚拟环境，虚拟环境储存在tflite/tflite-env中
source tflite-env/bin/activate                :进入虚拟环境，每次推出terminal后都要执行此命令以进入虚拟环境
```
&emsp;&emsp;**2. 安装包和依赖：**  

&emsp;&emsp;在进入虚拟环境后，提取出该项目中的get_pi_requirements.sh，放在tflite文件夹下：
```
bash get_pi_requirements.sh                   :下载包和依赖
```
&emsp;&emsp;此时可通过以下代码来测试cv2模块是否安装好(opencv-python模块经常抽风)：
```
python3
import cv2
```  
&emsp;&emsp;此时若没有报错则说明opencv-python安装成功，但经常出现以下错误：
```
ImportError: libjasper.so.1: cannot open shared object file: No such file or directory
```
&emsp;&emsp;这个报错说明少安装了依赖，执行以下命令即可：(我是这样解决的，若解决不了请百度)
```
sudo apt-get install libjasper-dev
```

&emsp;&emsp;**3. 在tflite文件夹下创建Sample_TFlite_model文件夹，其中存放训练好的tflite模型。**  

&emsp;&emsp;**4. 运行模型**
在tflite文件夹下，运行：
```
python3 TFLite_detection_stream.py
```
&emsp;&emsp;即可看到效果
#### 注意：若是自己的训练的模型而不是该项目里的，需要到TFLite_detection_stream.py中修改图片分辨率等参数。  


## 由于树莓派要和小车通信，因此这里在记录一下在树莓派用AMA0实现串口通信的过程。  
首先安装gedit编辑器，比vim好用一些：  
```
sudo apt-get install gedit
```
然后禁用串口启动，开启串口硬件：  
```
sudo raspi-config
interfacing options --> would you like a login shell to be accessible  over serial? --> No
                    --> would you like the serial port hardware to be enabled? --> Yes
```
由于蓝牙和AMA0使用的是同一个GPIO，将ttyAMA0和ttyS0的映射调换：
```
sudo gedit /boot/config.txt
在最后一行添加：dtoverlay=pi3-miniuart-bt
sudo reboot
```
因为控制台使用串口和通信串口只能存在一个，所以要禁用控制台来使用串口：
```
sudo systemctl stop serial-getty@ttyAMA0.service
sudo systemctl disable serial-getty@ttyAMA0.service
```
然后删除serial0相关：
```
sudo gedit /boot/cmdline.txt
删除 console=serial0,115200 ，没有就不管
sudo reboot
```
至此串口设置就完了，因为树莓派的python3解释器自带serial库，但我们之前创建的虚拟环境没有，所以要在虚拟环境再次安装：
```
sudo pip3 install pyserial
sudo pip3 install serial
```
可以通过以下代码来控制串口：
```
import serial
ser = serial.Serial('/dev/ttyAMA0',115200)      # 获取串口
if(ser.isOpen):
  ser.write(b'123')                             # 出现编码问题可以尝试加上 .encode()
```

## 通过开关来控制识别的开始和结束 
### 这一部分的文件在该仓库的GPIO文件夹中可找到。  
***由于通过ssh连接树莓派比较复杂，且每次运行程序都需要电脑在手边，因此若能通过树莓派自身来控制程序的跑与结束，是最方便不过的了。***  
***因此，我选择用一个按键开关来控制***  
![image](https://github.com/Charlie839242/Deploy-yolo-fastest-tflite-on-raspberry/blob/main/img/switch.jpg)  
这是一个双刀双掷开关，这里只用其中两个引脚。  
&emsp;&emsp;**1. 写一个脚本来实现启动py文件：**  
&emsp;&emsp;在/home/pi目录下编写charlie.sh文件：  
```
cd /home/pi/Desktop/demo1/tflite
source tflite-env/bin/activate
python3 TFLite_detection_stream.py
```
&emsp;&emsp;此时通过命令行输入bash /home/pi/charlie.sh即可运行py文件。  
&emsp;&emsp;**2. 连线：**  
&emsp;&emsp;将树莓派的3，5引脚连到开关的一段，GND连接到另一端。  
&emsp;&emsp;这样，在初始化时将3，5拉高。当开关按下时，3被拉低，可以此作为启动程序的标志。当开关被松开后，5被拉高，可以此作为退出程序的标志。  
&emsp;&emsp;**3. 编写GPIO.py:**  
&emsp;&emsp;首先在虚拟环境中安装RPi库：
```
pip3 install RPi.GPIO
```
&emsp;&emsp;在/home/pi/Desktop/demo1/tflite的目录下编写GPIO.py,使当引脚3被拉低后运行charlie.sh：  
```
import time
import RPi.GPIO as GPIO
import os

run_yolo_cmd = 'bash /home/pi/charlie.sh'

GPIO.setmode(GPIO.BOARD)
GPIO.setup(3, GPIO.IN, pull_up_down=GPIO.PUD_UP)

while(True):
    while(True):
        x = GPIO.input(3)
        if(x == 0):
            break

    print('pressed')
    time.sleep(1)
    os.system(run_yolo_cmd)
    

    while(True):
        x = GPIO.input(3)
        if (x == 1):
            break
    print('not pressed')
    time.sleep(1)
```
注意，time.sleep(1)是必要的，因为按键在按下和松开时，电压是不稳定的，延时可以消抖。  
&emsp;&emsp;**4. 修改TFLite_detection_stream.py:**    
&emsp;&emsp;修改TFLite_detection_stream.py以使得其拥有检测到引脚5升高后自动结束运行的功能：  
 ```
 在TFLite_detection_stream中作如下添加：
 import RPi.GPIO as GPIO
 
 GPIO.setmode(GPIO.BOARD)
 GPIO.setup(5, GPIO.IN, pull_up_down=GPIO.PUD_UP)
 
 在进行模型推理的大循环中添加：
 x = GPIO.input(5)
 if(x == 1):
     print('camera exit')
     sys.exit(0)
 ```
 &emsp;&emsp;**5. 实现开机自动运行GPIO.py**  
 ```
 sudo gedit /etc/rc.local
 在exit 0的上一行添加：
 python3 /home/pi/Desktop/demo1/tflite/GPIO.py &
 (&符号使得其一直在后台运行)
 ```
 至此，开机后会自动运行GPIO.py,GPIO.py会不停检测引脚3。当按下引脚3后，GPIO.py会调用charlie.py来运行TFLite_detection_stream.py。TFLite_detection_stream.py会检测引脚5，当按键松开后，TFLite_detection_stream.py会自动退出。这是一个循环。再按下会在启动......  
 
 











