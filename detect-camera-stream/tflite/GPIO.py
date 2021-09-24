import time
import RPi.GPIO as GPIO
import os

run_yolo = 'bash /home/pi/charlie.sh'
stop_yolo = 'killall -9 python3'

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(3, GPIO.IN, pull_up_down=GPIO.PUD_UP)

while(True):
    while(True):
        x = GPIO.input(3)
        if(x == 0):
            break

    print('pressed')
    time.sleep(1)
    os.system(run_yolo)
    

    while(True):
        x = GPIO.input(3)
        if (x == 1):
            break
    print('not pressed')
    time.sleep(1)

