import csv
import serial
import time
from transitions import Machine
import random
import numpy as np
# 引入 time 模組
import time

# 從 1970/1/1 00:00:00 至今的秒數

# 將秒數轉為本地時間

f = open('C:/Users/sokel/Desktop/c/論文/testcsv/lidarfsrbeddatatest8.csv',
         'w',
         newline='',
         encoding='utf-8')
w = csv.writer(f)
'''
word = np.array([1, 0, 0])
print(word)
if (s == word).all():
    print(word)
'''

ser = serial.Serial('com7', 115200)
'''
while True:
    data = ser.readline()
    point = data.decode('utf-8')
    #print(point)

    data = ser.readline()
    point2 = data.decode('utf-8')
    #print(point)

    data = ser.readline()
    point3 = data.decode('utf-8')

    print(point.strip(), "  ", point2.strip(), "  ", point3.strip(), " ")

    w.writerow([point.strip(), point2.strip(), point3.strip()])

    #print(data.decode('utf-8'))

w.writerow(['檔名', '編號'])
w.writerow(['test.jpg', '0'])
w.writerow(['train.jpg', '1'])

f.close
'''

change = 0
#坐著 change姿態轉變 flag 避免lidar偵測離床到躺在床上
while True:

    seconds = time.time()
    local_time = time.ctime(seconds)
    data = ser.readline()
    point = data.decode('utf-8')
    distance = int(point)
    data = ser.readline()
    point2 = data.decode('utf-8')
    FSRReading2 = int(point2)
    data = ser.readline()
    point3 = data.decode('utf-8')
    FSRReading = int(point3)
    if FSRReading > 700 and FSRReading2 < 700:
        state = 1
        change = 0
        flag = 1

    elif FSRReading < 700:
        #時間三秒內壓力感測器小於700時 lidar距離要小於200 視為躺在床上
        if distance > 200 and change == 0:
            state = 0

#躺上床 床的距離約200 lidar掃描小於200
        elif distance < 200:
            change = 1
            state = 2

#躺上床 床的距離約200 lidar掃描大於200
        elif distance > 200 and change == 1:
            state = 2

        if state == 2 and FSRReading2 < 700:
            if distance < 200:
                state = 2

            elif distance > 200 and change == 0:
                state = 0

    if FSRReading2 > 700:
        state = 2

    if state == 0:
        statenum = '0'
    elif state == 1:
        statenum = '1'
    elif state == 2:
        statenum = '2'
    '''
    if FSRReading2 > 400:
        FSRReading2 = FSRReading2 - 400
    if FSRReading > 400:
        FSRReading = FSRReading - 400
    '''
    print(local_time, distance, FSRReading2, FSRReading, statenum)
    w.writerow([local_time, distance, FSRReading2, FSRReading, statenum])
