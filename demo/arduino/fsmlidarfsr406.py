import csv
import serial
import time
from transitions import Machine
import random
import numpy as np
#需先接上 lidarfsrtest.ino
'''
f = open('C:/Users/sokel/Desktop/c/論文/lidarfsrbeddata.csv',
         'w',
         newline='',
         encoding='utf-8')
w = csv.writer(f)
'''


class Player(object):
    pass


model = Player()

states = ['lying', 'sit', 'leave']

actions = ['NYN', 'NNY', 'NNN']

transitions = [
    {
        'trigger': 'NYN',
        'source': 'sit',
        'dest': 'lying'
    },
    {
        'trigger': 'NNY',
        'source': 'lying',
        'dest': 'sit'
    },
    {
        'trigger': 'NNY',
        'source': 'leave',
        'dest': 'sit'
    },
    {
        'trigger': 'NNN',
        'source': 'sit',
        'dest': 'leave'
    },
    {
        'trigger': 'NNN',
        'source': 'leave',
        'dest': 'leave'
    },
    {
        'trigger': 'NNY',
        'source': 'sit',
        'dest': 'sit'
    },
    {
        'trigger': 'NYN',
        'source': 'leave',
        'dest': 'lying'
    },
    {
        'trigger': 'NYN',
        'source': 'lying',
        'dest': 'lying'
    },
    {
        'trigger': 'NNN',
        'source': 'lying',
        'dest': 'lying'
    },
    {
        'trigger': 'YNN',
        'source': 'lying',
        'dest': 'lying'
    },
]

machine = Machine(model=model,
                  states=states,
                  transitions=transitions,
                  initial='leave')
currents = np.array([0, 0, 0])
lyingsit = [0, 0, 1]
sitleave = [0, 0, 0]
leavesit = [0, 0, 1]
sitlying = [0, 1, 0]
'''
word = np.array([1, 0, 0])
print(word)
if (s == word).all():
    print(word)
'''

ser = serial.Serial('com7', 115200)

while True:
    data = ser.readline()
    point = data.decode('utf-8')
    #print(point)

    data = ser.readline()
    point2 = data.decode('utf-8')
    #print(point)

    data = ser.readline()
    point3 = data.decode('utf-8')

    #lidar值
    if int(point) > 200:
        currents[0] = 0
    else:
        currents[0] = 1
    #枕頭壓力值
    if int(point2) > 200:
        currents[1] = 1
    else:
        currents[1] = 0
    #床邊壓力值
    if int(point3) > 200:
        currents[2] = 1
    else:
        currents[2] = 0

    if (currents == lyingsit).all():
        #print(currents,sitlying)
        model.NNY()
    elif (currents == sitlying).all():
        model.NYN()
    elif (currents == sitleave).all():
        model.NNN()
    elif (currents == leavesit).all():
        model.NNY()

    print(point.strip(), "  ", point2.strip(), "  ", point3.strip(), " ",
          currents, "  ", model.state)

    #w.writerow([point.strip(), point2.strip(), point3.strip()])

    #print(data.decode('utf-8'))

w.writerow(['檔名', '編號'])
w.writerow(['test.jpg', '0'])
w.writerow(['train.jpg', '1'])

f.close