

#!/usr/bin/python

def search(state):
    while True:
        #datas=fb.get("/students",None)
        no=3
        temp = random.random()
        if no =="": break
        key_id = checkkey(int(no))
        if key_id != "":
            print("原來type:{}".format(datas[key_id]["type"]))
            #name=input("請輸入姓名:")
            name=datas[key_id]["name"]
            data = {"no":int(no),"name":name,"type":state}
            datas[key_id]=data
            fb.put(url + "/students/",data=data,name=key_id)
            print("已修改完畢\n")
            break
        else:
            print("未建立\n".format(no))
            break

def checkkey(no):
    key_id=""
    if datas != None:
        for key in datas:
            if no==datas[key]["no"]:
                key_id = key
                break
    return key_id


#print("123")
'''
print("123")
'''
#import serial  # 引用pySerial模組
#from tkinter import *
#import time
 
#COM_PORT = 'COM7'    # 指定通訊埠名稱
#BAUD_RATES = 115200    # 設定傳輸速率
#ser = serial.Serial(COM_PORT, BAUD_RATES)   # 初始化序列通訊埠

#ser = serial.Serial('/dev/ttyACM0',115200)

import serial  # 引用pySerial模組
from tkinter import *
import time
import random


ser = serial.Serial('/dev/ttyACM0',115200)

from firebase import firebase
students = [{"no":1,"name":"李天龍"},
{"no":2,"name":"李天"},
{"no":3,"name":"李"}]

url="https://superb-reporter-313014-default-rtdb.firebaseio.com/"
fb =firebase.FirebaseApplication(url,None)

datas=fb.get("/students",None)

num=0
 
'''
tk=Tk()
tk.title('人員狀態')
canvas=Canvas(tk,width=500,height=500)
canvas.pack()
itext=canvas.create_text(250,250,font=('標楷體',40),text="12")
'''

'''
for student in students:
    fb.post("/students",student)
    print("{} 儲存完畢".format(student))
'''

'''
while True:
    current = random.random()
    search(current)
'''

#刪除
'''
while True:
    datas=fb.get("/students",None)
    no=1
    if no =="": break
    key_id = checkkey(int(no))
    if key_id != "":
        print("確定刪除".format(datas[key_id]["name"]))
        fb.delete("/students/"+key_id,None)
        print("確定刪除完畢\n")
    else:
        print("未建立\n".format(no))
        break
'''
#修改


'''
users = fb.get('students',None)
for key in users:
    print(users[key]['no'])
    num = users[key]['no']
'''

'''

tk=Tk()
canvas=Canvas(tk,width=500,height=500)
canvas.pack()
itext=canvas.create_text(250,250,text=str(num))

if num==3:
    canvas.configure(bg='green')
    canvas.itemconfig(itext,text='躺在床上')

canvas.insert(itext,12,'') 
tk.update()
tk.after(10)
tk.mainloop()


'''



num=0
'''
tk=Tk()
tk.title('人員狀態')
canvas=Canvas(tk,width=500,height=500)
canvas.pack()
itext=canvas.create_text(250,250,font=('標楷體',40),text="12")
'''

current=0

now=0
#print("ewegfe")
try:
    while True:
        #print("erg")
        while ser.inWaiting():          # 若收到序列資料…
            data_raw = ser.readline()  # 讀取一行
            state = data_raw.decode(encoding='unicode_escape')   # 用預設的UTF-8解碼
            print('接收到的原始資料：', data_raw)
            print('接收到的資料：', state)
	    
            #current = state
            #print(type(state))
            #canvas.configure(bg='red')
            #canvas.itemconfig(itext,text='目前狀態:離開床')
            if current == 0:
                now = 0
                #search(1)
                #canvas.configure(bg='red')
                #canvas.itemconfig(itext,text='目前狀態:離開床')

            elif current == 1:
                now = 1
                #search(2)
                #canvas.configure(bg='blue')
                #canvas.itemconfig(itext,text='目前狀態:坐在床邊')

            elif current == 2:
                now = 2
                #search(3)
                #canvas.configure(bg='green')
                #canvas.itemconfig(itext,text='目前狀態:躺在床上')
    #current=random.random()        
        search(int(state))
            #canvas.insert(itext,12,'') 
            #tk.update()
            #tk.after(10)
            


except KeyboardInterrupt:
    ser.close()    # 清除序列通訊物件
    print('再見！')
