'''
Firebase是可以即時讀取的資料庫，主要以JSON格式為主。除了透過程式操作外，也能在Firebase的網頁界面上進行資料操作(上傳、讀取、修改、刪除)
'''

###程式1###

#引入模組
from firebase import firebase
import time
from tkinter import *
import serial


def agent():
    camera = [0, 0, 0]
    state = 82
    ser = serial.Serial('COM8', 9600)
    print(ser.name)
    print("Port Open")

    #ser.write(str('Y').encode())
    time.sleep(2)
    while ser.isOpen():
        for i in range(3):
            #ser.write(b'5')
            #ser.write(chr(5))
            #ser.write(5)
            #ser.write(state)
            ser.write(str('S').encode())
            ser.write(str('R').encode())
            print(camera)
            camera[1] = camera[1] + 1
        ser.write(str('F').encode())
        break
    ser.close()


#產生資料
new_users = [{
    'name': 'Richard Ho',
    'type': 1
}, {
    'name': 'Tom Wu',
    'type': 2
}, {
    'name': 'Judy Chen',
    'type': 3
}, {
    'name': 'Lisa Chang',
    'type': 4
}]

#連接資料庫(由於設定任何人都能存取，所以不需要設定其他的API Key)
db_url = 'https://superb-reporter-313014-default-rtdb.firebaseio.com/'
fdb = firebase.FirebaseApplication(db_url, None)
'''
#在user下建立四筆資料(來自new_users)(.post新增)
for data in new_users:			
	fdb.post('/user', data)   
	time.sleep(3)
'''
#在user下查詢新增的資料(.get讀取)
tk = Tk()
canvas = Canvas(tk, width=500, height=500)
canvas.pack()
itext = canvas.create_text(250, 250, font=('標楷體', 40), text="12")

#canvas.configure(bg='white')
#canvas.itemconfig(itext, text='躺在床上')
num = 0

while True:
    users = fdb.get('/students', None)  #None全部讀取，1代表讀取第一筆，以此類推
    print("資料庫中找到以下的使用者")
    for key in users:
        print(users[key]['type'])
        agentstate = 0
        prestate = num
        #time.sleep(2)
        num = users[key]['type']
        if prestate == 1:
            if num == 0:
                agentstate = 1
        if num == 0:
            canvas.configure(bg='red')
            canvas.itemconfig(itext, text='人員狀態:離開床')
            #if agentstate == 1:
            #agent()
        elif num == 1:
            canvas.configure(bg='blue')
            canvas.itemconfig(itext, text='人員狀態:坐床邊')
        elif num == 2:
            canvas.configure(bg='green')
            canvas.itemconfig(itext, text='人員狀態:躺在床上')
    #canvas.insert(itext, 12, '')
    tk.update()
    tk.after(10)
    #tk.mainloop()

#整理：使用到的方法Method
#result = fdb.post('/user', user)  #建立(C)
#result = fdb.get('/user',None)    #讀取(R)
#result = fdb.delete('/user',None) #刪除(D)
