//From the article: http://bildr.org/2012/11/force-sensitive-resistor-arduino
#include <DFRobot_TFmini.h>

SoftwareSerial mySerial(8, 7); // RX, TX

DFRobot_TFmini  TFmini;

#define pinX  2
const int onTime=1000; // in ms 
int distance,strength;
int FSR_Pin = A0; //analog pin 0
int FSR_Pin2 = A1; //analog pin 0
int FSRReading = 0;
int FSRReading2=0;
int state = 0;
int change = 0;
int flag = 1;
int value = 0;
int interval=3000;
int count = 0;
int second2 = 0;

unsigned long start=millis();
unsigned long delay2=3000;

unsigned long starttime; 
unsigned long endtime=0; 


void setup(){
Serial.begin(115200);
pinMode(pinX, INPUT);
TFmini.begin(mySerial);
}

void loop(){

delay(25);
//value = digitalRead(pinX);

//FSRReading床邊位置FSR406壓力感測器 FSRReading2枕頭位置FSR406壓力感測器
FSRReading = analogRead(FSR_Pin);
FSRReading2 = analogRead(FSR_Pin2);


if(TFmini.measure()){                      //Measure Distance and get signal strength     
    strength = TFmini.getStrength();       //Get signal strength data
    distance = TFmini.getDistance();
    //Serial.print("Strength = "); 
    //Serial.println(strength); 
    //Serial.println(distance); 
    }
	
//坐著 change姿態轉變 flag 避免lidar偵測離床到躺在床上
    if(FSRReading > 700 && FSRReading2 < 700)
    {
      //delay(500);
      state=1;
      change = 0;
      flag = 1;    
      //count=0;
      endtime=millis();
      second2=0;          
      start=millis();
    }

    else if(FSRReading < 700)
    {
      //delay(500);
      distance = TFmini.getDistance();
        
      //  時間三秒內壓力感測器小於700時 lidar距離要小於200 視為躺在床上 
      if (millis()-start>delay2 && second2 == 0)
      {                  
          second2=1;
      }
          if(second2 == 0)
          {
            if(distance < 200)
            {
              flag = 2;             
            }           
          }                             
      //Serial.println(flag);
//離開床
      //Serial.println(distance); 
      if(distance>200 && change == 0)
      {
        state=0;
        starttime=millis();
        //count = 0;

      }
//躺上床 床的距離約200 lidar掃描小於200
      else if(distance<200 && flag == 2)
      {
            change = 1;
            state = 2;          
      }

//躺上床 床的距離約200 lidar掃描大於200
      else if(distance > 200 && change == 1 )
      {
       state = 2;
      }


//當狀態為躺上床 LIDAR距離大於200是離開床
      if(state == 2 && FSRReading2 < 700)
      {
        if(distance < 200)
        {
          state = 2;
        }

        else if(distance > 200)
        {
          state = 0;
        }             
      }
    }

    if(FSRReading2 > 500)
    {
      state = 2;      
    }


    

    if(state==0)
    {
      //Serial.println("不在床上");
      Serial.println(state);
    }

    else if(state==1)
    {
      //Serial.println("坐在床邊");
      Serial.println(state);
    }

    else if(state==2)
    {
      //Serial.println("躺在床上");
      Serial.println(state);
    }

    
delay(25); //just here to slow down the output for easier reading
}


 /*
 * JoyStick
 * 双轴按键摇杆
 */
