#include <DFRobot_TFmini.h>

SoftwareSerial mySerial(8, 7); // RX, TX

DFRobot_TFmini  TFmini;
int distance,strength;


void setup(){
    Serial.begin(115200);
    TFmini.begin(mySerial);
     
}
int timer;
void loop(){
    
    if(TFmini.measure()){                      //Measure Distance and get signal strength     
        strength = TFmini.getStrength();       //Get signal strength data
        distance = TFmini.getDistance();
        //Serial.print("Strength = ");
   
        //Serial.println(strength); 
timer = millis();

        //Serial.println(timer); 
        Serial.println(distance); 
     
       delay(25);
   }
}
