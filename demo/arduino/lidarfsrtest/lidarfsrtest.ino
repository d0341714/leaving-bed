//From the article: http://bildr.org/2012/11/force-sensitive-resistor-arduino
#include <DFRobot_TFmini.h>

SoftwareSerial mySerial(8, 7); // RX, TX

DFRobot_TFmini  TFmini;
int distance,strength;

int FSR_Pin = A1; //analog pin 0
int FSR_Pin2 = A0;
void setup(){
Serial.begin(115200);
TFmini.begin(mySerial);
}

void loop(){
int FSRReading = analogRead(FSR_Pin);
int FSRReading2 = analogRead(FSR_Pin2);

if(TFmini.measure()){                      //Measure Distance and get signal strength     
        strength = TFmini.getStrength();       //Get signal strength data
        distance = TFmini.getDistance();
        //Serial.print("Strength = ");
   
        //Serial.println(strength); 
        //timer = millis();

        //Serial.println(timer); 
        Serial.println(distance); 
        Serial.println(FSRReading );
        Serial.println(FSRReading2);
     
       delay(25);
   }
delay(25); //just here to slow down the output for easier reading
}
