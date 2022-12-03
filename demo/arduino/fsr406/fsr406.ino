//From the article: http://bildr.org/2012/11/force-sensitive-resistor-arduino

int FSR_Pin = A1; //analog pin 0
int FSR_Pin2 = A0;
void setup(){
Serial.begin(115200);
}

void loop(){

int state = 0;
int FSRReading = analogRead(FSR_Pin);

int FSRReading2 = analogRead(FSR_Pin2);

if(FSRReading > 400)
{
  state = 2;
}

if(FSRReading2 > 400)
{
  state = 1;
}


if(FSRReading < 400 && FSRReading2 < 400)
{
  state = 0;
}



Serial.println(state );
delay(25); //just here to slow down the output for easier reading
}
