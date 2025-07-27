
#include <WiFi.h>
#include <ESPmDNS.h>
#include <Wire.h> 

#include <WebSocketsServer_Generic.h>
#include "MLX90640_API.h"
#include "MLX90640_I2C_Driver.h"
#include "webpage.h"

#define SCL_PIN 5
#define SDA_PIN 4

#define SSID "Not for you"
#define PWD "smriti123456789"

#define DEFAULT_SSID_HEAD "Makerfabs_IRCamera_"



WiFiServer server(80);


WebSocketsServer webSocket = WebSocketsServer(81);

#define TA_SHIFT -64; 
static float mlx90640To[768];


char positive[27] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
char negative[27] = "abcdefghijklmnopqrstuvwxyz";

TaskHandle_t TaskA;

xQueueHandle xQueue;

int total = 0;

void setup()
{
  Serial.begin(115200);
  Serial.flush();
  
  delay(1000);

  Serial.println();
  wifi_init();


  if (!MDNS.begin("thermal"))
  {
    Serial.println("Error setting up MDNS responder!");
  }
  else
  {
    MDNS.addService("http", "tcp", 80);
    MDNS.addService("ws", "tcp", 81);
    Serial.println("mDNS responder started");
  }

  server.begin();

  xQueue = xQueueCreate(1, sizeof(mlx90640To));
  xTaskCreatePinnedToCore(
      Task1,      
      "Workload1", 
      100000,      
      NULL,      
      1,       
      &TaskA,     
      0);   

  xTaskCreate(
      receiveTask,   
      "receiveTask",
      10000,         
      NULL,          
      1,             
      NULL);        

  webSocket.begin();
  webSocket.onEvent(webSocketEvent);
}

void loop()
{
  webSocket.loop();

  WiFiClient client = server.available(); 

  if (client)
  {
    Serial.println("New Client.");
    String currentLine = "";
    while (client.connected())
    {
      if (client.available())
      {
        char c = client.read();
        Serial.write(c);
        if (c == '\n')
        {

          if (currentLine.length() == 0)
          {
         
            client.println("HTTP/1.1 200 OK");
            client.println("Content-type:text/html");
            client.println();
            client.print("<script>const ipAddress = '");
            client.print(WiFi.localIP());
            client.print("'</script>");
            client.println();
            client.print(canvas_htm);


            client.println();

            break;
          }
          else
          {
            currentLine = "";
          }
        }
        else if (c != '\r')
        {                  
          currentLine += c; 
        }
      }
    }

    client.stop();
    Serial.println("Client Disconnected.");
  }
}

void Task1(void *parameter)
{
  const byte MLX90640_address = 0x33; 

  Wire.setClock(400000L);
  Wire.begin(SDA_PIN, SCL_PIN);

  paramsMLX90640 mlx90640;
  Wire.beginTransmission((uint8_t)MLX90640_address);
  if (Wire.endTransmission() != 0)
  {
    Serial.println("MLX90640 not detected at default I2C address. Please check wiring. Freezing.");
    while (1)
      ;
  }
  Serial.println("MLX90640 online!");

  
  int status;
  uint16_t eeMLX90640[832];
  status = MLX90640_DumpEE(MLX90640_address, eeMLX90640);

  if (status != 0)
  {
    Serial.println("Failed to load system parameters");
  }
  status = MLX90640_ExtractParameters(eeMLX90640, &mlx90640);
  if (status != 0)
  {
    Serial.println("Parameter extraction failed");
  }
  MLX90640_SetRefreshRate(MLX90640_address, 0x05);
  Wire.setClock(1000000L);
  float mlx90640Background[768];
  while (1)
  {
   
    for (byte x = 0; x < 2; x++) 
    {
      uint16_t mlx90640Frame[834];
      int status = MLX90640_GetFrameData(MLX90640_address, mlx90640Frame);
      if (status < 0)
      {
        Serial.print("GetFrame Error: ");
        Serial.println(status);
      }

      float vdd = MLX90640_GetVdd(mlx90640Frame, &mlx90640);
      float Ta = MLX90640_GetTa(mlx90640Frame, &mlx90640);

      float tr = Ta - TA_SHIFT; 
      float emissivity = 0.95;

      MLX90640_CalculateTo(mlx90640Frame, &mlx90640, emissivity, tr, mlx90640Background);
    }

    const TickType_t xTicksToWait = pdMS_TO_TICKS(100);
    xQueueSendToFront(xQueue, &mlx90640Background, xTicksToWait);

    const TickType_t xDelay = 20 / portTICK_PERIOD_MS; 
    
    vTaskDelay(xDelay);
  }
}

void receiveTask(void *parameter)
{
 
  BaseType_t xStatus;
  
  const TickType_t xTicksToWait = pdMS_TO_TICKS(100);
  while (1)
  {
 
    xStatus = xQueueReceive(xQueue, &mlx90640To, xTicksToWait);
   
    if (xStatus == pdPASS)
    {
      compressAndSend();
      total += 1;
    }
  }
  vTaskDelete(NULL);
}

void webSocketEvent(uint8_t num, WStype_t type, uint8_t *payload, size_t length)
{

  switch (type)
  {
  case WStype_DISCONNECTED:
    Serial.println("Socket Disconnected.");
    break;
  case WStype_CONNECTED:
  {
    IPAddress ip = webSocket.remoteIP(num);
    Serial.println("Socket Connected.");
    
    webSocket.sendTXT(num, "Connected");
  }
  break;
  case WStype_TEXT:
   
    break;
  case WStype_BIN:
  case WStype_ERROR:
  case WStype_FRAGMENT_TEXT_START:
  case WStype_FRAGMENT_BIN_START:
  case WStype_FRAGMENT:
  case WStype_FRAGMENT_FIN:
    break;
  }
}


void compressAndSend()
{
  String resultText = "";
  int numDecimals = 1;
  int accuracy = 8;
  int previousValue = round(mlx90640To[0] * pow(10, numDecimals));
  previousValue = previousValue - (previousValue % accuracy);
  resultText.concat(numDecimals);
  resultText.concat(accuracy);
  resultText.concat(previousValue);
  resultText.concat(".");
  char currentLetter = 'A';
  char previousLetter = 'A';
  int letterCount = 1;
  int columnCount = 32;

  for (int x = 1; x < 768; x += 1)
  {
    int currentValue = round(mlx90640To[x] * pow(10, numDecimals));
    currentValue = currentValue - (currentValue % accuracy);
    if (x % columnCount == 0)
    {
      previousValue = round(mlx90640To[x - columnCount] * pow(10, numDecimals));
      previousValue = previousValue - (previousValue % accuracy);
    }
    int correction = 0;
    int diffIndex = (int)(currentValue - previousValue);
    if (abs(diffIndex) > 0)
    {
      diffIndex = diffIndex / accuracy;
    }
    if (diffIndex > 25)
    {
      
      diffIndex = 25;
    }
    else if (diffIndex < -25)
    {
     
      diffIndex = -25;
    }

    if (diffIndex >= 0)
    {
      currentLetter = positive[diffIndex];
    }
    else
    {
      currentLetter = negative[abs(diffIndex)];
    }

    if (x == 1)
    {
      previousLetter = currentLetter;
    }
    else if (currentLetter != previousLetter)
    {

      if (letterCount == 1)
      {
        resultText.concat(previousLetter);
      }
      else
      {
        resultText.concat(letterCount);
        resultText.concat(previousLetter);
      }
      previousLetter = currentLetter;
      letterCount = 1;
    }
    else
    {
      letterCount += 1;
    }

    previousValue = currentValue - correction;
  }
  if (letterCount == 1)
  {
    resultText.concat(previousLetter);
  }
  else
  {
    resultText.concat(letterCount);
    resultText.concat(previousLetter);
  }
  webSocket.broadcastTXT(resultText);
}

void wifi_init()
{
  Serial.print("Connecting to ");
  Serial.println(SSID);

  
  WiFi.begin(SSID, PWD);

  String Host_name = "";
  Host_name = Host_name + DEFAULT_SSID_HEAD + get_uid();

  WiFi.setHostname(Host_name.c_str());

  while (WiFi.status() != WL_CONNECTED)
  {
    delay(1000);
    Serial.print(".");
  }

  Serial.println("");
  Serial.println("WiFi connected.");
  Serial.println("IP address: ");
  Serial.println(WiFi.localIP());
}

void ap_init()
{

  String AP_name = "";
  AP_name = AP_name + DEFAULT_SSID_HEAD + get_uid();

  Serial.println(AP_name);
  Serial.print("Start AP ");

  WiFi.softAP(AP_name.c_str(), "12345678");
  WiFi.setHostname("esp32thing1");

  IPAddress myIP = WiFi.softAPIP();

  Serial.println("");
  Serial.println("WiFi connected.");
  Serial.println("IP address: ");
  Serial.println(myIP);
}

String get_uid()
{
 
  uint32_t chipid = 0;
  char c[20];


  for (int i = 0; i < 17; i = i + 8)
  {
    chipid |= ((ESP.getEfuseMac() >> (40 - i)) & 0xff) << i;
  }
  sprintf(c, "%08X", (uint32_t)chipid);

  return (String)c;
}