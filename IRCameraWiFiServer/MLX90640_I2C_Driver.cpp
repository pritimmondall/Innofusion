#include <Wire.h>
#include "MLX90640_I2C_Driver.h"

void MLX90640_I2CInit()
{
}

int MLX90640_I2CRead(uint8_t _deviceAddress, unsigned int startAddress, unsigned int nWordsRead, uint16_t *data)
{
  uint16_t bytesRemaining = nWordsRead * 2;
  uint16_t dataSpot = 0;

  while (bytesRemaining > 0)
  {
    Wire.beginTransmission(_deviceAddress);
    Wire.write(startAddress >> 8);
    Wire.write(startAddress & 0xFF);
    if (Wire.endTransmission(false) != 0)
    {
      return (0);
    }

    uint16_t numberOfBytesToRead = bytesRemaining;
    if (numberOfBytesToRead > I2C_BUFFER_LENGTH) numberOfBytesToRead = I2C_BUFFER_LENGTH;

    Wire.requestFrom((uint8_t)_deviceAddress, numberOfBytesToRead);
    if (Wire.available())
    {
      for (uint16_t x = 0 ; x < numberOfBytesToRead / 2; x++)
      {
        data[dataSpot] = Wire.read() << 8;
        data[dataSpot] |= Wire.read();
        dataSpot++;
      }
    }

    bytesRemaining -= numberOfBytesToRead;
    startAddress += numberOfBytesToRead / 2;
  }

  return (0);
}

void MLX90640_I2CFreqSet(int freq)
{
  Wire.setClock((long)1000 * freq);
}

int MLX90640_I2CWrite(uint8_t _deviceAddress, unsigned int writeAddress, uint16_t data)
{
  Wire.beginTransmission((uint8_t)_deviceAddress);
  Wire.write(writeAddress >> 8);
  Wire.write(writeAddress & 0xFF);
  Wire.write(data >> 8);
  Wire.write(data & 0xFF);
  if (Wire.endTransmission() != 0)
  {
    return (-1);
  }

  uint16_t dataCheck;
  MLX90640_I2CRead(_deviceAddress, writeAddress, 1, &dataCheck);
  if (dataCheck != data)
  {
    return -2;
  }

  return (0);
}
