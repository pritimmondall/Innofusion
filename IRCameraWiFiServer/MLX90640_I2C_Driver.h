#ifndef _MLX90640_I2C_Driver_H_
#define _MLX90640_I2C_Driver_H_

#include <stdint.h>

#if defined(__AVR_ATmega328P__) || defined(__AVR_ATmega168__)
#define I2C_BUFFER_LENGTH BUFFER_LENGTH

#elif defined(__SAMD21G18A__)
#define I2C_BUFFER_LENGTH SERIAL_BUFFER_SIZE

#elif __MK20DX256__
#define I2C_BUFFER_LENGTH 32

#else
#define I2C_BUFFER_LENGTH 32

#endif

void MLX90640_I2CInit(void);
int MLX90640_I2CRead(uint8_t slaveAddr, unsigned int startAddress, unsigned int nWordsRead, uint16_t *data);
int MLX90640_I2CWrite(uint8_t slaveAddr, unsigned int writeAddress, uint16_t data);
void MLX90640_I2CFreqSet(int freq);

#endif
