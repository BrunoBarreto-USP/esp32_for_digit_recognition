/************************************************************************************************
 * PROJETO: RECONHECIMENTO DE DÍGITOS - VERSÃO FINAL COM CNN E TENSORFLOW LITE
 * - Usa a biblioteca oficial TensorFlow Lite para rodar o modelo CNN.
 * - Modelo é quantizado (usa int8), o que exige que o buffer de imagem seja int8.
 * - Inclui pré-processamento de imagem completo.
 * - Ambiente: VS Code + PlatformIO + Wokwi Simulator
 ************************************************************************************************/

#include <Arduino.h>
#include <SPI.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_ILI9341.h>
#include <LiquidCrystal_I2C.h>
#include <Adafruit_FT6206.h>
#include <TensorFlowLite_ESP32.h>

// Includes da biblioteca TensorFlow Lite
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Nosso modelo, como um array de bytes C++
#include "model_data.h"

// --- DEFINIÇÕES ---
#define TFT_CS 5
#define TFT_DC 4
#define TFT_RST 2
#define IMG_WIDTH 28
#define IMG_HEIGHT 28
#define MAX_POINTS 250

// --- VARIÁVEIS GLOBAIS PARA O TFLITE ---
tflite::ErrorReporter *error_reporter = nullptr;
const tflite::Model *model = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *input = nullptr;
TfLiteTensor *output = nullptr;

// A CNN precisa de mais memória.
constexpr int kTensorArenaSize = 30 * 1024; // 30KB
uint8_t tensor_arena[kTensorArenaSize];

// --- VARIÁVEIS DO PROJETO ---
Adafruit_ILI9341 tft(TFT_CS, TFT_DC, TFT_RST);
LiquidCrystal_I2C lcd(0x27, 16, 2);
Adafruit_FT6206 ts;
struct Point
{
  int16_t x;
  int16_t y;
};
Point captured_points[MAX_POINTS];
int point_count = 0;
// O buffer de imagem agora é int8 para combinar com o modelo quantizado
int8_t image_buffer[IMG_WIDTH * IMG_HEIGHT];
bool is_drawing = false;

// Protótipos das funções
void drawInitialUI();
void processDrawing();
int perform_real_inference();
void preprocessImage();
void drawLineOnBuffer(int x0, int y0, int x1, int y1, int8_t *buffer);

// --- SETUP ---
void setup()
{
  Serial.begin(115200);
  lcd.init();
  lcd.backlight();
  lcd.print("Iniciando TFLite...");

  // Configuração do TensorFlow Lite
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // g_model é o nome padrão do array no model_data.h gerado pelo xxd
  model = tflite::GetModel(mnist_model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION)
  {
    error_reporter->Report("Modelo incompativel!");
    return;
  }

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk)
  {
    error_reporter->Report("Falha ao alocar tensores!");
    return;
  }
  input = interpreter->input(0);
  output = interpreter->output(0);

  if (!ts.begin(40))
  {
    Serial.println("Falha no touch!");
    while (1)
      ;
  }

  tft.begin();
  tft.setRotation(1);
  drawInitialUI();
}

// --- LOOP PRINCIPAL ---
void loop()
{
  if (ts.touched())
  {
    if (!is_drawing)
    {
      is_drawing = true;
      point_count = 0;
      memset(captured_points, 0, sizeof(captured_points));
      tft.fillRect(10, 10, 220, 220, 0x0000);
    }
    TS_Point p = ts.getPoint(0);
    int16_t x = map(p.y, 0, 240, 0, 320);
    int16_t y = map(p.x, 0, 320, 240, 0);
    if (x > 10 && x < 230 && y > 10 && y < 230 && point_count < MAX_POINTS)
    {
      tft.fillCircle(x, y, 4, 0xFFFF);
      captured_points[point_count] = {x, y};
      point_count++;
    }
  }
  else
  {
    if (is_drawing)
    {
      is_drawing = false;
      if (point_count > 10)
      {
        processDrawing();
      }
      drawInitialUI();
    }
  }
  delay(20);
}



// --- FUNÇÕES DO PROJETO ---
void processDrawing()
{
  lcd.clear();
  lcd.print("Processando...");
  preprocessImage();
  int predicted_digit = perform_real_inference();
  lcd.clear();
  lcd.print("Digito: ");
  lcd.print(predicted_digit);
  delay(3000);
}

int perform_real_inference()
{
  // Copia os dados do nosso buffer para o tensor de entrada do modelo
  for (int i = 0; i < IMG_WIDTH * IMG_HEIGHT; i++)
  {
    input->data.int8[i] = image_buffer[i];
  }

  if (interpreter->Invoke() != kTfLiteOk)
  {
    error_reporter->Report("Falha na Inferencia!");
    return -1;
  }

  // Encontra o dígito com a maior pontuação na saída
  int8_t best_score = -128;
  int predicted_digit = -1;
  for (int i = 0; i < 10; i++)
  {
    if (output->data.int8[i] > best_score)
    {
      best_score = output->data.int8[i];
      predicted_digit = i;
    }
  }
  return predicted_digit;
}

void drawInitialUI()
{
  tft.fillScreen(0x0000);
  tft.drawRect(10, 10, 220, 220, 0x001F);
  tft.setCursor(240, 20);
  tft.setTextColor(0xFFFF);
  tft.setTextSize(2);
  tft.println("Desenhe");
  tft.setCursor(240, 40);
  tft.println("aqui");
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Aguardando um");
  lcd.setCursor(0, 1);
  lcd.print("digito...");
}

void drawLineOnBuffer(int x0, int y0, int x1, int y1, int8_t *buffer)
{
  int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
  int dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
  int err = dx + dy, e2;
  for (;;)
  {
    for (int i = 0; i < 2; i++)
    {
      for (int j = 0; j < 2; j++)
      {
        int px = x0 + i;
        int py = y0 + j;
        if (px >= 0 && px < IMG_WIDTH && py >= 0 && py < IMG_HEIGHT)
        {
          buffer[py * IMG_WIDTH + px] = 127; // 127 representa "branco" para o modelo int8
        }
      }
    }
    if (x0 == x1 && y0 == y1)
      break;
    e2 = 2 * err;
    if (e2 >= dy)
    {
      err += dy;
      x0 += sx;
    }
    if (e2 <= dx)
    {
      err += dx;
      y0 += sy;
    }
  }
}

void preprocessImage()
{
  // -128 representa "preto" para o modelo int8
  for (int i = 0; i < IMG_WIDTH * IMG_HEIGHT; i++)
    image_buffer[i] = -128;
  if (point_count < 2)
    return;

  int8_t temp_buffer[IMG_WIDTH * IMG_HEIGHT];
  for (int i = 0; i < IMG_WIDTH * IMG_HEIGHT; i++)
    temp_buffer[i] = -128;

  int16_t minX = 32767, minY = 32767, maxX = -1, maxY = -1;
  for (int i = 0; i < point_count; i++)
  {
    minX = min(minX, captured_points[i].x);
    maxX = max(maxX, captured_points[i].x);
    minY = min(minY, captured_points[i].y);
    maxY = max(maxY, captured_points[i].y);
  }

  if (maxX == minX || maxY == minY)
    return;
  float scale = min((float)(IMG_WIDTH - 8) / (maxX - minX), (float)(IMG_HEIGHT - 8) / (maxY - minY));

  for (int i = 1; i < point_count; i++)
  {
    int16_t x0 = (captured_points[i - 1].x - minX) * scale;
    int16_t y0 = (captured_points[i - 1].y - minY) * scale;
    int16_t x1 = (captured_points[i].x - minX) * scale;
    int16_t y1 = (captured_points[i].y - minY) * scale;
    drawLineOnBuffer(x0, y0, x1, y1, temp_buffer);
  }

  float sumX = 0, sumY = 0, totalMass = 0;
  for (int y = 0; y < IMG_HEIGHT; y++)
  {
    for (int x = 0; x < IMG_WIDTH; x++)
    {
      if (temp_buffer[y * IMG_WIDTH + x] > -128)
      {
        sumX += x;
        sumY += y;
        totalMass++;
      }
    }
  }

  if (totalMass == 0)
    return;

  int comX = sumX / totalMass;
  int comY = sumY / totalMass;
  int shiftX = (IMG_WIDTH / 2) - comX;
  int shiftY = (IMG_HEIGHT / 2) - comY;

  for (int y = 0; y < IMG_HEIGHT; y++)
  {
    for (int x = 0; x < IMG_WIDTH; x++)
    {
      if (temp_buffer[y * IMG_WIDTH + x] > -128)
      {
        int newX = x + shiftX;
        int newY = y + shiftY;
        if (newX >= 0 && newX < IMG_WIDTH && newY >= 0 && newY < IMG_HEIGHT)
        {
          image_buffer[newY * IMG_WIDTH + newX] = 127;
        }
      }
    }
  }
}