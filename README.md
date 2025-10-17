Overview
--------
Real-time handwritten digit recognition (0–9) on ESP32 using PlatformIO.
The firmware captures touch strokes on a TFT ILI9341 with FT6206 touch,
preprocesses to 28x28, and runs an int8 neural network (TensorFlow Lite
Micro). The code can also use a tiny pure C++ MLP if desired.

Hardware
--------
- ESP32 DevKit (board: esp32dev)
- TFT ILI9341 (SPI), default pins in code: TFT_CS=5, TFT_DC=4, TFT_RST=2
- FT6206 capacitive touch (I2C on default SDA/SCL)
- (Optional) 16x2 I2C LCD for status (addr 0x27)

Key Libraries (PlatformIO)
--------------------------
- Adafruit GFX
- Adafruit ILI9341
- Adafruit FT6206 Library
- LiquidCrystal_I2C (optional)
- TensorFlow Lite for Microcontrollers (ESP32 build)  OR  a pure C++ MLP

Project Structure (typical)
---------------------------
- src/main.cpp            : App entry; drawing, preprocessing, inference, serial output
- include/model_data.h    : Embedded TFLite model array (int8) named g_model
- include/*.h             : Optional pure C++ classifiers (e.g., DigitClassifier/PlainCppClassifier)
- platformio.ini          : Board/env configuration and library deps
- (optional) scripts/     : Training & conversion utilities for your MLP/CNN

How It Works
------------
1) Touch strokes are sampled and rasterized into a grayscale buffer.
2) The digit ROI is found via bounding box; image is scaled to 28x28 with aspect preserved.
3) The 28x28 is centered using center-of-mass shifting.
4) Data is normalized/quantized to match the model input (int8 for TFLM).
5) Inference outputs 10 probabilities and latency (microseconds). Top-k is printed over Serial.

Quick Start (Build & Flash)
---------------------------
1) Open the project in VS Code with the PlatformIO extension.
2) Confirm in platformio.ini:
     board = esp32dev
     monitor_speed = 115200
3) Wire the display/touch per your board; adjust pin defines if needed.
4) Build -> Upload; then open Serial Monitor at 115200.
5) Draw a digit on the screen; observe top-k predictions and a JSON line with probabilities/latency.

Using a Tiny MLP (Two Options)
------------------------------
A) TFLite-Micro path:
   - Train a small MLP (e.g., 784->128->64->10) on MNIST in Python/Keras.
   - Convert to TFLite int8 (incl. representative dataset for quantization).
   - Export the .tflite as a C array (xxd -i or script) with symbol g_model.
   - Replace include/model_data.h with the new array and rebuild.

B) Pure C++ path:
   - Use the provided MLP header (e.g., DigitClassifier.h).
   - Feed a normalized float[784] (0..1) to predict().
   - Remove/disable TFLM init calls in main.cpp and print the MLP outputs.

Notes & Tips
------------
- Keep models fully int8 for best RAM/flash usage and speed on ESP32.
- If TFLM initialization fails, increase the tensor arena size in code or reduce model size.
- If predictions look off, verify preprocessing matches what you used at training time.
- For different wiring, update pin defines for ILI9341 and verify FT6206 I2C address.
