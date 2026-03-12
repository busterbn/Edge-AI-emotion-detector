# Emotion Recognition on ESP32-S3

Real-time facial emotion recognition running on an ESP32-S3 with camera. A lightweight CNN is trained on the FER2013 dataset, quantized to int8 TFLite, and deployed via TFLite Micro.

## Project Structure

```
my_project/
├── train_tiny_model.py      # Train the tiny CNN (TensorFlow/Keras)
├── convert_tflite.py        # Convert .h5 → int8 .tflite → C header
├── requirements.txt         # Python dependencies
├── tiny_model.h5            # Trained Keras model (generated)
├── tiny_model.tflite        # Quantized TFLite model (generated)
└── firmware/                # ESP-IDF project
    ├── CMakeLists.txt
    ├── sdkconfig.defaults   # ESP32-S3 config (PSRAM, 240MHz, camera)
    └── main/
        ├── main.c           # App entry: capture → preprocess → infer → print
        ├── camera.c/h       # ESP32-CAM driver init/capture
        ├── preprocess.c/h   # Crop + resize to 48x48 grayscale int8
        ├── inference.cpp/h  # TFLite Micro interpreter setup + inference
        ├── model_data.h     # Model weights as C array (generated)
        ├── idf_component.yml # Managed deps (esp32-camera, esp-tflite-micro)
        └── CMakeLists.txt
```

## Pipeline

```
FER2013 dataset → train_tiny_model.py → tiny_model.h5
                                            ↓
                  convert_tflite.py → tiny_model.tflite → model_data.h
                                                              ↓
                  idf.py build → firmware binary → ESP32-S3
```

## Prerequisites

- Python 3.11+
- [ESP-IDF v5.5+](https://docs.espressif.com/projects/esp-idf/en/latest/esp32s3/get-started/)
- FER2013 dataset at `dataset/images/` (relative to this directory), with `train/` and `validation/` subdirectories containing class folders (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise)
- ESP32-S3 board with camera module and PSRAM

## Setup

### 1. Install Python dependencies

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 2. Download the dataset

The model expects the FER2013 dataset. Download it using the Kaggle CLI (included in requirements):

```bash
kaggle datasets download -d jonathanoheix/face-expression-recognition-dataset
unzip face-expression-recognition-dataset.zip -d dataset
rm face-expression-recognition-dataset.zip
```

This creates the expected directory structure:
```
dataset/images/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── validation/
    └── (same subdirectories)
```

> **Note:** You need a Kaggle account and API token (`~/.kaggle/kaggle.json`). See [Kaggle API docs](https://github.com/Kaggle/kaggle-api#api-credentials).

### 3. Train the model

```bash
python train_tiny_model.py
```

This trains a ~21K parameter CNN with data augmentation. Outputs `tiny_model.h5` with best validation accuracy (~53%).

### 4. Convert to TFLite and generate C header

```bash
python convert_tflite.py
```

This:
- Loads `tiny_model.h5`
- Quantizes to int8 using calibration images from the training set
- Saves `tiny_model.tflite` (~39KB)
- Generates `firmware/main/model_data.h` (model weights as C byte array)

### 5. Build and flash firmware

```bash
deactivate
cd firmware
source ~/esp/esp-idf/export.sh
idf.py set-target esp32s3
idf.py build
idf.py flash monitor
```

## Model Architecture

| Layer | Output Shape | Params |
|-------|-------------|--------|
| Conv2D 16, 3x3 + BN + ReLU + MaxPool | 24x24x16 | 224 |
| SeparableConv2D 32, 3x3 + BN + ReLU + MaxPool | 12x12x32 | 816 |
| SeparableConv2D 64, 3x3 + BN + ReLU + MaxPool | 6x6x64 | 2,464 |
| SeparableConv2D 128, 3x3 + BN + ReLU + GAP | 128 | 9,152 |
| Dense 64 + ReLU + Dropout | 64 | 8,256 |
| Dense 7 (softmax) | 7 | 455 |
| **Total** | | **~21,800** |

## Emotion Classes

0: Angry, 1: Disgust, 2: Fear, 3: Happy, 4: Neutral, 5: Sad, 6: Surprise

## Hardware

Configured for ESP32-S3 with:
- 240 MHz CPU
- Octal PSRAM at 80 MHz
- 64KB data cache with 64-byte lines
- Camera on core 1
