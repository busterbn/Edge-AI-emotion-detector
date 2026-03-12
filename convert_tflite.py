import os
import numpy as np
import tensorflow as tf
from tf_keras.preprocessing.image import ImageDataGenerator

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'tiny_model.h5')
TFLITE_PATH = os.path.join(SCRIPT_DIR, 'tiny_model.tflite')
HEADER_PATH = os.path.join(SCRIPT_DIR, 'firmware', 'main', 'model_data.h')
DATASET_PATH = os.path.join(SCRIPT_DIR, '..', '..', 'input',
                             'face-expression-recognition-dataset', 'images')

PICTURE_SIZE = 48
NUM_CALIBRATION_SAMPLES = 200


def representative_dataset_gen():
    datagen = ImageDataGenerator(rescale=1.0 / 255)
    gen = datagen.flow_from_directory(
        os.path.join(DATASET_PATH, 'train'),
        target_size=(PICTURE_SIZE, PICTURE_SIZE),
        color_mode='grayscale',
        batch_size=1,
        class_mode=None,
        shuffle=True,
    )
    for i in range(NUM_CALIBRATION_SAMPLES):
        yield [next(gen).astype(np.float32)]


def convert_to_tflite():
    import tf_keras
    print(f"Loading model from {MODEL_PATH}")
    model = tf_keras.models.load_model(MODEL_PATH)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    print("Converting to int8 TFLite...")
    tflite_model = converter.convert()

    with open(TFLITE_PATH, 'wb') as f:
        f.write(tflite_model)
    print(f"Saved TFLite model: {TFLITE_PATH} ({len(tflite_model):,} bytes)")
    return tflite_model


def generate_c_header(tflite_model):
    print(f"\nGenerating C header: {HEADER_PATH}")
    os.makedirs(os.path.dirname(HEADER_PATH), exist_ok=True)

    with open(HEADER_PATH, 'w') as f:
        f.write('#ifndef MODEL_DATA_H\n')
        f.write('#define MODEL_DATA_H\n\n')
        f.write(f'const unsigned int model_data_len = {len(tflite_model)};\n')
        f.write('alignas(16) const unsigned char model_data[] = {\n')
        for i, byte in enumerate(tflite_model):
            if i % 12 == 0:
                f.write('  ')
            f.write(f'0x{byte:02x},')
            if i % 12 == 11:
                f.write('\n')
            else:
                f.write(' ')
        f.write('\n};\n\n')
        f.write('#endif  // MODEL_DATA_H\n')

    print(f"  Header size: {os.path.getsize(HEADER_PATH):,} bytes")


if __name__ == '__main__':
    tflite_model = convert_to_tflite()
    generate_c_header(tflite_model)
