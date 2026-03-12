import os
import numpy as np
import tensorflow as tf
import tf_keras
from tf_keras.preprocessing.image import ImageDataGenerator
from tf_keras.layers import (
    Input, Conv2D, SeparableConv2D, BatchNormalization, ReLU,
    MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout
)
from tf_keras.models import Model
from tf_keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPT_DIR, 'dataset', 'images')
TEACHER_PATH = os.path.join(SCRIPT_DIR, '..', 'model.h5')
OUTPUT_PATH = os.path.join(SCRIPT_DIR, 'tiny_model.h5')

PICTURE_SIZE = 48
BATCH_SIZE = 128
EPOCHS = 60
NUM_CLASSES = 7
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


def build_tiny_model():
    inp = Input(shape=(PICTURE_SIZE, PICTURE_SIZE, 1))

    x = Conv2D(16, (3, 3), padding='same')(inp)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = SeparableConv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = SeparableConv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = SeparableConv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = GlobalAveragePooling2D()(x)

    x = Dense(64)(x)
    x = ReLU()(x)
    x = Dropout(0.3)(x)
    out = Dense(NUM_CLASSES, activation='softmax')(x)

    return Model(inputs=inp, outputs=out)


if __name__ == '__main__':
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPUs: {tf.config.list_physical_devices('GPU')}")

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
    )
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_set = train_datagen.flow_from_directory(
        os.path.join(DATASET_PATH, 'train'),
        target_size=(PICTURE_SIZE, PICTURE_SIZE),
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
    )
    val_set = val_datagen.flow_from_directory(
        os.path.join(DATASET_PATH, 'validation'),
        target_size=(PICTURE_SIZE, PICTURE_SIZE),
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
    )

    student = build_tiny_model()
    student.summary()

    student.compile(
        optimizer=tf_keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    callbacks = [
        ModelCheckpoint(OUTPUT_PATH, monitor='val_accuracy', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1),
    ]

    history = student.fit(
        train_set,
        steps_per_epoch=train_set.n // train_set.batch_size,
        epochs=EPOCHS,
        validation_data=val_set,
        validation_steps=val_set.n // val_set.batch_size,
        callbacks=callbacks,
    )

    print(f"\nModel saved to {OUTPUT_PATH}")
    print(f"Total parameters: {student.count_params():,}")
