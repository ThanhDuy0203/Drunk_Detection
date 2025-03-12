import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from pathlib import Path

# Constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 2
EPOCHS = 50


def check_gpu():
    """Kiểm tra và hiển thị thông tin GPU"""
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("Không tìm thấy GPU. Huấn luyện sẽ chạy trên CPU.")
    else:
        print(f"Tìm thấy {len(gpus)} GPU: {gpus}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    return len(gpus) > 0


def prepare_data(train_dir):
    """Chuẩn bị dữ liệu huấn luyện với augmentation"""
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=True
    )

    return train_generator, validation_generator


def build_model():
    """Xây dựng mô hình MobileNetV3 với các lớp tùy chỉnh"""
    base_model = MobileNetV3Small(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def setup_callbacks(model_checkpoint_path):
    """Thiết lập callbacks cho huấn luyện"""
    checkpoint = ModelCheckpoint(
        model_checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=22,
        restore_best_weights=True,
        verbose=1
    )

    return [checkpoint, early_stopping]


def train_model(model, train_generator, validation_generator, callbacks):
    """Huấn luyện mô hình trên GPU nếu có"""
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=callbacks,
        verbose=1
    )
    return history


def save_model(model, output_path):
    """Lưu mô hình đã huấn luyện ở định dạng HDF5"""
    model.save(output_path, save_format='h5')  # Chỉ định định dạng HDF5
    print(f"Mô hình đã được lưu tại: {output_path}")


def main():
    """Hàm chính để chạy toàn bộ quy trình"""
    has_gpu = check_gpu()

    train_dir = "D:/FPTUniversity/Capstone_Project/Drunk_Detection/train"
    model_checkpoint_path = "D:/FPTUniversity/Capstone_Project/Drunk_Detection/models/best_model.h5"  # Đổi sang .h5
    final_model_path = "D:/FPTUniversity/Capstone_Project/Drunk_Detection/models/drunk_detection_model.h5"  # Đổi sang .h5

    train_generator, validation_generator = prepare_data(train_dir)
    model = build_model()
    model.summary()

    callbacks = setup_callbacks(model_checkpoint_path)
    history = train_model(model, train_generator, validation_generator, callbacks)
    save_model(model, final_model_path)


if __name__ == "__main__":
    main()