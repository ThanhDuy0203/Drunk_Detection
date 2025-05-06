import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from pathlib import Path

# Constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 2
EPOCHS = 100
FINE_TUNE_EPOCHS = 20
INITIAL_LR = 1e-4  # Learning rate cho Phase 1
FINE_TUNE_LR = 1e-5  # Learning rate cho Phase 2

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

def prepare_data(train_dir, val_dir, test_dir):
    """Chuẩn bị dữ liệu huấn luyện, validation và test với augmentation cho train và val"""
    train_val_datagen = ImageDataGenerator(
        rescale=1. / 255,
        brightness_range=(0.8, 1.2),
        zoom_range=0.1,
        width_shift_range=0.05,
        height_shift_range=0.05
    )

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_val_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    validation_generator = train_val_datagen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, validation_generator, test_generator

def build_model():
    """Xây dựng mô hình MobileNetV3 với các lớp tùy chỉnh"""
    base_model = MobileNetV3Small(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
    base_model.trainable = False  # Freeze backbone trong Phase 1

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='swish')(x)
    x = BatchNormalization()(x)
    x = Dense(64, activation='swish')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    return model, base_model

def setup_callbacks(model_checkpoint_path):
    """Thiết lập callbacks: dừng sớm và lưu checkpoint tốt nhất"""
    checkpoint = ModelCheckpoint(
        model_checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=25,
        restore_best_weights=True,
        verbose=1
    )

    return [checkpoint, early_stopping]

def train_model(model, train_generator, validation_generator, callbacks, epochs, lr):
    """Huấn luyện mô hình với learning rate cụ thể"""
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=callbacks,
        verbose=1
    )
    return history

def fine_tune_model(model, base_model, train_generator, validation_generator, callbacks):
    """Fine-tune: mở một phần backbone và huấn luyện với learning rate thấp"""
    print("\n🔧 Phase 2: Fine-tuning mô hình...")
    base_model.trainable = True
    fine_tune_at = 100  # Chỉ mở khóa các lớp sau lớp thứ 100
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    history_fine = train_model(
        model,
        train_generator,
        validation_generator,
        callbacks,
        epochs=FINE_TUNE_EPOCHS,
        lr=FINE_TUNE_LR
    )
    return history_fine

def save_model(model, output_path):
    """Lưu mô hình đã huấn luyện ở định dạng HDF5"""
    model.save(output_path, save_format='h5')
    print(f"Mô hình đã được lưu tại: {output_path}")

def plot_training_history(history, history_fine=None):
    """Vẽ biểu đồ accuracy và loss, kết hợp Phase 1 và Phase 2"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    if history_fine:
        train_acc += history_fine.history['accuracy']
        val_acc += history_fine.history['val_accuracy']
        train_loss += history_fine.history['loss']
        val_loss += history_fine.history['val_loss']

    ax1.plot(train_acc, label='Training Accuracy')
    ax1.plot(val_acc, label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(train_loss, label='Training Loss')
    ax2.plot(val_loss, label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

def main():
    """Hàm chính để chạy toàn bộ quy trình"""
    has_gpu = check_gpu()

    # Đường dẫn tới các folder
    train_dir = "D:/FPTUniversity/Capstone_Project/Drunk_Detection/train2"
    val_dir = "D:/FPTUniversity/Capstone_Project/Drunk_Detection/val"
    test_dir = "D:/FPTUniversity/Capstone_Project/Drunk_Detection/test"
    model_checkpoint_path = "D:/FPTUniversity/Capstone_Project/Drunk_Detection/models/best_model.h5"
    final_model_path = "D:/FPTUniversity/Capstone_Project/Drunk_Detection/models/drunk_detection_model_finetuned.h5"

    # Chuẩn bị dữ liệu
    train_generator, validation_generator, test_generator = prepare_data(train_dir, val_dir, test_dir)
    model, base_model = build_model()
    model.summary()

    callbacks = setup_callbacks(model_checkpoint_path)

    # Phase 1: Huấn luyện các lớp Dense (freeze backbone)
    print("\nPhase 1: Huấn luyện các lớp Dense mới...")
    history = train_model(model, train_generator, validation_generator, callbacks, EPOCHS, INITIAL_LR)

    # Phase 2: Fine-tuning một phần MobileNetV3
    history_fine = fine_tune_model(model, base_model, train_generator, validation_generator, callbacks)

    # Vẽ biểu đồ lịch sử huấn luyện
    plot_training_history(history, history_fine)

    # Đánh giá trên tập test
    print("\nĐánh giá mô hình trên tập test...")
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    save_model(model, final_model_path)

if __name__ == "__main__":
    main()