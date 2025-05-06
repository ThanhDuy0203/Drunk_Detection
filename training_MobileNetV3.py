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
from tensorflow.nn import relu6
from pathlib import Path

# Constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 2
EPOCHS = 100

def hard_swish(x):
    return x * relu6(x + 3) / 6

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
    # Data augmentation cho tập train và validation
    train_val_datagen = ImageDataGenerator(
        rescale=1. / 255,
        brightness_range=(0.8, 1.2),
        zoom_range=0.1,
        width_shift_range=0.05,
        height_shift_range=0.05
    )

    # Không augmentation cho tập test, chỉ rescale
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    # Train generator
    train_generator = train_val_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    # Validation generator (giữ augmentation)
    validation_generator = train_val_datagen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False  # Không shuffle để đánh giá nhất quán
    )

    # Test generator
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False  # Không shuffle để dễ đánh giá
    )

    return train_generator, validation_generator, test_generator

def build_model():
    """Xây dựng mô hình MobileNetV3 với các lớp tùy chỉnh"""
    base_model = MobileNetV3Small(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='swish')(x)
    x = BatchNormalization()(x)
    x = Dense(64, activation='swish')(x)
    x = Dropout(0.4)(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    optimizer = Adam(learning_rate=0.0001)
    model.compile(
        optimizer=optimizer,
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
        patience=36,
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
    """Lưu mô hình đã huấn luyện ở định dạng mặc định (SavedModel)"""
    output_dir = Path(output_path)
    output_dir.parent.mkdir(parents=True, exist_ok=True)  # Tạo thư mục cha nếu chưa tồn tại
    model.save(output_path)  # Lưu ở định dạng mặc định (SavedModel)
    print(f"Mô hình đã được lưu tại: {output_path}")

def plot_training_history(history):
    """Vẽ biểu đồ accuracy và loss của quá trình huấn luyện"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
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
    model_checkpoint_path = "D:/FPTUniversity/Capstone_Project/Drunk_Detection/models/best_model.keras"
    final_model_path = "D:/FPTUniversity/Capstone_Project/Drunk_Detection/models/drunk_detection_model.keras"

    # Chuẩn bị dữ liệu
    train_generator, validation_generator, test_generator = prepare_data(train_dir, val_dir, test_dir)
    model = build_model()
    model.summary()

    # Thiết lập callbacks
    callbacks = setup_callbacks(model_checkpoint_path)

    # Huấn luyện mô hình
    history = train_model(model, train_generator, validation_generator, callbacks)
    plot_training_history(history)

    # Đánh giá trên tập test
    print("\nĐánh giá mô hình trên tập test...")
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # Lưu mô hình
    save_model(model, final_model_path)

if __name__ == "__main__":
    main()