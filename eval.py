import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def load_trained_model(model_path):
    print(f"Đang tải mô hình từ: {model_path}")
    model = tf.keras.models.load_model(model_path)
    return model

def prepare_test_data(test_dir):
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False  # Không shuffle để dễ đánh giá
    )

    return test_generator

def evaluate_model(model, test_generator):
    # Dự đoán nhãn của tập test
    predictions = model.predict(test_generator, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)  # Lấy lớp có xác suất cao nhất
    true_classes = test_generator.classes  # Nhãn thật

    # Tính toán các chỉ số đánh giá
    accuracy = accuracy_score(true_classes, predicted_classes)
    f1 = f1_score(true_classes, predicted_classes, average='weighted')
    cm = confusion_matrix(true_classes, predicted_classes)

    # Tính False Negative Rate (FNR)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    # In kết quả
    print("\n Kết quả đánh giá:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"False Negative Rate (FNR): {fnr:.4f}")

    # Vẽ confusion matrix
    plot_confusion_matrix(cm, test_generator.class_indices)

def plot_confusion_matrix(cm, class_indices):
    """Vẽ confusion matrix"""
    labels = list(class_indices.keys())
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

def main():

    model_path = "D:/FPTUniversity/Capstone_Project/Drunk_Detection/models/drunk_detection_model.keras"
    test_dir = "D:/FPTUniversity/Capstone_Project/Drunk_Detection/test"


    model = load_trained_model(model_path)
    test_generator = prepare_test_data(test_dir)

    evaluate_model(model, test_generator)

if __name__ == "__main__":
    main()