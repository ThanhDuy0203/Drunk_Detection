import tensorflow as tf
import os


def load_keras_model(model_path):
    """
    Load a Keras model from a .keras file.

    Args:
        model_path (str): Path to the .keras model file.

    Returns:
        tf.keras.Model: Loaded Keras model.

    Raises:
        FileNotFoundError: If the model file does not exist.
        ValueError: If the file is not a valid .keras model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Successfully loaded Keras model from {model_path}")
        return model
    except Exception as e:
        raise ValueError(f"Failed to load .keras model: {str(e)}")


def convert_to_tflite(model, optimize=False):
    """
    Convert a Keras model to TFLite format.

    Args:
        model (tf.keras.Model): Keras model to convert.
        optimize (bool): If True, apply default optimizations (e.g., size reduction).

    Returns:
        bytes: TFLite model content.

    Raises:
        RuntimeError: If conversion fails.
    """
    try:
        # Initialize TFLite converter
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        # Apply optimizations if requested
        if optimize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # Convert the model
        tflite_model = converter.convert()
        print("Model successfully converted to TFLite format")
        return tflite_model

    except Exception as e:
        raise RuntimeError(f"Failed to convert model to TFLite: {str(e)}")


def save_tflite_model(tflite_model, output_path):
    """
    Save the TFLite model to a file.

    Args:
        tflite_model (bytes): TFLite model content.
        output_path (str): Path to save the .tflite file.

    Raises:
        IOError: If saving the file fails.
    """
    try:
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        print(f"TFLite model saved to {output_path}")
    except Exception as e:
        raise IOError(f"Failed to save TFLite model: {str(e)}")


def main():
    """
    Main function to convert a .keras model to TFLite.
    """
    # Configuration
    model_path = r'D:\FPTUniversity\Capstone_Project\Drunk_Detection\models\drunk_detection_model.keras'  # Replace with your .keras model path
    output_path = r'D:\FPTUniversity\Capstone_Project\Drunk_Detection\models\final_model.tflite'  # Output path for TFLite model
    apply_optimizations = False  # Set to True for size optimization

    try:
        # Load the Keras model
        model = load_keras_model(model_path)

        # Convert to TFLite
        tflite_model = convert_to_tflite(model, optimize=apply_optimizations)

        # Save the TFLite model
        save_tflite_model(tflite_model, output_path)

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()