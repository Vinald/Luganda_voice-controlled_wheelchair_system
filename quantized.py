import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Function to create a quantized TensorFlow Lite model
def create_quantized_tflite_model(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    quantized_tflite_model = converter.convert()
    return quantized_tflite_model


# Function to evaluate the quantized tflite model on the test dataset
def plot_confusion_matrix(y_true, y_pred, label_names):
    confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_mtx,
                xticklabels=label_names,
                yticklabels=label_names,
                annot=True, fmt='g')
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    plt.title('Confusion Matrix')
    plt.show()


# Function to run and evaluate the TFLite model
def evaluate_quantized_tflite_model(interpreter, test_ds, label_names):
    y_true = []
    y_pred = []

    for mel_spectrogram, label in test_ds:
        interpreter.set_tensor(interpreter.get_input_details()[0]['index'], mel_spectrogram)
        interpreter.invoke()
        output = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
        predicted_label = tf.argmax(output, axis=1)[0]
        y_true.append(label.numpy()[0])
        y_pred.append(predicted_label.numpy())

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print("Evaluation results for quantized model:")
    print(f"Accuracy:   {accuracy}")
    print(f"Precision:  {precision}")
    print(f"Recall:     {recall}")
    print(f"F1-score:   {f1}")

    plot_confusion_matrix(y_true, y_pred, label_names)