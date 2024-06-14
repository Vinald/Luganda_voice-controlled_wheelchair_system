import seaborn as sns
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pathlib
from tensorflow.keras.models import load_model

import matplotlib.pyplot as plt

# For a TFLite model
def create_tflite_model(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    return tflite_model

def save_tflite_model(model, model_path='Spectrogram_model_1.tflite'):
    tflite_model = create_tflite_model(model)
    with open(model_path, 'wb') as f:
        f.write(tflite_model) 

def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def get_model_details(interpreter):
    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']
    print('Input shape:', input_shape)

    output_details = interpreter.get_output_details()
    output_shape = output_details[0]['shape']
    print('Output shape:', output_shape)

    return input_shape, output_shape

def run_and_evaluate_tflite_model(interpreter, test_mel_spec_ds, label_names):
    test_mel_spec_ds = test_mel_spec_ds.unbatch().batch(1)

    y_true = []
    y_pred = []

    for mel_spectrogram, label in test_mel_spec_ds:
        interpreter.set_tensor(interpreter.get_input_details()[0]['index'], mel_spectrogram)
        interpreter.invoke()
        output = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])

        y_true.append(label.numpy()[0])
        y_pred.append(np.argmax(output))

    accuracy = accuracy_score(y_true, y_pred)
    print(f'Accuracy is {int(accuracy*100)}%')

    plot_confusion_matrix(y_true, y_pred, label_names)

    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print("Evaluation results:")
    print(f"Accuracy:   {accuracy}")
    print(f"Precision:  {precision}")
    print(f"Recall:     {recall}")
    print(f"F1-score:   {f1}")

def create_quantized_tflite_model(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    quantized_tflite_model = converter.convert()
    return quantized_tflite_model

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

def save_models(keras_model, tflite_model, quantized_tflite_model,
                keras_model_path, tflite_model_path,
                quantized_tflite_model_path):
    keras_model.save(keras_model_path)

    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)

    with open(quantized_tflite_model_path, 'wb') as f:
        f.write(quantized_tflite_model)

SAMPLE_RATE = 16000

KERAS_MODEL_PATH = pathlib.Path('model/model_1.keras')
TFLITE_MODEL_PATH = pathlib.Path('model/model_1.tflite')

if __name__ == "__main__":
    while True:
        audio_file_path = record_audio(duration=3)
        predicted_label, probability = predict_audio(audio_file_path, interpreter, SAMPLE_RATE)
        print(f"Predicted label: {predicted_label}, Probability: {probability}")

        user_input = input("Do you want to continue? (yes/no): ")
        if user_input.lower() != "y":
            break


# For a quantized model
# Function to create a quantized TensorFlow Lite model
def create_quantized_tflite_model(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    quantized_tflite_model = converter.convert()
    return quantized_tflite_model


# Function to evaluate the quantized tflite model on the test dataset
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

