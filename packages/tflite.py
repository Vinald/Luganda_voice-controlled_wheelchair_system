from packages.utils import tf, np, precision_score, accuracy_score, recall_score, f1_score, plt, sns


# For a TFLite model
def create_and_save_tflite_model(model, model_path='Spectrogram_model_1.tflite'):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(model_path, 'wb') as f:
        f.write(tflite_model)
    return tflite_model


# Function to load a saved TFLite model
def load_and_get_model_details(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']
    print('Input shape:', input_shape)

    output_details = interpreter.get_output_details()
    output_shape = output_details[0]['shape']
    print('Output shape:', output_shape)

    return interpreter, input_shape, output_shape


def run_and_evaluate_tflite_model(interpreter, test_mel_spec_ds, label_names):
    test_mel_spec_ds = test_mel_spec_ds.unbatch().batch(1)

    y_true = []
    y_pred = []

    for mel_spectrogram, label in test_mel_spec_ds:
        interpreter.set_tensor(interpreter.get_input_details()[
                               0]['index'], mel_spectrogram)
        interpreter.invoke()
        output = interpreter.get_tensor(
            interpreter.get_output_details()[0]['index'])

        y_true.append(label.numpy()[0])
        y_pred.append(np.argmax(output))

    accuracy = accuracy_score(y_true, y_pred)
    print(f'Accuracy is {int(accuracy*100)}%')

    plot_confusion_matrix(y_true, y_pred, label_names)

    precision = precision_score(
        y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print("Evaluation results:")
    print(f"Accuracy:   {accuracy}")
    print(f"Precision:  {precision}")
    print(f"Recall:     {recall}")
    print(f"F1-score:   {f1}")


# Quantized Model
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
        interpreter.set_tensor(interpreter.get_input_details()[
                               0]['index'], mel_spectrogram)
        interpreter.invoke()
        output = interpreter.get_tensor(
            interpreter.get_output_details()[0]['index'])
        predicted_label = tf.argmax(output, axis=1)[0]
        y_true.append(label.numpy()[0])
        y_pred.append(predicted_label.numpy())

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(
        y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print("Evaluation results for quantized model:")
    print(f"Accuracy:   {accuracy}")
    print(f"Precision:  {precision}")
    print(f"Recall:     {recall}")
    print(f"F1-score:   {f1}")

    plot_confusion_matrix(y_true, y_pred, label_names)
