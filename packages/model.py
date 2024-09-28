from packages.utils import os, tf, layers, models, EarlyStopping, plt, f1_score, sns, precision_score, recall_score
from packages.utils import EPOCHS, PATIENCE, LEARNING_RATE


# --------------------------------------------------------------------
# Model 1
def model(input_shape, num_labels):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(16, 3, activation='relu',
                      padding='same'), layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation='relu',
                      padding='same'), layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu',
                      padding='same'), layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation='relu',
                      padding='same'), layers.GlobalMaxPooling2D(),
        layers.Dense(128, activation='relu'), layers.Dropout(0.5),
        layers.Dense(num_labels, activation='softmax')
    ])
    return model


# --------------------------------------------------------------------
# Model 2
def model2(input_shape, num_labels):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Resizing(64, 64),
        layers.Normalization(),
        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Conv2D(128, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_labels, activation='softmax')
    ])
    return model


# --------------------------------------------------------------------
# Model 3
def model3(input_shape, num_labels):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(16, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation='relu', padding='same'),
        layers.GlobalMaxPooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_labels, activation='softmax')
    ])
    return model


# --------------------------------------------------------------------
# Function to compile and train the model
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)


def compile_and_train_model(model, train_ds, val_ds, learning_rate=LEARNING_RATE):
    """
    This function compiles and trains a given model using the provided training and validation datasets.

    Parameters:
    - model: The Keras model to be trained.
    - train_ds: The training dataset. It should be a TensorFlow Dataset object containing audio data and corresponding labels.
    - val_ds: The validation dataset. It should be a TensorFlow Dataset object containing audio data and corresponding labels.
    - learning_rate (optional): The learning rate for the optimizer. Default value is defined by the LEARNING_RATE constant.

    Returns:
    - history: The history object returned by the model.fit method, which contains the training and validation loss and accuracy for each epoch.

    Raises:
    - Exception: If an error occurs during model compilation and training, it will be caught and printed.
    """
    try:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
        early_stopping = EarlyStopping(
            monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.1, patience=PATIENCE, min_lr=1e-6)
        history = model.fit(train_ds, validation_data=val_ds,
                            epochs=EPOCHS, callbacks=[early_stopping, reduce_lr])
        return history
    except Exception as e:
        print(
            f"An error occurred during model compilation and training: {str(e)}")


# --------------------------------------------------------------------
# Function to evaluate the model on the test dataset
def evaluate_model(model, test_ds):
    """
    Evaluates a trained model on a test dataset and prints the evaluation metrics.

    Parameters:
    - model: A trained Keras model to be evaluated.
    - test_ds: A TensorFlow Dataset object containing the test audio data and corresponding labels.

    Returns:
    - None. The function prints the test accuracy, loss, precision, recall, and F1-score.

    Raises:
    - Exception: If an error occurs during model evaluation, it will be caught and printed.
    """
    try:
        y_true = []
        y_pred = []
        for audio, labels in test_ds:
            predictions = model.predict(audio, verbose=0)
            y_true.extend(labels.numpy())
            y_pred.extend(tf.argmax(predictions, axis=1).numpy())

        loss, accuracy = model.evaluate(test_ds, verbose=0)
        precision = precision_score(
            y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(
            y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        print(f"Test accuracy:      {int(accuracy * 100)}%")
        print(f"Test loss:          {loss}")
        print(f"Precision:          {precision}")
        print(f"Recall:             {recall}")
        print(f"F1-score:           {f1}")
    except Exception as e:
        print(f"An error occurred during model evaluation: {str(e)}")


# --------------------------------------------------------------------
# Function to plot the training history
def plot_training_history(history):
    """
    This function plots the training and validation accuracy and loss for a given Keras model's training history.

    Parameters:
    - history: A TensorFlow History object returned by the model.fit method. It contains the training and validation loss and accuracy for each epoch.

    Returns:
    - None. The function generates a plot displaying the training and validation accuracy and loss over epochs.

    Raises:
    - Exception: If an error occurs during the plot generation, it will be caught and printed.
    """
    try:
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(len(acc))

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, acc, 'r', label='Training accuracy')
        plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, loss, 'r', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(
            f"An error occurred during plotting the training history: {str(e)}")


# --------------------------------------------------------------------
# Function to plot the confusion matrix
def plot_confusion_matrix(y_true, y_pred, label_names):
    """
    Plots a confusion matrix using seaborn heatmap.

    Parameters:
    - y_true (numpy.ndarray): A 1D array containing the true labels of the samples.
    - y_pred (numpy.ndarray): A 1D array containing the predicted labels of the samples.
    - label_names (list): A list of strings representing the names of the classes.

    Returns:
    - None. The function generates a plot displaying the confusion matrix.

    Raises:
    - Exception: If an error occurs during the plot generation, it will be caught and printed.
    """
    try:
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
    except Exception as e:
        print(
            f"An error occurred during plotting the confusion matrix: {str(e)}")


# ---------------------------------------------------------------------------
# Function to get the Model size in KB or MB
def get_and_convert_file_size(file_path, unit=None):
    """
    This function calculates and prints the size of a file in bytes, kilobytes, or megabytes.

    Parameters:
    - file_path (str): The path to the file for which the size needs to be calculated.
    - unit (str, optional): The unit in which the size should be displayed. It can be either 'KB' or 'MB'. If not provided, the size will be displayed in bytes.

    Returns:
    - None: The function prints the size of the file in the specified unit.
    """
    size = os.path.getsize(file_path)
    if unit == "KB":
        print('File size: ' + str(round(size / 1024, 3)) + ' Kilobytes')
    elif unit == "MB":
        print('File size: ' + str(round(size / (1024 * 1024), 3)) + ' Megabytes')
    else:
        print('File size: ' + str(size) + ' bytes')
