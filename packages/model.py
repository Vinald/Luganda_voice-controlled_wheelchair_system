from packages.utils import os, tf, layers, models, EarlyStopping, plt, sns
from packages.utils import f1_score, precision_score, recall_score

# ----------------------------------------------------
# Parameters for model compiling and training
EPOCHS = 30
PATIENCE = 30
LEARNING_RATE = 0.001
MIN_LR = 1e-6


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
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
loss_sic = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
loss_ww = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)


# Function to compile and train the model
def compile_and_train_model_sic(model, train_ds, val_ds, learning_rate=LEARNING_RATE):
    try:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss=loss_sic, metrics=['accuracy'])
        early_stopping = EarlyStopping(
            monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.1, patience=PATIENCE, min_lr=MIN_LR)
        history = model.fit(train_ds, validation_data=val_ds,
                            epochs=EPOCHS, callbacks=[early_stopping, reduce_lr])
        return history
    except Exception as e:
        print(
            f"An error occurred during model compilation and training: {str(e)}")


def compile_and_train_model_ww(model, train_ds, val_ds, learning_rate=LEARNING_RATE):
    try:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss=loss_ww, metrics=['accuracy'])
        early_stopping = EarlyStopping(
            monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.1, patience=PATIENCE, min_lr=MIN_LR)
        history = model.fit(train_ds, validation_data=val_ds,
                            epochs=EPOCHS, callbacks=[early_stopping, reduce_lr])
        return history
    except Exception as e:
        print(
            f"An error occurred during model compilation and training: {str(e)}")


# --------------------------------------------------------------------
# Function to evaluate the model on the test dataset
def evaluate_model(model, test_ds):
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


# --------------------------------------------------------------------
# Function to get the Model size in KB or MB
def get_model_size(file_path, unit=None):
    size = os.path.getsize(file_path)
    if unit == "KB":
        print('File size: ' + str(round(size / 1024, 3)) + ' Kilobytes')
    elif unit == "MB":
        print('File size: ' + str(round(size / (1024 * 1024), 3)) + ' Megabytes')
    else:
        print('File size: ' + str(size) + ' bytes')
