import os
import time
import pathlib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

SEED = 42
BATCH_SIZE = 64
FRAME_STEP = 128
FRAME_LENGTH = 255
SAMPLE_RATE = 16000
VALIDATION_SPLIT = 0.2

start_time = time.time()

test_data_dir = pathlib.Path('audios')

KERAS_MODEL_PATH = "model/spec_model_1.keras"
model = load_model(KERAS_MODEL_PATH)




# Function to get the labels
def get_label_names():
    label_names_slice = ['ddyo', 'emabega', 'gaali', 'kkono', 'mumaaso', 'unknown', 'yimirira']
    return label_names_slice


def get_mel_spectrogram(waveform, sample_rate=SAMPLE_RATE, n_mels=128):
    stft = tf.signal.stft(waveform, frame_length=FRAME_LENGTH, frame_step=FRAME_STEP)
    spectrogram = tf.abs(stft)
    mel_spectrogram = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=n_mels,
        num_spectrogram_bins=spectrogram.shape[-1],
        sample_rate=sample_rate
    )
    mel_spectrogram = tf.tensordot(spectrogram, mel_spectrogram, 1)
    mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
    mel_spectrogram = tf.reshape(mel_spectrogram, (-1, 124, 128, 1))

    return mel_spectrogram


def predict_audio(file_path, model, sample_rate):
    x = tf.io.read_file(str(file_path))
    x, _ = tf.audio.decode_wav(x, desired_channels=1, desired_samples=16000)
    x = tf.squeeze(x, axis=-1)
    waveform = x
    x = get_mel_spectrogram(x, sample_rate)

    max_frames = 124
    pad_size = max_frames - x.shape[1]
    if pad_size > 0:
        x = tf.pad(x, [[0, 0], [0, pad_size], [0, 0], [0, 0]])
    else:
        x = x[:, :max_frames, :, :]

    predictions = model.predict(x)
    predicted_label_index = tf.argmax(predictions[0])
    label_names = get_label_names()
    predicted_label = label_names[predicted_label_index]

    return predicted_label, tf.nn.softmax(predictions[0])[predicted_label_index]


emabega_file_path = os.path.join(test_data_dir, 'emabega_92.wav')
ddyo_file_path = os.path.join(test_data_dir, 'ddyo_141.wav')
yimirira_file_path = os.path.join(test_data_dir, 'yimirira_100.wav')
kkono_file_path = os.path.join(test_data_dir, 'kkono_113.wav')
mumasso_file_path = os.path.join(test_data_dir, 'mumaaso_40.wav')
gaali_file_path = os.path.join(test_data_dir, 'gaali_91.wav')


file_path_inference = yimirira_file_path
predicted_label, probability = predict_audio(file_path_inference, model, SAMPLE_RATE)
print(f"Predicted label: {predicted_label}, Probability: {probability}")

end_time = time.time()
print(f'Execution time {end_time - start_time} secs')


import os
import pathlib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython.display import Audio


def get_label_names():
    label_names_slice = ['emabega', 'noise', 'ddyo', 'yimirira', 'kkono', 'mu masso', 'gaali']
    return label_names_slice


def get_mel_spectrogram(waveform, sample_rate, n_mels=128):
    stft = tf.signal.stft(waveform, frame_length=255, frame_step=128)
    spectrogram = tf.abs(stft)
    mel_spectrogram = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=n_mels,
        num_spectrogram_bins=spectrogram.shape[-1],
        sample_rate=sample_rate
    )
    mel_spectrogram = tf.tensordot(spectrogram, mel_spectrogram, 1)
    mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
    mel_spectrogram = tf.reshape(mel_spectrogram, (-1, 124, 128, 1))
    return mel_spectrogram


def plot_predictions(predictions, label_names):
    plt.bar(label_names, tf.nn.softmax(predictions[0]))
    plt.title(tf.argmax(predictions[0]))
    plt.show()


def predict_audio(file_path, model, sample_rate):
    x = tf.io.read_file(str(file_path))
    x, _ = tf.audio.decode_wav(x, desired_channels=1, desired_samples=16000)
    x = tf.squeeze(x, axis=-1)
    waveform = x
    x = get_mel_spectrogram(x, sample_rate)

    max_frames = 124
    pad_size = max_frames - x.shape[1]
    if pad_size > 0:
        x = tf.pad(x, [[0, 0], [0, pad_size], [0, 0], [0, 0]])
    else:
        x = x[:, :max_frames, :, :]

    predictions = model.predict(x)
    predicted_label_index = tf.argmax(predictions[0])
    label_names = get_label_names()
    predicted_label = label_names[predicted_label_index]

    plot_predictions(predictions, label_names)

    return predicted_label, tf.nn.softmax(predictions[0])[predicted_label_index]


import tensorflow as tf
import tensorflow_io as tfio


def load_audio_file(file_path):
    audio = tfio.audio.AudioIOTensor(file_path)
    waveform = tf.squeeze(audio[:], axis=[-1])
    return waveform


def get_mel_spectrogram(waveform, sample_rate=16000, n_mels=128):
    if not isinstance(waveform, tf.Tensor):
        raise ValueError("Input waveform must be a Tensor.")

    # Compute the Short-Time Fourier Transform (STFT)
    stft = tf.signal.stft(waveform, frame_length=255, frame_step=128)

    # Convert the magnitude of the STFT to a Mel spectrogram
    spectrogram = tf.abs(stft)
    mel_spectrogram = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=n_mels,
        num_spectrogram_bins=spectrogram.shape[-1],
        sample_rate=sample_rate
    )
    mel_spectrogram = tf.tensordot(spectrogram, mel_spectrogram, 1)
    mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)  # Log-scale

    # Reshape the Mel spectrogram to match the input shape of the Conv2D model
    mel_spectrogram = tf.transpose(mel_spectrogram, perm=[1, 0])
    mel_spectrogram = tf.expand_dims(mel_spectrogram, axis=-1)

    return mel_spectrogram


def make_inference(file_path, model, sample_rate=16000, n_mels=128):
    waveform = load_audio_file(file_path)
    mel_spectrogram = get_mel_spectrogram(waveform, sample_rate, n_mels)

    # Make an inference
    prediction = model(mel_spectrogram)

    return prediction