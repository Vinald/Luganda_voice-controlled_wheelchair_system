import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time
from time import sleep
import wave
import pyaudio
import pathlib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


KERAS_MODEL_PATH = "model/spec_model_1.keras"
model = load_model(KERAS_MODEL_PATH)

SEED = 42
BATCH_SIZE = 64
FRAME_STEP = 128
FRAME_LENGTH = 255
SAMPLE_RATE = 16000


start_time = time.time()


# Function to get the labels
def get_label_names():
    label_names_slice = ['ddyo', 'emabega', 'gaali', 'kkono', 'mumaaso', 'unknown', 'yimirira']
    return label_names_slice


def record_audio(filename='recorded_audio.wav', duration=3, fs=SAMPLE_RATE, channels=1, format=pyaudio.paInt16):
    CHUNK = 1024
    audio = pyaudio.PyAudio()

    stream = audio.open(format=format, channels=channels, rate=fs, input=True, frames_per_buffer=CHUNK)

    print("Recording started...")
    frames = []

    for i in range(0, int(fs / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Recording finished.")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save the audio file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(audio.get_sample_size(format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()

    return filename


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
    x, _ = tf.audio.decode_wav(x, desired_channels=1, desired_samples=SAMPLE_RATE)
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

    return predicted_label


def direction(predicted_label):
    if predicted_label == 'ddyo':
        return 'ddyo'
    elif predicted_label == 'emabega':
        return 'emabega'
    elif predicted_label == 'gaali':
        return 'gaali'
    elif predicted_label == 'kkono':
        return 'kkono'
    elif predicted_label == 'mumaaso':
        return 'mumaaso'
    elif predicted_label == 'yimirira':
        return 'yimirira'
    else:
        return 'Unknown'


file_path_inference = record_audio()
sleep(1)
predicted_label = predict_audio(file_path_inference, model, SAMPLE_RATE)

command = direction(predicted_label)
print(f'Predicted label: {command}')

end_time = time.time()
print(f'Execution time {end_time - start_time} secs')