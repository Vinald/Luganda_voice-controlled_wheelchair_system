from flask import Flask, request, jsonify, render_template, jsonify
import numpy as np
import tensorflow as tf
import sounddevice as sd
import soundfile as sf
from datetime import datetime
import os


app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('model/spec_model_1.keras')

# Define the labels
label_names = ['ddyo', 'emabega', 'gaali', 'kkono', 'mumaaso', 'unknown', 'yimirira']


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/labels')
def labels():
    return render_template('labels.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Record audio from microphone
    duration = 2
    sample_rate = 16000
    channels = 1
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels, dtype='int16')
    sd.wait()

    # Generate a unique filename using current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = f"recording/recorded_audio_{timestamp}.wav"

    # Check if the directory exists, if not create it
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Save the recorded audio to a file
    sf.write(file_path, audio_data, sample_rate)

    # Load the audio file and preprocess for inference
    x = tf.io.read_file(str(file_path))
    x, _ = tf.audio.decode_wav(x, desired_channels=1, desired_samples=16000,)
    x = tf.squeeze(x, axis=-1)
    x = get_mel_spectrogram(x)
    x = x[np.newaxis, ..., np.newaxis]  

    # Perform inference
    prediction = model.predict(x)

    # Get the predicted label
    predicted_label_index = np.argmax(prediction[0])
    predicted_label = label_names[predicted_label_index]

    return jsonify({'prediction': predicted_label})


def get_mel_spectrogram(waveform, sample_rate=16000, n_mels=128):
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

    # Add an axis for the batch size
    mel_spectrogram = mel_spectrogram[..., tf.newaxis]

    return mel_spectrogram


if __name__ == '__main__':
    app.run(debug=True)