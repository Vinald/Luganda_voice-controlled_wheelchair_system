import os
from flask import Flask, render_template, request
import tensorflow as tf
import sounddevice as sd
from datetime import datetime
import soundfile as sf

app = Flask(__name__)

# Check if the file exists
if os.path.isfile('model/model_1.tflite'):
    print("File exists")
else:
    print("File does not exist")

# Check if the file is readable
if os.access('model/model_1.tflite', os.R_OK):
    print("File is readable")
else:
    print("File is not readable")

# Load the model
interpreter = tf.lite.Interpreter(model_path=str('model/model_1.tflite'))
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

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

    # Your prediction code here

    return render_template('predict.html')

if __name__ == "__main__":
    app.run(debug=True)