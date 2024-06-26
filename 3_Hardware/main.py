from time import sleep

import tensorflow as tf
from modules.SIC import record_audio, predict_sic_label, predict_ww_label
from modules.prediction import process_predicted_label

SAMPLE_RATE = 16000

# Load the model
try:
    WW_model = tf.keras.models.load_model('model/wakeword_model_1.keras')
    print("WW Model loaded successfully.")
    SIC_model = tf.keras.models.load_model('model/spec_model_np_1.keras')
    print("SIC Model loaded successfully.")
except Exception as e:
    print("Error loading model: ", e)
    exit(1)


while True:
    print(' Say The Wakeword To Turn On System')
    sleep(2)

    wakeword_file = record_audio('wakeword.wav', 4)
    wakeword, _ = predict_ww_label(wakeword_file, WW_model, SAMPLE_RATE)
    wakeword = wakeword.lower()
    sleep(5)

    if wakeword == 'gaali':
        print("WAKEWORD DETECTED, Speak a command")
        sleep(3)

        while True:
            audio_file = record_audio('output.wav', 4)
            command, _ = predict_sic_label(audio_file, SIC_model, SAMPLE_RATE)
            command = command.lower()
            process_predicted_label(command)

    elif wakeword == 'no_gaali':
        print("INVALID WAKEWORD, Say the right wakeword")
        sleep(3)
        continue

    else:
        print("INVALID WAKEWORD, Say the right wakeword")
        sleep(3)
        continue

