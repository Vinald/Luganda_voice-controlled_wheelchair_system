# Packages and libraries
import os
import sys
import csv
import json
import math
import wave
import time
import shutil
import pyaudio
import librosa
import pathlib
import numpy as np
import pandas as pd
import seaborn as sns
import librosa.display
import soundfile as sf
import tensorflow as tf
import sounddevice as sd
import matplotlib.pyplot as plt
from tensorflow import keras
from IPython.display import Audio, display
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# ----------------------------------------------------
# Constants
SEED = 42
BATCH_SIZE = 32
SAMPLE_RATE = 16000
VALIDATION_SPLIT = 0.2

# ----------------------------------------------------
# Parameters for mel-spectrogram
FRAME_LENGTH = 255
FRAME_STEP = 128
N_MELS = 128

DURATION = 2
SAMPLES_PER_AUDIO = SAMPLE_RATE * DURATION

# ----------------------------------------------------
# Parameters for MFCCs
N_MFCC = 13
HOP_LENGTH = 512
N_FFT = 2048

# ----------------------------------------------------
# Parameters for model compiling and training
EPOCHS = 5
PATIENCE = 10
LEARNING_RATE = 0.001

# ----------------------------------------------------
# Parameters for recording an audio file for inference
CHANNEL = 1
RECORDING_DURATION = 3
CHUNK_SIZE = 1024


# ----------------------------------------------------
# File paths for the speech classification dataset
aug_train_data_dir = pathlib.Path(
    'Dataset/speech_intent_classification/New_Train')
train_data_dir = pathlib.Path('Dataset/speech_intent_classification/Train')
test_data_dir = pathlib.Path('Dataset/speech_intent_classification/Test')
train_data_needs_preprocessing = pathlib.Path(
    'Dataset/speech_intent_classification/Train_need_preprocessing')
test_data_needs_preprocessing = pathlib.Path(
    'Dataset/speech_intent_classification/Test_need_preprocessing')

# ----------------------------------------------------
# File Path for the wake word model
ww_aug_train_data_dir = pathlib.Path('Dataset/wake_word/New_Train')
ww_train_data_dir = pathlib.Path('Dataset/wake_word/Train')
ww_test_data_dir = pathlib.Path('Dataset/wake_word/Test')
ww_train_data_needs_preprocessing = pathlib.Path(
    'Dataset/wake_word/Train_need_preprocessing')
ww_test_data_needs_preprocessing = pathlib.Path(
    'Dataset/wake_word/Test_need_preprocessing')

# ----------------------------------------------------
# File path for SIC CSV files
sic_aug_train_csv_dir = pathlib.Path('Dataset/csv_files/sic_aug_train.csv')
sic_train_csv_dir = pathlib.Path('Dataset/csv_files/sic_train.csv')
sic_test_csv_dir = pathlib.Path('Dataset/csv_files/sic_test.csv')

# ----------------------------------------------------
# File path for the wake word files
ww_aug_train_csv_dir = pathlib.Path('Dataset/csv_files/ww_aug_train.csv')
ww_train_csv_dir = pathlib.Path('Dataset/csv_files/ww_train.csv')
ww_test_csv_dir = pathlib.Path('Dataset/csv_files/ww_test.csv')

# ----------------------------------------------------
# Mfcc json files
aug_train_json = pathlib.Path('json/mfcc_aug_train_data.json')
train_json = pathlib.Path('json/mfcc_train_data.json')
test_json = pathlib.Path('json/mfcc_test_data.json')

ww_aug_train_json = pathlib.Path('json/ww_mfcc_aug_train_data.json')
ww_train_json = pathlib.Path('json/ww_mfcc_train_data.json')
ww_test_json = pathlib.Path('json/ww_mfcc_test_data.json')

# ---------------------------------------------------------------------------
# Function to print the directory  labels
# ---------------------------------------------------------------------------


def list_directory_contents(directory, label):
    contents = np.array(tf.io.gfile.listdir(str(directory)))
    print(f'{label} commands labels: {contents}')
    return contents


# ---------------------------------------------------------------------------
# Function to get the Model size in KB or MB
# ---------------------------------------------------------------------------
def get_and_convert_file_size(file_path, unit=None):
    size = os.path.getsize(file_path)
    if unit == "KB":
        return print('File size: ' + str(round(size / 1024, 3)) + ' Kilobytes')
    elif unit == "MB":
        return print('File size: ' + str(round(size / (1024 * 1024), 3)) + ' Megabytes')
    else:
        return print('File size: ' + str(size) + ' bytes')
