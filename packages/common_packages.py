# Packages and libraries
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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


# ----------------------------------------------------
# Constants
SEED = 42
BATCH_SIZE = 32
SAMPLE_RATE = 16000
VALIDATION_SPLIT = 0.2

# ----------------------------------------------------
# Parameters for melspectrogram
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
SAMPLE_RATE = 16000 * 2

# ----------------------------------------------------
# Parameters for model compiling and training
Epochs = 5
patience = 10
learning_rate = 0.001
