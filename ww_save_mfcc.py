
import os
import json
import math
import librosa
import numpy as np

# Constants
SAMPLE_RATE = 16000  # Ensure this matches the sample rate of your audio files
DURATION = 2  # Duration in seconds
SAMPLES_PER_AUDIO = SAMPLE_RATE * DURATION
