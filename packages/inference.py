from utils import wave, pyaudio, os, librosa, Audio, display, tf
from utils import FRAME_LENGTH,  SAMPLE_RATE, CHANNEL, CHUNK_SIZE, RECORDING_DURATION, FRAME_STEP, N_MELS


# Get Mel-spectrogram
def get_mel_spectrogram(waveform, sample_rate=SAMPLE_RATE, n_mels=N_MELS):
    """
    Generate a Mel-frequency spectrogram from the given waveform.

    Parameters:
    waveform (tf.Tensor): A 1D tensor representing the audio waveform.
    sample_rate (int, optional): The sample rate of the audio waveform. Defaults to SAMPLE_RATE.
    n_mels (int, optional): The number of Mel bands to generate. Defaults to N_MELS.

    Returns:
    tf.Tensor: A 4D tensor representing the Mel-frequency spectrogram. The shape is (-1, 124, 128, 1).
    """
    stft = tf.signal.stft(
        waveform, frame_length=FRAME_LENGTH, frame_step=FRAME_STEP)
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


# Function to get the labels
def get_label_names_ww():
    label_names = ['gaali', 'no_gaali']
    return label_names


def get_label_names_sic():
    label_names_slice = ['ddyo', 'emabega', 'gaali',
                         'kkono', 'mu maaso', 'unknown', 'yimirira']
    return label_names_slice


# Record Audio
def record_audio(filename, duration=RECORDING_DURATION, rate=SAMPLE_RATE, channels=CHANNEL, chunk_size=CHUNK_SIZE):
    """
    Records an audio file using the PyAudio library.

    Parameters:
    filename (str): The name of the output audio file.
    duration (int, optional): The duration of the recording in seconds. Defaults to RECORDING_DURATION.
    rate (int, optional): The sample rate of the audio. Defaults to SAMPLE_RATE.
    channels (int, optional): The number of audio channels. Defaults to CHANNEL.
    chunk_size (int, optional): The size of each chunk of data read from the audio stream. Defaults to CHUNK_SIZE.

    Returns:
    str: The name of the recorded audio file.
    """
    audio = pyaudio.PyAudio()

    # open stream
    stream = audio.open(format=pyaudio.paInt16,
                        channels=channels,
                        rate=rate,
                        input=True,
                        frames_per_buffer=chunk_size)

    print("Recording...")

    # record for duration
    frames = []
    for i in range(0, int(rate / chunk_size * duration)):
        data = stream.read(chunk_size)
        frames.append(data)

    print("Finished recording.")

    # stop and close stream
    stream.stop_stream()
    stream.close()

    # terminate pyaudio object
    audio.terminate()

    # save audio file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()

    return filename


# Function to print audio properties
def print_audio_properties(file_path):
    """
    Prints the properties of an audio file.

    Parameters:
    file_path (str): The path to the audio file.

    Returns:
    None. Prints the audio properties to the console.
    """
    if not os.path.isfile(file_path):
        print(f"File {file_path} does not exist.")
        return

    try:
        audio_signal, sample_rate = librosa.load(file_path, sr=None)
        with wave.open(file_path, 'rb') as audio:
            num_channels = audio.getnchannels()
            frame_rate = audio.getframerate()
            num_frames = audio.getnframes()
            duration = num_frames / float(frame_rate)

            print(f"\nAudio Properties for {os.path.basename(file_path)}")
            print(f"Number of channels:  {num_channels}")
            print(f"Sample rate (Hz):    {sample_rate}")
            print(f"Number of frames:    {num_frames}")
            print(f"Duration (s):        {duration}")
            print()

        # Play the audio files
        audio = Audio(file_path)
        display(audio)
    except Exception as e:
        print(f"An error occurred while processing the file: {e}")
    return


def predict_audio_ww(file_path, model, sample_rate):
    """
    Predicts the label of an audio file using a pre-trained model.

    Parameters:
    file_path (str): The path to the audio file to be predicted.
    model (tf.keras.Model): The pre-trained model for audio classification.
    sample_rate (int): The sample rate of the audio file.

    Returns:
    tuple: A tuple containing the predicted label and its corresponding confidence score.
    """
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
    label_names = get_label_names_ww()
    predicted_label = label_names[predicted_label_index]

    return predicted_label, tf.nn.softmax(predictions[0])[predicted_label_index]


def predict_audio_sic(file_path, model, sample_rate):
    """
    Predicts the label of an audio file using a pre-trained model.

    Parameters:
    file_path (str): The path to the audio file to be predicted.
    model (tf.keras.Model): The pre-trained model for audio classification.
    sample_rate (int): The sample rate of the audio file.

    Returns:
    tuple: A tuple containing the predicted label and its corresponding confidence score.
    """
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
    label_names = get_label_names_sic()
    predicted_label = label_names[predicted_label_index]

    return predicted_label, tf.nn.softmax(predictions[0])[predicted_label_index]
