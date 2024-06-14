import librosa
import numpy as np
import os

N_MELS = 128
SAMPLE_RATE = 16000
HOP_LENGTH = 512
MAX_LENGTH = 128


def extract_melspectrogram(audio_path, n_mels=N_MELS, sr=SAMPLE_RATE, hop_length=HOP_LENGTH):
    y, sr = librosa.load(audio_path, sr=sr)
    melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    melspectrogram_db = librosa.power_to_db(melspectrogram, ref=np.max)
    return melspectrogram_db


def pad_melspectrogram(melspectrogram, max_len=MAX_LENGTH):
    if melspectrogram.shape[1] < max_len:
        pad_width = max_len - melspectrogram.shape[1]
        melspectrogram = np.pad(melspectrogram, ((0, 0), (0, pad_width)), mode='constant')
    else:
        melspectrogram = melspectrogram[:, :max_len]
    return melspectrogram


def process_audio_files(root_directory, n_mels=N_MELS, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, max_len=MAX_LENGTH):
    data = []
    labels = []
    
    for root, _, files in os.walk(root_directory):
        for file_name in files:
            if file_name.endswith('.wav'):
                audio_path = os.path.join(root, file_name)
                melspectrogram = extract_melspectrogram(audio_path, n_mels, sr, hop_length)
                melspectrogram = pad_melspectrogram(melspectrogram, max_len)
                
                # Normalize the file path for consistent labels
                label = os.path.relpath(root, root_directory)
                
                data.append(melspectrogram)
                labels.append(label)
                
                print(f'Processed {audio_path}')
    
    return np.array(data), np.array(labels)


def save_melspectrogram_data(data, labels, output_path):
    np.savez(output_path, data=data, labels=labels)
    print(f'Saved data to {output_path}')


# Example usage:
train_root_directory = 'Dataset/Train'
test_root_directory = 'Datraset/Test'
train_output_path = 'Extracted_data/train_melspectrogram_data.npz'
test_output_path = 'Extracted_data/test_melspectrogram_data.npz'

# Process training data
train_data, train_labels = process_audio_files(train_root_directory)
save_melspectrogram_data(train_data, train_labels, train_output_path)

# Process testing data
test_data, test_labels = process_audio_files(test_root_directory)
save_melspectrogram_data(test_data, test_labels, test_output_path)

