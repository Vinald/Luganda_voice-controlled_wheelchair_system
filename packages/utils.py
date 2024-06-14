from packages.common_packages import *


# ---------------------------------------------------------------------------
# Function to add white noise and picth scaling
def modify_audio(root_folder, destination_folder):
    # Create the destination directory if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)

    # Loop through all subfolders
    for subdir, dirs, files in os.walk(root_folder):
        for file in files:
            # Check if the file is an audio file
            if file.endswith('.wav'):
                file_path = os.path.join(subdir, file)
                
                # Load the audio file
                audio, sr = librosa.load(file_path, sr=None)

                # Pad the audio to 2 seconds
                target_length = sr * 2  # 2 seconds
                if len(audio) < target_length:
                    padding = np.zeros(target_length - len(audio))
                    audio = np.concatenate((audio, padding))

                # Add white noise
                noise = np.random.normal(0, 0.03, audio.shape) 
                audio_with_noise = audio + noise

                # Apply pitch scaling of 3
                audio_pitch_scaled = librosa.effects.pitch_shift(audio, sr=sr, n_steps=2)

                # Create a new directory for the modified audio files inside the destination folder
                relative_subdir = os.path.relpath(subdir, root_folder)
                new_dir = os.path.join(destination_folder, relative_subdir)
                os.makedirs(new_dir, exist_ok=True)

                # Save the modified audio with noise to the new directory
                new_file_path_with_noise = os.path.join(new_dir, f'noisy_{file}')
                sf.write(new_file_path_with_noise, audio_with_noise, sr)
                print(f'Saved {new_file_path_with_noise}')

                # Save the pitch scaled audio to the new directory
                new_file_path_pitch_scaled = os.path.join(new_dir, f'pitch_scaled_{file}')
                sf.write(new_file_path_pitch_scaled, audio_pitch_scaled, sr)
                print(f'Saved {new_file_path_pitch_scaled}')


# aug_train_data_dir = pathlib.Path('Dataset/speech_intent_classification/New_Train')
# train_data_dir = pathlib.Path('Dataset/speech_intent_classification/Train')


# # File path for the wake word model
# ww_aug_train_data_dir = pathlib.Path('Dataset/wake_word/New_Train')
# ww_train_data_dir = pathlib.Path('Dataset/wake_word/Train')

    
# modify_audio(train_data_dir, aug_train_data_dir)
# modify_audio(ww_train_data_dir, ww_aug_train_data_dir)


# ---------------------------------------------------------------------------
# Function to rename the aduio files to their directory name
def rename_audio_files(root_folder):
    # Loop through all subfolders
    for subdir, dirs, files in os.walk(root_folder):
        for i, file in enumerate(sorted(files)):
            # Check if the file is an audio file
            if file.endswith('.wav'):
                file_path = os.path.join(subdir, file)
                
                # Get the name of the subfolder
                subfolder_name = os.path.basename(subdir)
                
                # Create new file name with ascending order
                new_file_name = f"{subfolder_name}__{i+1}.wav"
                new_file_path = os.path.join(subdir, new_file_name)
                
                # Rename the file
                os.rename(file_path, new_file_path)
                print(f'Renamed {file_path} to {new_file_path}')


# train_data_dir = pathlib.Path('Dataset/speech_intent_classification/Train')
# rename_audio_files(train_data_dir)


# ---------------------------------------------------------------------------
# Function to print the directory structure and labels
def print_directory_tree(root_dir, indent=''):
    print(indent + os.path.basename(root_dir) + os.path.sep)
    indent += '    '
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        if os.path.isdir(item_path):
            print_directory_tree(item_path, indent)


def list_directory_contents(directory, label):
    contents = np.array(tf.io.gfile.listdir(str(directory)))
    print(f'{label} commands labels: {contents}')
    return contents


# ---------------------------------------------------------------------------
# Function to get the file size
def get_and_convert_file_size(file_path, unit=None):
    size = os.path.getsize(file_path)
    if unit == "KB":
        return print('File size: ' + str(round(size / 1024, 3)) + ' Kilobytes')
    elif unit == "MB":
        return print('File size: ' + str(round(size / (1024 * 1024), 3)) + ' Megabytes')
    else:
        return print('File size: ' + str(size) + ' bytes')


# ---------------------------------------------------------------------------
# Function to get the labels
def get_label_names():
    label_names_slice = ['ddyo', 'emabega', 'gaali', 'kkono', 'mumaaso', 'unknown', 'yimirira']
    return label_names_slice


