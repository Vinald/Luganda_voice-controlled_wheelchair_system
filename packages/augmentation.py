from packages.utils import librosa, os, sf, np
from packages.utils import aug_train_data_dir, train_data_dir, ww_aug_train_data_dir, ww_train_data_dir


def main():
    modify_audio(train_data_dir, aug_train_data_dir)
    modify_audio(ww_train_data_dir, ww_aug_train_data_dir)


# ---------------------------------------------------------------------------
# Function to add white noise and pitch scaling
# ---------------------------------------------------------------------------
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
                target_length = sr * 2
                if len(audio) < target_length:
                    padding = np.zeros(target_length - len(audio))
                    audio = np.concatenate((audio, padding))

                # Add white noise of 0.003
                noise = np.random.normal(0, 0.03, audio.shape) 
                audio_with_noise = audio + noise

                # Apply pitch scaling of 2
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


if __name__ == '__main__':
    main()
