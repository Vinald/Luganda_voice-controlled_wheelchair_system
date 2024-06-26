from common_packages import os, librosa, json, math, pathlib
from common_packages import N_MFCC, N_FFT, HOP_LENGTH, SAMPLE_RATE, SAMPLES_PER_AUDIO

# File paths for the speech classification dataset
aug_train_data_dir = pathlib.Path('Dataset/speech_intent_classification/New_Train')
train_data_dir = pathlib.Path('Dataset/speech_intent_classification/Train')
test_data_dir = pathlib.Path('Dataset/speech_intent_classification/Test')

# File Path for the wake word model
ww_aug_train_data_dir = pathlib.Path('Dataset/wake_word/New_Train')
ww_train_data_dir = pathlib.Path('Dataset/wake_word/Train')
ww_test_data_dir = pathlib.Path('Dataset/wake_word/Test')

# Mfcc json files
aug_train_json = pathlib.Path('json/mfcc_aug_train_data.json')
train_json = pathlib.Path('json/mfcc_train_data.json')
test_json = pathlib.Path('json/mfcc_test_data.json')

ww_aug_train_json = pathlib.Path('json/ww_mfcc_aug_train_data.json')
ww_train_json = pathlib.Path('json/ww_mfcc_train_data.json')
ww_test_json = pathlib.Path('json/ww_mfcc_test_data.json')


def main():
    prepare_dataset(aug_train_data_dir, aug_train_json)
    prepare_dataset(train_data_dir, train_json)
    prepare_dataset(test_data_dir, test_json)

    prepare_dataset(ww_aug_train_data_dir, ww_aug_train_json)
    prepare_dataset(ww_train_data_dir, ww_train_json)
    prepare_dataset(ww_test_data_dir, ww_test_json)


def prepare_dataset(dataset_path, json_path, num_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH, num_segments=5):
    # Dictionary to store mapping, labels, and MFCCs
    data = {
        "mapping": [],
        "labels": [],
        "mfcc": []
    }

    samples_per_segment = int(SAMPLES_PER_AUDIO / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    # Loop through all sub-folders
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        # Ensure we're processing the sub-folder level
        if dirpath != dataset_path:
            # Save genre label (i.e., sub-folder name) in the mapping
            semantic_label = dirpath.split("/")[-1]
            data["mapping"].append(semantic_label)
            print(f"\nProcessing: {semantic_label}")

            # Process all audio files in genre sub-dir
            for f in filenames:
                # Load audio file
                file_path = os.path.join(dirpath, f)
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                # Process all segments of audio file
                for d in range(num_segments):
                    start = samples_per_segment * d
                    finish = start + samples_per_segment

                    # Extract MFCC
                    mfcc = librosa.feature.mfcc(y=signal[start:finish],
                                                sr=sr,
                                                n_mfcc=num_mfcc,
                                                n_fft=n_fft,
                                                hop_length=hop_length)
                    mfcc = mfcc.T

                    # Store only MFCC feature with the expected number of vectors
                    if len(mfcc) == num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i - 1)
                        print(f"{file_path}, segment:{d+1}")

    # Save MFCCs to JSON file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

    return data

if __name__ == '__main__':
    main()
