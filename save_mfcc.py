from packages.common_packages import *


def save_mfcc(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
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
                    # Calculate start and finish sample for the current segment
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


# Paths
new_train_data_dir = 'Dataset/New_Train'
train_data_dir = 'Dataset/Train'
test_data_dir = 'Dataset/Test'

new_train_json_path = 'new_train_mfccs.json'
train_json_path = 'train_mfccs.json'
test_json_path = 'test_mfccs.json'


new_train_json_path_10 = 'new_train_mfccs_10.json'
train_json_path_10 = 'train_mfccs_10.json'
test_json_path_10 = 'test_mfccs_10.json'



# Save MFCCs for training and test datasets
new_train_data = save_mfcc(new_train_data_dir, new_train_json_path)
train_data = save_mfcc(train_data_dir, train_json_path)
test_data = save_mfcc(test_data_dir, test_json_path)


new_train_data_10 = save_mfcc(new_train_data_dir, new_train_json_path_10)
train_data_10 = save_mfcc(train_data_dir, train_json_path_10)
test_data_10 = save_mfcc(test_data_dir, test_json_path_10)


