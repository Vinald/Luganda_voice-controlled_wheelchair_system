from packages.utils import *


def main():
    extract_mfcc(aug_train_data_dir, aug_train_json)
    extract_mfcc(train_data_dir, train_json)
    extract_mfcc(test_data_dir, test_json)

    extract_mfcc(ww_aug_train_data_dir, ww_aug_train_json)
    extract_mfcc(ww_train_data_dir, ww_train_json)
    extract_mfcc(ww_test_data_dir, ww_test_json)


# ----------------------------------------------------------------
# Function to load train and validation datasets
def load_train_dataset(json_path, batch_size, validation_split=0.2):
    # Load MFCCs from JSON and create TensorFlow dataset
    with open(json_path, "r") as fp:
        data = json.load(fp)

    mfcc = np.array(data["mfcc"])
    labels = np.array(data["labels"])

    dataset = tf.data.Dataset.from_tensor_slices((mfcc, labels))
    dataset = dataset.shuffle(len(mfcc)).batch(batch_size)

    train_size = int((1 - validation_split) * len(mfcc))
    train_ds = dataset.take(train_size)
    val_ds = dataset.skip(train_size)

    train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, data["mapping"]


# ----------------------------------------------------------------
# Function to load test dataset
def load_test_dataset(json_path, batch_size):
    # Load MFCCs from JSON and create TensorFlow dataset
    with open(json_path, "r") as fp:
        data = json.load(fp)

    mfcc = np.array(data["mfcc"])
    labels = np.array(data["labels"])

    dataset = tf.data.Dataset.from_tensor_slices((mfcc, labels))
    dataset = dataset.shuffle(len(mfcc)).batch(batch_size)

    test_ds = dataset.cache().prefetch(tf.data.AUTOTUNE)

    return test_ds, data["mapping"]


# ----------------------------------------------------------------
# Extract mfccs
def extract_mfcc(dataset_path, json_path, num_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH, num_segments=NUM_SEGMENTS):
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


# ----------------------------------------------------------------
# Load a json file
def load_data(json_path):
    with open(json_path, "r") as fp:
        data = json.load(fp)
    return data


def save_mfcc(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=6):

    # dictionary to store mapping, labels, and MFCCs
    data = {
        "mapping": [],
        "labels": [],
        "mfcc": []
    }

    samples_per_segment = int(SAMPLES_PER_AUDIO / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    # loop through all sub-folders
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # ensure we're processing the sub-folder level
        if dirpath is not dataset_path:

            # save genre label (i.e., sub-folder name) in the mapping
            semantic_label = dirpath.split("/")[-1]
            data["mapping"].append(semantic_label)
            print(f"\nProcessing: {semantic_label}")

            # process all audio files in genre sub-dir
            for f in filenames:

                # load audio file
                file_path = os.path.join(dirpath, f)
                signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

                # process all segments of audio file
                for d in range(num_segments):

                    # calculate start and finish sample for the current segment
                    start = samples_per_segment * d
                    finish = start + samples_per_segment

                    # extract mfcc
                    mfcc = librosa.feature.mfcc(y=signal[start:finish],
                                                sr=sample_rate,
                                                n_mfcc=num_mfcc,
                                                n_fft=n_fft,
                                                hop_length=hop_length)

                    mfcc = mfcc.T

                    # store only mfcc feature with the expected number of vectors
                    if len(mfcc) == num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i-1)
                        print(f"{file_path}, segment:{d+1}")

    # save MFCCs to json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

    # return the data dictionary
    return data


data = save_mfcc(TRAIN_DATASET_PATH, JSON_PATH, num_segments=6)


def load_data(data):

    # convert lists to numpy arrays
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])

    print("Data successfully loaded!")

    return X, y


X, y = load_data(data)



# Function to prepare dataset
def prepare_dataset(data_dir, json_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=2):
    data = {
        "mapping": [],
        "mfcc": [],
        "labels": [],
        "files": []
    }

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(data_dir)):
        if dirpath is not data_dir:
            semantic_label = dirpath.split("/")[-1]
            data["mapping"].append(semantic_label)
            print("\nProcessing: {}".format(semantic_label))

            for f in filenames:
                file_path = os.path.join(dirpath, f)
                signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE, duration=TRACK_DURATION)

                if len(signal) >= SAMPLE_RATE:
                    signal = signal[:SAMPLE_RATE]

                    for d in range(num_segments):
                        start = samples_per_segment * d
                        finish = start + samples_per_segment

                        mfcc = librosa.feature.mfcc(y=signal[start:finish], sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
                        mfcc = mfcc.T

                        if len(mfcc) == num_mfcc_vectors_per_segment:
                            data["mfcc"].append(mfcc.tolist())
                            data["labels"].append(i-1)
                            data["files"].append(file_path)
                            print("{}, segment:{}".format(file_path, d+1))

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


if __name__ == '__main__':
    # main()
    ...
