from packages.utils import json, tf, np, math, os, librosa
from packages.utils import VALIDATION_SPLIT, BATCH_SIZE, SAMPLE_RATE, DURATION
from packages.utils import aug_train_json, train_json, test_json, ww_aug_train_json, ww_train_json, ww_test_json
from packages.utils import aug_train_data_dir, train_data_dir, test_data_dir, ww_aug_train_data_dir, ww_train_data_dir, ww_test_data_dir


# ----------------------------------------------------
# Parameters for MFCCs
N_MFCC = 13
HOP_LENGTH = 512
N_FFT = 2048
NUM_SEGMENTS = 5
SAMPLES_PER_AUDIO = SAMPLE_RATE * DURATION


# ----------------------------------------------------------------
# Function to load train and validation datasets
def load_train_json_dataset(json_path, batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT):
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
def load_test_json_dataset(json_path, batch_size=BATCH_SIZE):
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
def extract_mfcc_create_json_file(dataset_path, json_path, num_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH, num_segments=NUM_SEGMENTS):
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
        if dirpath == dataset_path:
            continue  # Skip the root folder

        # Save genre label (i.e., sub-folder name) in the mapping
        semantic_label = os.path.basename(dirpath)
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


def main():
    # extract_mfcc_create_json_file(aug_train_data_dir, aug_train_json)
    # extract_mfcc_create_json_file(train_data_dir, train_json)
    extract_mfcc_create_json_file(test_data_dir, test_json)

    # extract_mfcc_create_json_file(ww_aug_train_data_dir, ww_aug_train_json)
    # extract_mfcc_create_json_file(ww_train_data_dir, ww_train_json)
    # extract_mfcc_create_json_file(ww_test_data_dir, ww_test_json)


if __name__ == '__main__':
    main()
