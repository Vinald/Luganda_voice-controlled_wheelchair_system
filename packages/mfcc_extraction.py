from packages.common_packages import os, librosa, json
from packages.common_packages import N_MFCC, N_FFT, HOP_LENGTH, SAMPLE_RATE

from packages.dataset_path import aug_train_data_dir, train_data_dir, test_data_dir, ww_aug_train_data_dir, ww_test_data_dir, ww_train_data_dir, aug_train_json, train_json, test_json, ww_aug_train_json, ww_train_json, ww_test_json


def main():
    prepare_dataset(aug_train_data_dir, aug_train_json)
    prepare_dataset(train_data_dir, train_json)
    prepare_dataset(test_data_dir, test_json)

    prepare_dataset(ww_aug_train_data_dir, ww_aug_train_json)
    prepare_dataset(ww_train_data_dir, ww_train_json)
    prepare_dataset(ww_test_data_dir, ww_test_json)


def prepare_dataset(dataset_path, json_path, n_mfcc=N_MFCC, hop_length=HOP_LENGTH, n_fft=N_FFT):
    data = {
        'mappings': [],
        'labels': [],
        'MFCCs': [],
        'files': []
    }

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        # check if we are not at the root director
        if dirpath is not dataset_path:
            # update the mappings
            category = dirpath.split("/")[-1]
            data['mappings'].append(category)
            print(f"Processing {category}")

            # loop through all the filenames and extract the MFCCs
            for f in filenames:
                file_path = os.path.join(dirpath, f)
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                # ensure the audio file is at least 2 second
                if len(signal) >= SAMPLE_RATE:
                    # ensure the signal is at least 2 second
                    signal = signal[:SAMPLE_RATE]

                    # extract the MFCCs
                    MFCCs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
                    data['MFCCs'].append(MFCCs.T.tolist())
                    data['labels'].append(i-1)
                    data['files'].append(file_path)
                    print(f"{file_path}: {i-1}")
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


# def save_mfcc(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=2):

#     # dictionary to store mapping, labels, and MFCCs
#     data = {
#         "mapping": [],
#         "labels": [],
#         "mfcc": []
#     }

#     samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
#     num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

#     # loop through all genre sub-folder
#     for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

#         # ensure we're processing a genre sub-folder level
#         if dirpath is not dataset_path:

#             # save genre label (i.e., sub-folder name) in the mapping
#             semantic_label = dirpath.split("/")[-1]
#             data["mapping"].append(semantic_label)
#             print("\nProcessing: {}".format(semantic_label))

#             # process all audio files in genre sub-dir
#             for f in filenames:

# 		# load audio file
#                 file_path = os.path.join(dirpath, f)
#                 signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

#                 # process all segments of audio file
#                 for d in range(num_segments):

#                     # calculate start and finish sample for current segment
#                     start = samples_per_segment * d
#                     finish = start + samples_per_segment

#                     # extract mfcc
#                     mfcc = librosa.feature.mfcc(y=signal[start:finish], sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)

#                     # mfcc = librosa.feature.mfcc(signal[start:finish], sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)

#                     # mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
#                     mfcc = mfcc.T

#                     # store only mfcc feature with expected number of vectors
#                     if len(mfcc) == num_mfcc_vectors_per_segment:
#                         data["mfcc"].append(mfcc.tolist())
#                         data["labels"].append(i-1)
#                         print("{}, segment:{}".format(file_path, d+1))

#     # save MFCCs to json file
#     with open(json_path, "w") as fp:
#         json.dump(data, fp, indent=4)


if __name__ == "__main__":
    main()
