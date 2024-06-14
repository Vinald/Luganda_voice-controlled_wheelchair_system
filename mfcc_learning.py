from packages.common_imports import *
from packages.dataset_path import *


def prepare_dataset(dataset_path, json_path, n_mfcc=13, hop_length=512, n_fft=2048):
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


if __name__ == "__main__":
    prepare_dataset(TRAIN_DATASET_PATH, TRAIN_JSON_PATH)
    prepare_dataset(TEST_DATASET_PATH, TEST_JSON_PATH)