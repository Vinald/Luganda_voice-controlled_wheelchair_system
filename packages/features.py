from packages.common_packages import *


# ---------------------------------------------------------------------------
# Function to create train and validation audio datasets
def create_train_audio_dataset(data_dir, batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT, seed=SEED, output_sequence_length=SAMPLE_RATE):
    train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
        directory=data_dir,
        batch_size=batch_size,
        validation_split=validation_split,
        seed=seed,
        output_sequence_length=output_sequence_length,
        subset='both'
    )

    label_names = np.array(train_ds.class_names)

    def squeeze(audio, labels):
        audio = tf.squeeze(audio, axis=-1)
        return audio, labels

    train_ds = train_ds.map(squeeze, tf.data.AUTOTUNE)
    val_ds = val_ds.map(squeeze, tf.data.AUTOTUNE)

    for example_audio, example_labels in train_ds.take(1):
        print(example_audio.shape)
        print(example_labels.shape)
    
    train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, label_names


# ---------------------------------------------------------------------------
# Function to create test audio dataset
def create_test_audio_dataset(data_dir, batch_size=BATCH_SIZE, output_sequence_length=SAMPLE_RATE):
    test_ds = tf.keras.utils.audio_dataset_from_directory(
        directory=data_dir,
        batch_size=batch_size,
        validation_split=None,
        seed=0,
        output_sequence_length=output_sequence_length,
        shuffle=False
    )

    def squeeze(audio, labels):
        audio = tf.squeeze(audio, axis=-1)
        return audio, labels

    test_ds = test_ds.map(squeeze, tf.data.AUTOTUNE)

    for example_audio, example_labels in test_ds.take(1):
        print(example_audio.shape)
        print(example_labels.shape)

    test_ds = test_ds.cache().prefetch(tf.data.AUTOTUNE)

    return test_ds


# ---------------------------------------------------------------------------
# Function to create mel spectrogram dataset
def preprocess_melspec_audio_datasets(train_ds, val_ds, test_ds):
    def get_mel_spectrogram(waveform, sample_rate=SAMPLE_RATE, n_mels=128):
        stft = tf.signal.stft(waveform, frame_length=255, frame_step=128)
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

    def make_mel_spec_ds(ds):
        return ds.map(lambda x, y: (get_mel_spectrogram(x), y), tf.data.AUTOTUNE)

    train_mel_spec_ds = make_mel_spec_ds(train_ds)
    val_mel_spec_ds = make_mel_spec_ds(val_ds)
    test_mel_spec_ds = make_mel_spec_ds(test_ds)

    train_mel_spec_ds = train_mel_spec_ds.cache().shuffle(10000).prefetch(tf.data.AUTOTUNE)
    val_mel_spec_ds = val_mel_spec_ds.cache().prefetch(tf.data.AUTOTUNE)
    test_mel_spec_ds = test_mel_spec_ds.cache().prefetch(tf.data.AUTOTUNE)

    return train_mel_spec_ds, val_mel_spec_ds, test_mel_spec_ds
