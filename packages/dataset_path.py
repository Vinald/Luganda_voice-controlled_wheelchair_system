from common_packages import pathlib

# File paths for the speech classification dataset
aug_train_data_dir = pathlib.Path('Dataset/speech_intent_classification/New_Train')
train_data_dir = pathlib.Path('Dataset/speech_intent_classification/Train')
test_data_dir = pathlib.Path('Dataset/speech_intent_classification/Test')
train_data_needs_preprocessing = pathlib.Path('Dataset/speech_intent_classification/Train_need_preprocessing')
test_data_needs_preprocessing = pathlib.Path('Dataset/speech_intent_classification/Test_need_preprocessing')


# File Path for the wake word model
ww_aug_train_data_dir = pathlib.Path('Dataset/wake_word/New_Train')
ww_train_data_dir = pathlib.Path('Dataset/wake_word/Train')
ww_test_data_dir = pathlib.Path('Dataset/wake_word/Test')
ww_train_data_needs_preprocessing = pathlib.Path('Dataset/wake_word/Train_need_preprocessing')
ww_test_data_needs_preprocessing = pathlib.Path('Dataset/wake_word/Test_need_preprocessing')


# File path for sic CSV files
sic_aug_train_csv_dir = pathlib.Path('files/sic_aug_train.csv')
sic_train_csv_dir = pathlib.Path('files/sic_train.csv')
sic_test_csv_dir = pathlib.Path('files/sic_test.csv')


# File path for the wake word files
ww_aug_train_csv_dir = pathlib.Path('files/ww_aug_train.csv')
ww_train_csv_dir = pathlib.Path('files/ww_train.csv')
ww_test_csv_dir = pathlib.Path('files/ww_test.csv')


# Mfcc json files
aug_train_json = pathlib.Path('json/mfcc_aug_train_data.json')
train_json = pathlib.Path('json/mfcc_train_data.json')
test_json = pathlib.Path('json/mfcc_test_data.json')

ww_aug_train_json = pathlib.Path('json/ww_mfcc_aug_train_data.json')
ww_train_json = pathlib.Path('json/ww_mfcc_train_data.json')
ww_test_json = pathlib.Path('json/ww_mfcc_test_data.json')
