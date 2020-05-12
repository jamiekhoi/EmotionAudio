import argparse
import os
import xlrd
import numpy as np
import random
import librosa
import skimage
import skimage.io
import shutil

joinpath = os.path.join
listdir = os.listdir


IDENTIFIERS = ['Modality',
               'Vocal channel',
               'Emotion',
               'Emotional intensity',
               'Statment',
               'Repitition',
               'Actor']

EMOTIONS = {
    '01': 'Neutral',
    '02': 'Calm',
    '03': 'Happy',
    '04': 'Sad',
    '05': 'Angry',
    '06': 'Fearful',
    '07': 'Disgust',
    '08': 'Surprised'
}


def get_id(id):
    return IDENTIFIERS.index(id)


def make_csv_info_file(args, data_type_folder, output_file):
    song_durations = {}
    sheet = xlrd.open_workbook(joinpath(args.datasetpath, 'dsd100.xlsx')).sheet_by_index(0)
    for i in range(1, sheet.nrows):
        duration_text = sheet.cell(i, 2).value.split("'")
        duration_in_seconds = int(duration_text[0]) * 60 + int(duration_text[1])
        song_durations[sheet.cell(i, 0).value] = str(duration_in_seconds)

    # Collect path information
    csv_list = []
    folder_path = joinpath(args.datasetpath, 'All', data_type_folder)
    folders = listdir(folder_path)
    for song_folder in folders:
        csv_row = []
        for song_folder_subfile in []:
            csv_row.append(joinpath(folder_path, song_folder, song_folder_subfile))
        csv_row.append(song_durations[song_folder[6:]])
        csv_list.append(csv_row)

    # Write to file. File paths are relative to where script was started
    # csv file with be created inside datasetpath directory
    with open(joinpath(args.datasetpath, output_file), 'w') as handle:
        for line in csv_list:
            handle.write(','.join(line) + '\n')


def convert_to_mel_spectrogram():
    pass


def extract_features(file_name):
    """
    Taken from https://medium.com/@mikesmales/sound-classification-using-deep-learning-8bc2aa1990b7
    :param file_name:
    :return:
    """
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccsscaled = np.mean(mfccs.T, axis=0)

    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None

    return mfccsscaled


def scale_minmax(X, min=0.0, max=1.0):
    """Taken from somewhere"""
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled


def create_and_save_thing(actor, dataset_path, file, OUTPUT_PATH):
    audio, sample_rate = librosa.load(joinpath(dataset_path, actor, file))
    # y, sr = librosa.load(TEST_FILE_PATH)
    S = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    # min-max scale to fit inside 8-bit range
    img = scale_minmax(S_dB, 0, 255).astype(np.uint8)
    img = np.flip(img, axis=0)  # put low frequencies at the bottom in image
    img = 255 - img  # invert. make black==more energy

    # Get emotion
    emotion_id = file.split('-')[get_id('Emotion')]
    # save as PNG
    skimage.io.imsave(joinpath(OUTPUT_PATH, EMOTIONS[emotion_id],os.path.splitext(file)[0] + '.png'), img)


def main():
    """
    create png files from audio. Dont use this one
    :return:
    """
    parser = argparse.ArgumentParser()
    #parser.add_argument('--makecsv', action='store_true')
    parser.add_argument('--dataset', required=True, choices=['ravdess'])  # , default=os.path.expanduser('~/tacotron'))
    parser.add_argument('--slack_url', help='Slack webhook URL to get periodic reports.')

    parser.add_argument('--test_split', type=float, default=0.2)
    parser.add_argument('--independent', action='store_true')  # Actor either in training or validation set, not both
    parser.add_argument('--normalize_volume', action='store_true')  # Normalize volume (amplitude?) of files. Anger
    # may be louder than sad f.ex. Does this affect or bias the training to this instead of emotion?

    args = parser.parse_args()

    if args.dataset == 'ravdess':
        TRAINING_OUTPUT_PATH = 'training'
        VALIDATION_OUTPUT_PATH = 'validation'

        dataset_path = '../Audio_Speech_Actors_01-24'
        actors = listdir(dataset_path)
        test_actors = random.sample(actors, int(len(actors)*args.test_split))
        train_actors = [actor for actor in actors if actor not in test_actors]

        if os.path.exists(TRAINING_OUTPUT_PATH):
            shutil.rmtree(TRAINING_OUTPUT_PATH)
        os.mkdir(TRAINING_OUTPUT_PATH)
        for emotion in EMOTIONS.values():
            os.mkdir(joinpath(TRAINING_OUTPUT_PATH, emotion))
        for actor in train_actors:
            actor_dir = listdir(joinpath(dataset_path, actor))
            for file in actor_dir:
                create_and_save_thing(actor, dataset_path, file, TRAINING_OUTPUT_PATH)

        if os.path.exists(VALIDATION_OUTPUT_PATH):
            shutil.rmtree(VALIDATION_OUTPUT_PATH)
        os.mkdir(VALIDATION_OUTPUT_PATH)
        for emotion in EMOTIONS.values():
            os.mkdir(joinpath(VALIDATION_OUTPUT_PATH, emotion))
        for actor in test_actors:
            actor_dir = listdir(joinpath(dataset_path, actor))
            for file in actor_dir:
                create_and_save_thing(actor, dataset_path, file, VALIDATION_OUTPUT_PATH)


if __name__ == '__main__':
    main()

