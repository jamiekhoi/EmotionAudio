import os
from tqdm import tqdm
import numpy as np
from scipy.io import wavfile
import pickle
from python_speech_features import mfcc
from keras.utils import to_categorical
from cfg import AudioConfig
import pandas as pd

def check_data(audio_config):
    if os.path.isfile(audio_config.p_path):
        print('Loading existing data for {} model'.format(audio_config.mode))
        with open(audio_config.p_path, 'rb') as handle:
            tmp = pickle.load(handle)
            return tmp
    else:
        return None

def build_rand_feat(audio_config, n_samples, classes, class_dist, prob_dist, df, MAIN_PATH):
    tmp = check_data(audio_config)
    if tmp:
        return tmp.data[0], tmp.data[1]

    X = []
    y = []
    _min, _max = float('inf'), -float('inf')
    for _ in tqdm(range(n_samples)):
        # Get random class
        rand_class = np.random.choice(class_dist.index, p=prob_dist)
        # Grab a file belonging to the random class
        file = np.random.choice(df[df.Emotion == rand_class].index)
        rate, wav = wavfile.read(MAIN_PATH + 'clean/' + file)
        label = df.at[file, 'Emotion']
        # Start to grab a random 10 millisecond interval from the file
        rand_index = np.random.randint(0, wav.shape[0] - audio_config.step)
        sample = wav[rand_index:rand_index + audio_config.step]
        # Get MFCC
        X_sample = mfcc(sample, rate,
                        numcep=audio_config.nfeat, nfilt=audio_config.nfilt, nfft=audio_config.nfft)
        # Update _min and _max
        _min = min(np.amin(X_sample), _min)
        _max = max(np.amax(X_sample), _max)

        X.append(X_sample)
        y.append(classes.index(label))  # label is an integer (index posistion of label)
    audio_config.min = _min
    audio_config.max = _max
    X, y = np.array(X), np.array(y)
    # Normalize X to rescale between 0 and 1
    X = (X - _min) / (_max - _min)

    # FOR TESTING
    pre_reshapeX = X
    #####

    if audio_config.mode == 'conv' or audio_config.mode == 'caps':
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    # For recurrent network
    elif audio_config.mode == 'time' or audio_config.mode == 'capsrnn':
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2])
    else:
        print('MODE NOT DEFINED FOR FEATURE BUILDER')
        raise ValueError

    # Because of the cost function for the neural network (categorical crossentropi?) we
    # one hot encode integer labels to a matrix
    y = to_categorical(y, num_classes=len(classes))

    audio_config.data = (X, y)
    with open(audio_config.p_path, 'wb') as handle:
        pickle.dump(audio_config, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return X, y

def get_training_data_conv():
    MAIN_PATH = 'Processed/RAVDESSspeech/'
    df = pd.read_csv(MAIN_PATH + 'ravdessSpeech.csv')
    df.set_index('Filename', inplace=True)

    # Add length column for each file
    for f in df.index:
        rate, signal = wavfile.read(MAIN_PATH + 'clean/' + f)
        df.at[f, 'length'] = signal.shape[0] / rate

    classes = list(np.unique(df.Emotion))
    class_dist = df.groupby(['Emotion'])['length'].mean()
    n_samples = int(df['length'].sum() / 0.1)  # change this?
    prob_dist = class_dist / class_dist.sum()
    audio_config = AudioConfig(data_save_path=MAIN_PATH, mode='caps')

    if audio_config.mode == 'caps':
        X_, y_ = build_rand_feat(audio_config, n_samples, classes, class_dist, prob_dist, df, MAIN_PATH)
        # y_flat = np.argmax(y_, axis=1)  # one-hot to index
        # input_shape = (X_.shape[1], X_.shape[2], 1)
    return X_, y_

def get_training_data_time():
    MAIN_PATH = 'Processed/RAVDESSspeech/'
    df = pd.read_csv(MAIN_PATH + 'ravdessSpeech.csv')
    df.set_index('Filename', inplace=True)

    # Add length column for each file
    for f in df.index:
        rate, signal = wavfile.read(MAIN_PATH + 'clean/' + f)
        df.at[f, 'length'] = signal.shape[0] / rate

    classes = list(np.unique(df.Emotion))
    class_dist = df.groupby(['Emotion'])['length'].mean()
    n_samples = int(df['length'].sum() / 0.1)  # change this?
    prob_dist = class_dist / class_dist.sum()
    audio_config = AudioConfig(data_save_path=MAIN_PATH, mode='capsrnn')

    if audio_config.mode == 'capsrnn':
        X_, y_ = build_rand_feat(audio_config, n_samples, classes, class_dist, prob_dist, df, MAIN_PATH)
        # y_flat = np.argmax(y_, axis=1)  # one-hot to index
        # input_shape = (X_.shape[1], X_.shape[2], 1)
    return X_, y_