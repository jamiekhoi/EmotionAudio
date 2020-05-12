import librosa
from scipy.io import wavfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def envelope(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/10), min_periods=1, center=True).mean()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask


def downsample_and_envelope(PROCESSED_PATH, kind):
    df = pd.read_csv(PROCESSED_PATH + 'ravdess{}.csv'.format(kind))
    df.set_index('FilePath', inplace=True)
    for f in df.index:
        signal, rate = librosa.load(f, sr=16000)  # downsampling to 16khz
        mask = envelope(signal, rate, 0.0005)
        # Remove point under the mask threshold
        new_signal = signal[mask]

        # Make clean directory if not exists
        if not os.path.exists(PROCESSED_PATH + 'clean/'):
            os.makedirs(PROCESSED_PATH + 'clean/')
        # Equivenlent lines
        librosa.output.write_wav(path=PROCESSED_PATH + 'clean/' + df.at[f, 'Filename'], y=new_signal, sr=rate)
        # wavfile.write(filename=PROCESSED_PATH + 'cleanspeech/' + df.at[f, 'Filename'], rate=rate, data=new_signal)


def main():
    """
    Cleans audio files by removing "dead" sounds and saving the "cleaned" files
    :return:
    """
    parser = argparse.ArgumentParser()
    #parser.add_argument('--makecsv', action='store_true')
    parser.add_argument('--dataset', required=True, choices=['ravdesssong', 'ravdessspeech'])  # , default=os.path.expanduser('~/tacotron'))
    args = parser.parse_args()

    if args.dataset == 'ravdesssong':
        print('Making Ravdess Song')
        downsample_and_envelope(PROCESSED_PATH='Processed/RAVDESSsong/', kind='Song')
    elif args.dataset == 'ravdessspeech':
        print('Making Ravdess Speech')
        downsample_and_envelope(PROCESSED_PATH='Processed/RAVDESSspeech/', kind='Speech')
    else:
        print('No option')


if __name__ == '__main__':
    main()
