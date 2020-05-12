import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank

def calculate_fft(signal, rate):
    """Calculate Fast Fourier Transform"""
    n = len(signal)
    frequency = np.fft.rfftfreq(n, d=1/rate)
    magnitude = abs(np.fft.rfft(signal)/n) # Magnitude also respresented as Y and signal as y
    return (magnitude, frequency)


def my_logfbank(signal, rate, nfilt, nfft):
    # nfft is the window size, usually 25ms, so sampling_rate*0.025. nfft will be padded with 0 to match a power of 2
    bank = logfbank(signal, rate, nfilt=nfilt, nfft=nfft)
    return bank


def freq_to_mel(f):
    return 1127*np.log(1 + (f/700))


def mel_to_freq(mel):
    return 700*((np.e**(mel/1127)) - 1)