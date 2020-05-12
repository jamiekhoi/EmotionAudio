import os
class AudioConfig:
    def __init__(self, data_save_path='', mode='caps', nfilt=26,
                 nfeat=13, nfft=512, rate=16000, step_seconds=0.4):
        self.rate = rate
        self.nfft = nfft
        self.nfeat = nfeat
        self.nfilt = nfilt
        self.mode = mode
        self.step = int(rate*step_seconds) #
        # newer lines
        self.model_path = os.path.join(data_save_path, mode + '.model')
        self.p_path = os.path.join(data_save_path, mode + '.p')