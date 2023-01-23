import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
import librosa
import librosa.display
import IPython.display as ipd
import tensorflow as tf

from itertools import cycle

class Silvertone(object):
    x = None
    sr = None
    S_db_mel = None
    spectral_centroid = None
    mfcc = None
    chroma_stft = None
    tonnetz = None
    trim=None

    def __init__ (self, audio, mels,trim, *args, **kwargs):
        if not trim:
            audio_file = audio
            self.x, self.sr = librosa.load(audio_file)
        else:
            audio_file = audio
            self.x, self.sr = librosa.load(audio_file)
            self.x = librosa.effects.trim(self.x,top_db=35) #trimming data
            if self.x[0].shape[0] >= 65000:
                self.x = self.x[0][:65000]
                self.x = tf.convert_to_tensor(self.x).numpy()
            else:
                zero_padding = tf.zeros([65000]-tf.shape(self.x[0]),dtype=tf.float32)
                self.x = tf.concat([self.x[0],zero_padding],0).numpy()
        self.S = librosa.feature.melspectrogram(y=self.x, sr=self.sr, n_mels=mels)
        self.S_db_mel = librosa.amplitude_to_db(self.S, ref=np.max)
        self.spectral_centroid = librosa.feature.spectral_centroid(y=self.x, sr=self.sr, S=self.S)
        self.mfcc = np.mean(librosa.feature.mfcc(y=self.x, sr=self.sr, n_mfcc=20), axis=0)
        self.chroma_stft = librosa.feature.chroma_stft(y=self.x, sr=self.sr, S=self.S)
        self.tonnetz = librosa.feature.tonnetz(y=self.x, sr=self.sr)
        self.fourrier =  tf.abs(librosa.stft(self.x))



    def plot_mel_spec(self, axes, *args, **kwargs):
        """
        S_db_mel should be collected using method get_mel_spec
        axes should be a tuple with the axes, like (10, 5)
        """
        fig, ax = plt.subplots(figsize=axes)
        img = librosa.display.specshow(self.S_db_mel,
                                      x_axis='time',
                                      y_axis='log',
                                      ax=ax)
        ax.set_title('Mel Spectogram Example', fontsize=20)
        fig.colorbar(img, ax=ax, format=f'%0.2f')
        return plt.show()
