from audio_formating import Silvertone
from glob import glob
# from google.colab import drive
import numpy as np
import pandas as pd
import os

from glob import glob
# from google.colab import drive
import numpy as np
import librosa
import tensorflow as tf

# os.chdir('/content/drive/MyDrive/Colab Notebooks')
path = '/content/drive/MyDrive/Emotion Datasets/Emotion Datasets/Audio'


def preprocessing (audio):
    mfcc = []
    X, sr = librosa.load(audio)
    X_trim = librosa.effects.trim(X,top_db=35)
    if X_trim[0].shape[0] >= 50000:
        X_final = X_trim[0][:50000]
        X_final = tf.convert_to_tensor(X_final).numpy()
    else:
        zero_padding = tf.zeros([50000]-tf.shape(X_trim[0]),dtype=tf.float32)
        X_final = tf.concat([X_trim[0],zero_padding],0).numpy()
    S = librosa.feature.mfcc(y=X_final, sr=sr)
    mfcc.append(S)
    X= np.array(mfcc)
    X_flat=X.reshape(20*98)
    return X_flat

def extract_ravdess(mels=256, trim=False, path=path):
    '''
    Extracts audio features from ravdess db using methods defined in the
    Silvertone object
    '''
    emotion =[]
    result = []
    speech = glob(f"{path}/RAVDESS/Audio_Speech_Actors_01-24/*.wav")

    for i in np.arange(0, len(speech)):
        emotion.append(speech[i][-17:-16])
        result.append(preprocessing(speech[i]))



    result = np.array(result)
    df = pd.DataFrame(result)
    y = pd.DataFrame(emotion)

    emo_relation = {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad',
            5: 'angry', 6: 'fearful', 7: 'disgust', 8: 'surprised',
            '1': 'neutral', '2': 'calm', '3': 'happy', '4': 'sad',
            '5': 'angry', '6': 'fearful', '7': 'disgust', '8': 'surprised'}


    y = y.replace({'emotion':emo_relation})

    return df, y

def extract_crema(mels=256, trim=False, path=path):
    '''
    Extracts audio features from crema db using methods defined in the
    Silvertone object
    '''

    emotion =[]
    result = []
    speech = glob(f"{path}/Crema (avail in TFDS)/*.wav")

    for i in np.arange(0, len(speech)):
        temp = preprocessing(speech[i])
        emotion.append(speech[i][-10:-7])
        result.append(temp)

    result = np.array(result)
    df = pd.DataFrame(result)
    y = pd.DataFrame(emotion)

    emo_relation = {'NEU': 'neutral', 2: 'calm', 'HAP': 'happy', 'SAD': 'sad',
            'ANG': 'angry', 'FEA': 'fearful', 'DIS': 'disgust', 8: 'surprised',
            '_SA': 'sad'}


    y = y.replace({'emotion':emo_relation})


    return df, y

def extract_savee(mels=256, trim=False, path=path):
    '''
    Extracts audio features from savee db using methods defined in the
    Silvertone object
    '''
    emotion =[]
    result = []
    speech = glob(f"{path}/Savee/*.wav")

    for i in np.arange(0, len(speech)):
        temp = preprocessing(speech[i])
        emotion.append(speech[i][-8:-6])
        result.append(temp)

    result = np.array(result)
    df = pd.DataFrame(result)
    y = pd.DataFrame(emotion)

    emo_relation = {'_n': 'neutral', 2: 'calm', '_h': 'happy', 'sa': 'sad',
            '_a': 'angry', '_f': 'fearful', '_d': 'disgust', 'su': 'surprised'}


    y = y.replace({'emotion':emo_relation})


    return df, y


def extract_tess(mels=256, trim=False, path=path):
    '''
    Extracts audio features from savee db using methods defined in the
    Silvertone object
    '''

    emotion =[]
    result = []
    speech = glob(f"{path}/Tess/*.wav")

    for i in np.arange(0, len(speech)):
        temp = preprocessing(speech[i])
        result.append(temp)
        if speech[i][-7:-4] == 'ral':
            emotion.append('neutral')
        elif speech[i][-7:-4] == 'ust':
            emotion.append('disgust')
        elif speech[i][-7:-4] == 'ppy':
            emotion.append('happy')
        elif speech[i][-7:-4] == '_ps':
            emotion.append('surprised')
        elif speech[i][-7:-4] == 'sad':
            emotion.append('sad')
        elif speech[i][-7:-4] == 'ear':
            emotion.append('fearful')
        else:
            emotion.append('angry')

    result = np.array(result)
    df = pd.DataFrame(result)
    y = pd.DataFrame(emotion)

    return df, y


def extract_ESD(mels=256, trim=False, path=path, eng_only=True):
    '''
    Extracts audio features from ravdess db using methods defined in the
    Silvertone object
    '''

    emotions = ["Angry", "Happy", "Neutral", "Sad","Surprise"]
    emotion =[]
    result = []

    for e in emotions:
        eng_path = f"{path}/Emotional Speech Dataset (ESD)/eng/{e}/*.wav"
        speech = glob(eng_path)
        for i in np.arange(0, len(speech)):
            temp = preprocessing(speech[i])
            emotion.append(e)
            result.append(temp)


    result = np.array(result)
    df = pd.DataFrame(result)
    y = pd.DataFrame(emotion)

    emo_relation = {'Neutral': 'neutral', 2: 'calm', 'Happy': 'happy', 'Sad': 'sad',
            'Angry': 'angry', 6: 'fearful', 7: 'disgust', 'Surprise': 'surprised',
            '1': 'neutral', '2': 'calm', '3': 'happy', '4': 'sad',
            '5': 'angry', '6': 'fearful', '7': 'disgust', '8': 'surprised'}


    y = y.replace({'emotion':emo_relation})


    return df, y


def extract_JL(mels=256, trim=False, path=path):
    '''
    Extracts audio features from ravdess db using methods defined in the
    Silvertone object
    '''

    emotion =[]
    result = []
    speech = glob(f"{path}/JL Corpus/Raw JL corpus (unchecked and unannotated)/JL(wav+txt)/*.wav")
    si = len(f"{path}/JL Corpus/Raw JL corpus (unchecked and unannotated)/JL(wav+txt)/")

    for i in np.arange(0, len(speech)):
        temp = preprocessing(speech[i])
        emotion.append(speech[i][si+5:si+10])
        result.append(temp)

    result = np.array(result)
    df = pd.DataFrame(result)
    y = pd.DataFrame(emotion)

    emo_relation ={'_angr': 'angry','e1_an': 'angry','e2_an': 'angry',
                   '_apol': 'apologetic','e1_ap': 'apologetic','e2_ap': 'apologetic',
                   '_conc': 'concerned','e1_co': 'concerned','e2_co': 'concerned',
                   '_enco': 'encouraging','e1_en': 'encouraging','e2_en': 'encouraging',
                   '_asse': 'assertive','e1_as': 'assertive','e2_as': 'assertive',
                   '_exci': 'excited','e1_ex': 'excited','e2_ex': 'excited',
                   '_happ': 'happy','e1_ha': 'happy','e2_ha': 'happy',
                   '_neut': 'neutral','e1_ne': 'neutral','e2_ne': 'neutral',
                   '_ques': 'question','e1_qu': 'question','e2_qu': 'question',
                   '_sad_': 'sad','e1_sa': 'sad','e2_sa': 'sad', '_anxi':'anxious'
                   }


    y = y.replace({'emotion':emo_relation})

    return df, y

def extract_ASVP(mels=256, trim=False, path=path):
    '''
    Extracts audio features from ravdess db using methods defined in the
    Silvertone object
    '''

    emotion =[]
    result = []
    speech = glob(f"{path}/ASVP-ESD(Speech & Non-Speech Emotional Sound)/ASVP-ESD-Update/Audio/*.wav")
    si = len(f"{path}/ASVP-ESD(Speech & Non-Speech Emotional Sound)/ASVP-ESD-Update/Audio/")

    for i in np.arange(0, len(speech)):
        temp = preprocessing(speech[i])
        emotion.append(speech[i][si+6:si+8])
        result.append(temp)

    result = np.array(result)
    df = pd.DataFrame(result)
    y = pd.DataFrame(emotion)

    emo_relation ={'01': 'bored', '02': 'neutral','03':'happy', '04': 'sad',
                   '05': 'angry','06': 'fearful','07': 'disgust',
                   '08':'surprised','09': 'excited','10': 'pleasure',
                   '11': 'pain', '12': 'disappointed'
                   }


    y = y.replace({'emotion':emo_relation})

    return df, y
