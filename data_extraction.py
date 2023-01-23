from audio_formating import Silvertone
from glob import glob
# from google.colab import drive
import numpy as np
import pandas as pd
import os

from glob import glob
# from google.colab import drive
import numpy as np

# os.chdir('/content/drive/MyDrive/Colab Notebooks')
path = '/content/drive/MyDrive/Emotion Datasets/Emotion Datasets/Audio'


def extract_ravdess(mels=256, trim=False, path=path):
    '''
    Extracts audio features from ravdess db using methods defined in the
    Silvertone object
    '''
    wave = []
    spec = []
    id = []
    spectral_centroid = []
    mfcc = []
    chroma_stft = []
    tonnetz = []
    source = []
    emotion = []
    intensity = []
    repetition = []
    actor_gender = []
    actor_age = []
    mel_spec = []
    fourrier = []

    speech = glob(f"{path}/RAVDESS/Audio_Speech_Actors_01-24/*.wav")

    for i in np.arange(0, len(speech)):
        tone = Silvertone(speech[i], mels, trim)
        spec.append(tone.S_db_mel)
        wave.append(tone.x)
        spectral_centroid.append(tone.spectral_centroid)
        mfcc.append(tone.mfcc)
        chroma_stft.append(tone.chroma_stft)
        tonnetz.append(tone.tonnetz)
        id.append(speech[i][-24:-4])
        source.append('ravdess')
        emotion.append(speech[i][-17:-16])
        intensity.append(speech[i][-14:-13])
        repetition.append(speech[i][-8:-7])
        actor_age.append(0)
        mel_spec.append(tone.S)
        fourrier.append(tone.fourrier)
        if int(speech[i][-8:-7]) % 2 == 0 or int(speech[i][-8:-7]) == 0:
            actor_gender.append('female')
        else:
            actor_gender.append('male')

    dic = {"id": id,"wave": wave, "mel_spec":mel_spec, "fourrier":fourrier,
       "spec": spec, "spectral_centroid": spectral_centroid,
       "mfcc": mfcc, "chroma_stft": chroma_stft, "tonnetz": tonnetz,
       "emotion":emotion, "intensity": intensity, "repetition": repetition,
       "actor_gender": actor_gender,"actor_age": actor_age, "source": source}
    df = pd.DataFrame(dic)

    emo_relation = {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad',
            5: 'angry', 6: 'fearful', 7: 'disgust', 8: 'surprised',
            '1': 'neutral', '2': 'calm', '3': 'happy', '4': 'sad',
            '5': 'angry', '6': 'fearful', '7': 'disgust', '8': 'surprised'}
    int_relation = {2:3 ,1:2}


    df = df.replace({'emotion':emo_relation,'intensity':int_relation,})

    return df

def extract_crema(mels=256, trim=False, path=path):
    '''
    Extracts audio features from crema db using methods defined in the
    Silvertone object
    '''
    wave = []
    spec = []
    id = []
    spectral_centroid = []
    mfcc = []
    chroma_stft = []
    tonnetz = []
    source = []
    emotion = []
    intensity = []
    actor_gender = []
    actor_age = []
    mel_spec = []
    fourrier = []
    demo = pd.read_csv('demographic_relation.csv')
    actids = list(demo.to_dict()['ActorID'].values())
    actids = [str(i) for i in actids]
    gender = list(demo.to_dict()['Sex'].values())
    gender = [i.lower() for i in gender]
    age = list(demo.to_dict()['Age'].values())
    age = [i.lower() for i in age]
    gender_relation = dict(zip(actids, gender))
    age_relation = dict(zip(actids, age))

    speech = glob(f"{path}/Crema (avail in TFDS)/*.wav")
    repetition = []

    for i in speech:
        tone = Silvertone(i, mels, trim)
        spec.append(tone.S_db_mel)
        wave.append(tone.x)
        spectral_centroid.append(tone.spectral_centroid)
        mfcc.append(tone.mfcc)
        chroma_stft.append(tone.chroma_stft)
        tonnetz.append(tone.tonnetz)
        id.append(i[-19:-4])
        source.append('crema')
        emotion.append(i[-10:-7])
        intensity.append(i[-6:-4])
        repetition.append(0)
        actor_gender.append(i[-19:-15])
        actor_age.append(i[-19:-15])
        mel_spec.append(tone.S)
        fourrier.append(tone.fourrier)



    dic = {"id": id,"wave": wave, "mel_spec":mel_spec, "fourrier":fourrier,
       "spec": spec, "spectral_centroid": spectral_centroid,
       "mfcc": mfcc, "chroma_stft": chroma_stft, "tonnetz": tonnetz,
       "emotion":emotion, "intensity": intensity, "repetition": repetition,
       "actor_gender": actor_gender,"actor_age": actor_age, "source": source}
    df = pd.DataFrame(dic)

    emo_relation = {'NEU': 'neutral', 2: 'calm', 'HAP': 'happy', 'SAD': 'sad',
            'ANG': 'angry', 'FEA': 'fearful', 'DIS': 'disgust', 8: 'surprised',
            '_SA': 'sad'}
    int_relation = {'XX': 0, 'LO':1,'MD':2,'HI':3}

    df = df.replace({'emotion':emo_relation,'intensity':int_relation,
                     'actor_gender':gender_relation, 'actor_age':age_relation})

    return df

def extract_savee(mels=256, trim=False, path=path):
    '''
    Extracts audio features from savee db using methods defined in the
    Silvertone object
    '''
    wave = []
    spec = []
    id = []
    spectral_centroid = []
    mfcc = []
    chroma_stft = []
    tonnetz = []
    source = []
    emotion = []
    intensity = []
    actor_gender = []
    actor_age = []
    repetition = []
    mel_spec = []
    fourrier = []

    speech = glob(f"{path}/Savee/*.wav")

    for i in np.arange(0, len(speech)):
        tone = Silvertone(speech[i], mels, trim)
        spec.append(tone.S_db_mel)
        wave.append(tone.x)
        spectral_centroid.append(tone.spectral_centroid)
        mfcc.append(tone.mfcc)
        chroma_stft.append(tone.chroma_stft)
        tonnetz.append(tone.tonnetz)
        id.append(speech[i][-10:-4])
        source.append('savee')
        emotion.append(speech[i][-8:-6])
        intensity.append(0)
        repetition.append(0)
        actor_gender.append('male')
        actor_age.append('young')
        mel_spec.append(tone.S)
        fourrier.append(tone.fourrier)


    dic = {"id": id,"wave": wave, "mel_spec":mel_spec, "fourrier":fourrier,
       "spec": spec, "spectral_centroid": spectral_centroid,
       "mfcc": mfcc, "chroma_stft": chroma_stft, "tonnetz": tonnetz,
       "emotion":emotion, "intensity": intensity, "repetition": repetition,
       "actor_gender": actor_gender,"actor_age": actor_age, "source": source}
    df = pd.DataFrame(dic)

    emo_relation = {'_n': 'neutral', 2: 'calm', '_h': 'happy', 'sa': 'sad',
            '_a': 'angry', '_f': 'fearful', '_d': 'disgust', 'su': 'surprised'}

    df = df.replace({'emotion':emo_relation})

    return df

def extract_tess(mels=256, trim=False, path=path):
    '''
    Extracts audio features from savee db using methods defined in the
    Silvertone object
    '''
    wave = []
    spec = []
    id = []
    spectral_centroid = []
    mfcc = []
    chroma_stft = []
    tonnetz = []
    source = []
    emotion = []
    intensity = []
    actor_gender = []
    actor_age = []
    repetition = []
    mel_spec = []
    fourrier = []

    speech = glob(f"{path}/Tess/*.wav")

    for i in np.arange(0, len(speech)):
        tone = Silvertone(speech[i], mels, trim)
        spec.append(tone.S_db_mel)
        wave.append(tone.x)
        spectral_centroid.append(tone.spectral_centroid)
        mfcc.append(tone.mfcc)
        chroma_stft.append(tone.chroma_stft)
        tonnetz.append(tone.tonnetz)
        id.append(speech[i][68:-4])
        source.append('tess')
        intensity.append(0)
        repetition.append(0)
        actor_gender.append('female')
        mel_spec.append(tone.S)
        fourrier.append(tone.fourrier)
        if speech[i][68:71] == 'YAF':
            actor_age.append('young')
        else:
            actor_age.append('old')
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

    dic = {"id": id,"wave": wave, "mel_spec":mel_spec, "fourrier":fourrier,
       "spec": spec, "spectral_centroid": spectral_centroid,
       "mfcc": mfcc, "chroma_stft": chroma_stft, "tonnetz": tonnetz,
       "emotion":emotion, "intensity": intensity, "repetition": repetition,
       "actor_gender": actor_gender,"actor_age": actor_age, "source": source}
    df = pd.DataFrame(dic)

    return df


def extract_full_db():
    ravdess = pd.DataFrame(extract_ravdess())
    crema = pd.DataFrame(extract_crema())
    savee = pd.DataFrame(extract_savee())
    tess = pd.DataFrame(extract_tess())

    final_df = pd.concat((ravdess, crema, savee,tess))

    return final_df
