import streamlit as st
import librosa
from audio_recorder_streamlit import audio_recorder
import pickle
import io
import numpy as np
import tensorflow as tf
import base64
import webbrowser
import speech_recognition as sr
from datetime import datetime
import os
import pandas as pd
import plotly.express as px
from plotly.graph_objs import *
import matplotlib.pyplot as plt
import librosa.display

now = datetime.now() # current date and time
date_time = now.strftime("%Y_%m_%d_%H_%M_%S")




st.title("Silvertone")
st.markdown("""We created this app to be able to recognize emotion in spoken english.
We hope you enjoy it
To use it:
1. Use the button below to record an audio acting out an emotion (try to keep it short, the app likes it better that way!)
2. Let the app do its magic
3. See if it matches! Clicking the Percentages and Graphs below should show you the classification percentages
4. The spectrogram tab will give you a visual representation of your audio""")
st.subheader("Record an audio, and receive a % .... :")
st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)

#st.sidebar.title("Welcome to Silvertone!")
st.sidebar.image("final_logo.png", use_column_width=True)
st.sidebar.write("[Repository](https://github.com/vsattamini/silvertone)")

st.sidebar.write("Contributors:")
st.sidebar.write("[Luiz Lianza](https://github.com/lalianza)")
st.sidebar.write("[Victor Sattamini](https://github.com/vsattamini)")
st.sidebar.write("[Lucas Gama](https://github.com/lucasgama1207)")
st.sidebar.write("[Guilherme Barreto](https://github.com/guipyc)")

#def add_bg_from_local(image_file):
#    with open(image_file, "rb") as image_file:
#        encoded_string = base64.b64encode(image_file.read())
#    st.markdown(
#    f"""
#    <style>
#    .stApp {{
#        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
#        background-size: cover
#    }}
#    </style>
#    """,
#    unsafe_allow_html=True
#    )
#add_bg_from_local('logo2.png')
filename = None

def preprocessing (audio):
    spectrograms = []
    X, sr = librosa.load(audio)
    X_trim = librosa.effects.trim(X,top_db=35)
    if X_trim[0].shape[0] >= 65000:
        X_final = X_trim[0][:65000]
        X_final = tf.convert_to_tensor(X_final).numpy()
    else:
        zero_padding = tf.zeros([65000]-tf.shape(X_trim[0]),dtype=tf.float32)
        X_final = tf.concat([X_trim[0],zero_padding],0).numpy()
    S = librosa.feature.melspectrogram(y=X_final, sr=sr,n_mels=128)
    spectrograms.append(S)
    X= np.array(spectrograms)
    X_flat=X.reshape(X.shape[0],128*127)
    return X_flat



audio_bytes = audio_recorder()
if audio_bytes:
    st.audio(audio_bytes, format='audio/wav')
    print(type(audio_bytes))

   #filename = (f'{date_time}audio.wav')


   #with open(filename, "bx") as file:
   #    file.write(audio_bytes)

    X_flat = preprocessing(io.BytesIO(audio_bytes))
    model_pickle = open("54,6_model.sav", "rb")
    model = pickle.load(model_pickle)
    y = model.predict(X_flat)
    y_proba = model.predict_proba(X_flat)
    if y == "01":
        response = "Neutral"
    elif y == "02":
        response = "Calm"
    elif y == "03":
        response = "Happy"
    elif y == "04":
        response = "Sad"
    elif y == "05":
        response = "Angry"
    elif y == "06":
        response = "Fearful"
    elif y == "07":
        response = "Disgust"
    elif y == "08":
        response = "Surprised"
    else:
        response = "Error"
    st.subheader(response)


import soundfile as sf

r = sr.Recognizer()

audio_source = sr.AudioData(audio_bytes,44100,4)
text = r.recognize_google(audio_data=audio_source, language = 'en', show_all = True )
st.subheader(text['alternative'][0]["transcript"])

tab1, tab2, tab3 = st.tabs([" ","Percentages and Graph","Spectrogram"])

with tab2:
   y_proba_graph = (pd.DataFrame(np.array(y_proba).reshape(-1,1)))
   y_proba_graph['Emotions'] = ["Neutral", "Calm", "Happy","Sad","Angry","Fearful","Disgust","Surprised"]
   y_proba_graph["Probability"] = (y_proba_graph[0]*100).astype(int)
   y_proba_graph = y_proba_graph.drop(0,axis=1)

   fig = px.bar(y_proba_graph, x="Emotions", y="Probability", text="Probability")
   fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
   fig.update_xaxes(showgrid=False)
   fig.update_yaxes(showgrid=False)
#   fig.update_xaxes(showticklabels=False)
   fig.update_yaxes(showticklabels=False)
   # Plot!
   st.plotly_chart(fig, use_container_width=True)

with tab3:
   fig, ax = plt.subplots(facecolor=(0, 0, 0,0))
   X, sr = librosa.load((io.BytesIO(audio_bytes)))
   #S_dB = librosa.power_to_db(y=X,sr=sr,n_mels=128, ref=np.max)
   S = librosa.feature.melspectrogram(y=X, sr=sr, n_mels=128)
   S_db_mel = librosa.amplitude_to_db(S, ref=np.max)
   #mel_spec = librosa.features.melspectogram
   img = librosa.display.specshow(S_db_mel, x_axis='time',
                            y_axis='mel', sr=sr,
                            fmax=8000, ax=ax)
   fig.colorbar(img, ax=ax, format='%+2.0f dB')
   #ax.set_facecolor(0,0,0,0)
   ax.set(title='Mel-frequency spectrogram')
   st.pyplot(fig)
#if audio_bytes:
#    st.audio(audio_bytes, format='audio/wav')
#    print(type(audio_bytes))
#    X_flat = preprocessing(io.BytesIO(audio_bytes))
#    model_pickle = open("54,6_model.sav", "rb")
#    model = pickle.load(model_pickle)
#    y = model.predict(X_flat)
#    y_proba = model.predict_proba(X_flat)
#    if y == "01":
#        response = "Neutral"
#    elif y == "02":
#        response = "Calm"
#    elif y == "03":
#        response = "Happy"
#    elif y == "04":
#        response = "Sad"
#    elif y == "05":
#        response = "Angry"
#    elif y == "06":
#        response = "Fearful"
#    elif y == "07":
#        response = "Disgust"
#    elif y == "08":
#        response = "Surprised"
#    else:
#        response = "Error"
#    st.subheader(response)
    #st.subheader(np.round(y_proba[0][0]*100))
    #st.subheader(np.round(y_proba[0][1]*100))
    #col1, col2, col3,col4,col5,col6,col7,col8 = st.columns(8)
    #col1.metric(f"Neutral", str(int(y_proba[0][0]*100)))
    #col2.metric("Calm", str(int(y_proba[0][1]*100)))
    #col3.metric("Happy", str(int(y_proba[0][2]*100)))
    #col4.metric("Sad", str(int(y_proba[0][3]*100)))
    #col5.metric("Angry", str(int(y_proba[0][4]*100)))
    #col6.metric("Fearful", str(int(y_proba[0][5]*100)))
    #col7.metric("Disgust", str(int(y_proba[0][6]*100)))
    #col8.metric("Surprised", str(int(y_proba[0][7]*100)))


    # data, samplerate = librosa.load(io.BytesIO(audio_bytes))
    # st.text(len(data))
    # st.text(type(data))
