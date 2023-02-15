# Silvertone

- Project Name: silvertone
- Description: A ML model that recognizes tone(emotion) in spoken english audio recordings
- Data Source: Various university databases, annotated by humans, with particular structures, links and origins documented in "Silver-Tone Reference v0.ipynb"
- Main objective: Create a model capable of recognizing emotions in audio and deploying the model on an online platform

## Background
This was the final project for a 9-week intensive Data Science and Machine Learning bootcamp. I pitched the idea, but the team was composed of myself and 3 other people, who's contributions can be seen on each of their working branches. The streamlit deployment was executed by lalianza and guipyc.

vsattamini: https://github.com/vsattamini

LucasGama1207: https://github.com/LucasGama1207

lalianza: https://github.com/lalianza

guipyc: https://github.com/guipyc


## Data analysis

Data analysis was less important in this project, as we had very well labeled data, and our core objective was to develop a model capable of recognizing emotion

## Data preparation

Librosa was the main library used for this project. It is an audio analyisis library with various built in tools. We tried out a few different methods and metrics contained in Librosa, but we had better success with cutting or padding the audios so that all instances were of equal length andcould be inputed in our models. We cut empty audio based on decebels, keeping only the spoken portions.

### Extracted features

- Wave
- Mel-frequency cepstral coefficients (MFCCs)
- Mel Spectrogram
- Spectral Centroid
- Chromagram
- Tonnetz

## Models

We tried out various different models, with varying levels of success:

- 1D and 2D convolutional neural networks (CNNs), applied to various metrics but focusing on the mel spectrogram and on tonnetz
-- 1D CNNs as the usual audio analysis tool (This analysis on MFCCs came close to replicating the results on ML Models)
-- 2D CNNs as a way to analyze spectrograms as images (On tonnetz, we had good results, but not quite as close to traditional ML models as we'd like)

- Traditional Machine Learning Models
-- Random Forest (intitially our most successful model) on MFCCs
-- SVM on MFCCs (Our final, most successful model)


## Deployment

We deployed out model on streamlit and included transcription and audio vizualization tool on the interface. It can be accessed on: https://lalianza-silvertone-app-app-bv573b.streamlit.app/

