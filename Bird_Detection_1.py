#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Install all the Reqiuired Libraries and Packages 
import os
import glob
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc , logfbank
import librosa as lr
import os, glob, pickle
import librosa
from scipy import signal
import noisereduce as nr
from glob import glob
import librosa
get_ipython().magic('matplotlib inline')


# In[2]:


#All the Required Packages and Libraies are installed.
import soundfile
from tensorflow.keras.layers import Conv2D,MaxPool2D, Flatten, LSTM
from keras.layers import Dropout,Dense,TimeDistributed
from keras.models import Sequential
#from keras.utils import to_categorical 
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


# In[3]:


os.listdir(path=r'C:\Users\Pooja\Downloads\Webapp Project_3\data')
def getListOfFiles(dirName):
    listOfFile=os.listdir(dirName)
    allFiles=list()
    for entry in listOfFile:
        fullPath=os.path.join(dirName, entry)
        if os.path.isdir(fullPath):
            allFiles=allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
    return allFiles

dirName = r'C:\Users\Pooja\Downloads\Webapp Project_3\data'
listOfFiles = getListOfFiles(dirName)
len(listOfFiles)


# In[4]:


def envelope(y , rate, threshold):
    mask=[]
    y=pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/10) ,  min_periods=1 , center = True).mean()
    for mean in y_mean:
        if mean>threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask


# In[5]:


#Next Step is In-Depth Visualisation of Audio Fiels and its certain features to plot for.
#They are the Plotting Functions to be called later. 
def plot_signals(signals):
    fig , axes = plt.subplots(nrows=2, ncols=5,sharex =False , sharey=True, figsize=(20,5))
    fig.suptitle('Time Series' , size=16)
    i=0
    for x in range(2):
        for y in range(5):
            axes[x,y].set_title(list(signals.keys())[i])
            axes[x,y].plot(list(signals.values())[i])
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i +=1

def plot_fft(fft):
    fig , axes = plt.subplots(nrows=2, ncols=5,sharex =False , sharey=True, figsize=(20,5))
    fig.suptitle('Fourier Transform' , size=16)
    i=0
    for x in range(2):
        for y in range(5):
            data = list(fft.values())[i]
            Y,freq = data[0] , data[1]
            axes[x,y].set_title(list(fft.keys())[i])
            axes[x,y].plot(freq , Y)
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i +=1
    
def plot_fbank(fbank):
    fig , axes = plt.subplots(nrows=2, ncols=5,sharex =False , sharey=True, figsize=(20,5))
    fig.suptitle('Filter Bank Coefficients' , size=16)
    i=0
    for x in range(2):
        for y in range(5):
            axes[x,y].set_title(list(fbank.keys())[i])
            axes[x,y].imshow(list(fbank.values())[i],cmap='hot', interpolation = 'nearest')
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i +=1
            
def plot_mfccs(mfccs):
    fig , axes = plt.subplots(nrows=2, ncols=5,sharex =False , sharey=True, figsize=(20,5))
    fig.suptitle('Mel Frequency Capstrum  Coefficients' , size=16)
    i=0
    for x in range(2):
        for y in range(5):
            axes[x,y].set_title(list(mfccs.keys())[i])
            axes[x,y].imshow(list(mfccs.values())[i],
                             cmap='hot', interpolation = 'nearest')
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i +=1

def calc_fft(y,rate):
    n = len(y)
    freq = np.fft.rfftfreq(n , d= 1/rate)
    Y= abs(np.fft.rfft(y)/n)
    return(Y,freq)


# In[6]:


#Now Cleaning Step is Performed where:
#DOWN SAMPLING OF AUDIO FILES IS DONE  AND PUT MASK OVER IT AND DIRECT INTO CLEAN FOLDER
#MASK IS TO REMOVE UNNECESSARY EMPTY VOIVES AROUND THE MAIN AUDIO VOICE 
def envelope(y , rate, threshold):
    mask=[]
    y=pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/10) ,  min_periods=1 , center = True).mean()
    for mean in y_mean:
        if mean>threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask


# In[7]:


#The clean Audio Files are redirected to Clean Audio Folder Directory 
import glob,pickle
for file in tqdm(glob.glob(r'data\\**\\*.wav')):
    file_name = os.path.basename(file)
    signal , rate = librosa.load(file, sr=16000)
    mask = envelope(signal,rate, 0.0005)
    wavfile.write(filename= 'clean_speech\\'+str(file_name), rate=rate,data=signal[mask])


# In[8]:


#Feature Extraction of Audio Files Function 
#Extract features (mfcc, chroma, mel) from a sound file
def extract_feature(file_name, mfcc, chroma, mel):
   with soundfile.SoundFile(file_name) as sound_file:
       X = sound_file.read(dtype="float32")
       sample_rate=sound_file.samplerate
       if chroma:
           stft=np.abs(librosa.stft(X))
       result=np.array([])
       if mfcc:
           mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
       result=np.hstack((result, mfccs))
       if chroma:
           chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
       result=np.hstack((result, chroma))
       if mel:
           mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
       result=np.hstack((result, mel))
   return result


# In[9]:


birds = {'11713':'Dendrocopos major',
         '11846':'Chloris chloris',
         '12577':'Corvus frugilegus',
         '12578':'Coccothraustes coccothraustes',
         '12876':'Columba palumbus',
         '12996':'Delichon urbicum',
         '13164':'Apus apus',
         '13602':'Sitta europaea',
         '13608':'Corvus monedula',
         '13609':'Phoenicurus ochruros',
         '13610':'Turdus merula',
         '14172':'Turdus pilaris',
         '14212':'Passer montanus',
         '14213':'Phylloscopus trochilus',
         '14231':'Phylloscopus collybita',
         '14426':'Phoenicurus phoenicurus',
         '14442':'Motacilla alba',
         '14518':'Erithacus rubecula',
         '14844':'Streptopelia decaocto',
         '15245':'Parus major',
         '15269':'Parus caeruleus',
         '15270':'Alauda arvensis',
         '18125':'Luscinia luscinia',
         '18247':'Garrulus glandarius',
         '18344':'Turdus philomelos',
         '18387':'Pica pica',
         '18388':'Troglodytes troglodytes',
         '18483':'Carduelis carduelis',
         '18484':'Sturnus vulgaris',
         '20420':'Emberiza citrinella'}

observed_birds = ['Dendrocopos major','Chloris chloris','Corvus frugilegus','Coccothraustes coccothraustes'
,'Columba palumbus','Delichon urbicum','Apus apus','Sitta europaea','Corvus monedula','Phoenicurus ochruros',
'Turdus merula','Turdus pilaris','Passer montanus','Phylloscopus trochilus','Phylloscopus collybita','Phoenicurus phoenicurus',
'Motacilla alba','Erithacus rubecula','Streptopelia decaocto','Parus major','Parus caeruleus','Alauda arvensis',
'Luscinia luscinia','Garrulus glandarius','Turdus philomelos','Pica pica','Troglodytes troglodytes','Carduelis carduelis',
  'Sturnus vulgaris','Emberiza citrinella']


# In[10]:


#Load the data and extract features for each sound file
from glob import glob
import os
import glob
def load_data(test_size=0.33):
    x,y=[],[]
    answer = 0
    for file in glob.glob(r'C:\Users\Pooja\Downloads\Webapp Project_3\clean_speech/*.wav'):
        file_name=os.path.basename(file)
        bird=birds[file_name.split("-")[0]]
        feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append([bird,file_name])
          
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)


# In[11]:


#Split the dataset
import librosa
import numpy as np
x_train,x_test,y_trai,y_tes=load_data(test_size=0.33)
print(np.shape(x_train),np.shape(x_test), np.shape(y_trai),np.shape(y_tes))
y_test_map = np.array(y_tes).T
y_test = y_test_map[0]
test_filename = y_test_map[1]
y_train_map = np.array(y_trai).T
y_train = y_train_map[0]
train_filename = y_train_map[1]
print(np.shape(y_train),np.shape(y_test))
print(*test_filename,sep="\n")


# In[12]:


#Get the shape of the training and testing datasets
# print((x_train.shape[0], x_test.shape[0]))
print((x_train[0], x_test[0]))
#Get the number of features extracted
print(f'Features extracted: {x_train.shape[1]}')


# In[13]:


# Initialize the Multi Layer Perceptron Classifier
model=MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)


# In[14]:


#Train the model
model.fit(x_train,y_train)


# In[15]:


#SAVING THE MODEL
import pickle
# Save the Modle to file in the current working directory
#For any new testing data other than the data in dataset

Pkl_Filename = "Bird_Voice_Detection_Model.pkl"  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(model, file)


# In[16]:


# Load the Model back from file
with open(Pkl_Filename, 'rb') as file:  
    Emotion_Voice_Detection_Model = pickle.load(file)

Emotion_Voice_Detection_Model


# In[17]:


x_test


# In[18]:


#predicting :
y_pred=Emotion_Voice_Detection_Model.predict(x_test)
y_pred


# In[19]:


#Store the Prediction probabilities into CSV file 
import numpy as np
import pandas as pd
y_pred1 = pd.DataFrame(y_pred, columns=['predictions'])
y_pred1['file_names'] = test_filename
print(y_pred1)
y_pred1.to_csv('predictionfinal.csv')


# In[ ]:




