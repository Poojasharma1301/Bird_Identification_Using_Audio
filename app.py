from typing import MutableMapping
from flask import Flask, render_template, request
import requests
import pickle
import pandas as pd
import numpy as np
import sklearn
import librosa
import soundfile
import os
from sklearn.model_selection import train_test_split

app = Flask(__name__)
model = pickle.load(open(r'E:\DATA SCIENCE\TechQ-Konnect_Internship\Project_3(Bird Identification Using Audio)\model\Bird_Voice_Detection_Model.pkl', 'rb'))
@app.route('/', methods=['GET'])
def Home():
    return render_template('index.html')

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
            mel=np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis=0)
        result=np.hstack((result, mel))
    return result

def load_data(file_name_1):
    x,y=[],[]
    feature=extract_feature(file_name_1, mfcc=True, chroma=True, mel=True)
    file_name = os.path.basename(file_name_1)
    
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
    bird=file_name.split("-")[0]
    x.append(feature)
    y.append([bird,file_name])
    x.append(feature)
    y.append([bird,"test"])
    return train_test_split(np.array(x), y, test_size=1, random_state=9)


@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        print(request.files['file'])
        
        file = request.files['file'].filename
        request.files['file'].save(f"{file}")
        
        x_train,x_test,y_trai,y_tes=load_data(file)
        prediction = model.predict(x_test)
    
        label_map =   ['Dendrocopos major','Chloris chloris','Corvus frugilegus','Coccothraustes coccothraustes'
,'Columba palumbus','Delichon urbicum','Apus apus','Sitta europaea','Corvus monedula','Phoenicurus ochruros',
'Turdus merula','Turdus pilaris','Passer montanus','Phylloscopus trochilus','Phylloscopus collybita','Phoenicurus phoenicurus',
'Motacilla alba','Erithacus rubecula','Streptopelia decaocto','Parus major','Parus caeruleus','Alauda arvensis',
'Luscinia luscinia','Garrulus glandarius','Turdus philomelos','Pica pica','Troglodytes troglodytes','Carduelis carduelis',
  'Sturnus vulgaris','Emberiza citrinella']
        
        

    birds_dict = {
        'Dendrocopos major':"https://i.ibb.co/9cZXyFZ/Dendrocopos-major.jpg",
        'Chloris chloris':"https://i.ibb.co/FBnjgMN/Chloris-chloris.jpg",
        'Corvus frugilegus':"https://i.ibb.co/gPD929b/Corvus-frugilegus.jpg",
        'Coccothraustes coccothraustes':"https://i.ibb.co/qsDsddv/Coccothraustes-coccothraustes.jpg",
        'Columba palumbus':"https://i.ibb.co/v1BnjY7/Columba-palumbus.jpg",
        'Delichon urbicum':"https://i.ibb.co/mczD4Lk/Delichon-urbicum.jpg",
        'Apus apus':"https://i.ibb.co/nCvWwrj/Apus-apus.jpg",
        'Sitta europaea':"https://i.ibb.co/Jr5LhcC/Sitta-europaea.jpg",
        'Corvus monedula':"https://i.ibb.co/MgzKJ2k/Corvus-monedula.jpg",
        'Phoenicurus ochruros':"https://i.ibb.co/TPc2bnV/Phoenicurus-ochruros.jpg",
        'Turdus merula':"https://i.ibb.co/1qV5x18/Turdus-merula.jpg",
        'Turdus pilaris':"https://i.ibb.co/d51GYgg/Turdus-pilaris.jpg",
        'Passer montanus':"https://i.ibb.co/v1Y24SK/Passer-montanus.jpg",
        'Phylloscopus trochilus':"https://i.ibb.co/pjD7m6B/Phylloscopus-trochilus.jpg",
        'Phylloscopus collybita':"https://i.ibb.co/tJHxzDz/Phylloscopus-collybita.jpg",
        'Phoenicurus phoenicurus':"https://i.ibb.co/f0PwfT2/Phoenicurus-phoenicurus.jpg",
        'Motacilla alba':"https://i.ibb.co/Kskn3LR/Motacilla-alba.jpg",
        'Erithacus rubecula':"https://i.ibb.co/5TnjmT2/Erithacus-rubecula.jpg",
        'Streptopelia decaocto':"https://i.ibb.co/kqR0Btt/Streptopelia-decaocto.jpg",
        'Parus major':"https://i.ibb.co/5GvNxb9/Parus-major.jpg",
        'Parus caeruleus':"https://i.ibb.co/vBJzcFV/Parus-caeruleus.jpg",
        'Alauda arvensis':"https://i.ibb.co/2FpSXdy/Alauda-arvensis.jpg",
        'Luscinia luscinia':"https://i.ibb.co/9872Sz9/Luscinia-luscinia.jpg",
        'Garrulus glandarius':"https://i.ibb.co/2MFtQ91/Garrulus-glandarius.jpg",
        'Turdus philomelos':"https://i.ibb.co/274wg6p/Turdus-philomelos.jpg",
        'Pica pica':"https://i.ibb.co/mFgGTz1/Pica-pica.jpg",
        'Troglodytes troglodytes':"https://i.ibb.co/Zx2LBSb/Troglodytes-troglodytes.jpg",
        'Carduelis carduelis':"https://i.ibb.co/8Yv3btX/Carduelis-carduelis.jpg",
        'Sturnus vulgaris':"https://i.ibb.co/HpHhLr0/Sturnus-vulgaris.jpg",
        'Emberiza citrinella':"https://i.ibb.co/zHqzcLf/Emberiza-citrinella.jpg"
    }

        
    imglink = birds_dict[prediction[0]]

        #final_prediction = label_map[prediction]
    print(imglink)
    
    print("Done")
    print(prediction,"<<<<<<")
    #return final_prediction
    return render_template('index.html',prediction_text=f'Bird={prediction}', imglink=imglink)
        
if __name__=="__main__":
    app.run(debug=True)
