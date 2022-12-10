from keras.models import load_model
from time import sleep
from tensorflow.keras.utils import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier(r'D:\dev\projects\IBM skills build emotion recognition multi model\face emotion recognition\Emotion_Detection_CNN-main\Emotion_Detection_CNN-main\haarcascade_frontalface_default.xml')
classifier =load_model(r"D:\dev\projects\IBM skills build emotion recognition multi model\face emotion recognition\Emotion_Detection_CNN-main\Emotion_Detection_CNN-main\model.h5")

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

cap = cv2.VideoCapture(r"D:\dev\projects\IBM skills build emotion recognition multi model\face emotion recognition\Emotion_Detection_CNN-main\Emotion_Detection_CNN-main\Make anyone angry in 10.38 seconds.mp4")



import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import librosa
import librosa.display
from IPython.display import Audio
import warnings
warnings.filterwarnings('ignore')








#Feature Extraction
def extract_mfcc(filename):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc


def final(df,model):
    


    X_mfcc = df['speech'].apply(lambda x: extract_mfcc(x))
    X = [x for x in X_mfcc]
    X = np.array(X)
    X.shape
    X = np.expand_dims(X, -1)
    prediction = model.predict(X)
    prediction = np.argmax(prediction)
    
    if(prediction == 0):
            prediction = 'angry'
    if(prediction == 1):
            prediction = 'disgust'
    if(prediction == 2):
            prediction = 'fear'
    if(prediction == 3):
            prediction = 'happy'
    if(prediction == 4):
            prediction = 'neutral'
    if(prediction == 5):
            prediction = 'pleasant_surprise'
    if(prediction == 6):
            prediction = 'sad'
            
    return prediction
    
    



total_list_instance = []
while True:
    _, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)



        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

            prediction = classifier.predict(roi)[0]
            label=emotion_labels[prediction.argmax()]
            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            total_list_instance.append(label)
            
            
            
            
        else:
            cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.imshow('Emotion Detector',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


chunks = [total_list_instance[x:x+60] for x in range(0, len(total_list_instance), 60)]



keymax_list = []
for list_instance in chunks:
    instance_dict = {"Neutral" : 0,
                     "Surprise" : 0,
                     "Sad" : 0,
                     "Happy" : 0,
                     "Disgust" : 0,
                     "Fear" : 0,
                     "Angry" : 0}
    for i in range(len(list_instance)):
        instance_dict[list_instance[i]]+=1

    Keymax = max(zip(instance_dict.values(), instance_dict.keys()))[1]
    keymax_list.append(Keymax)
    
    
    
    
    
    
    
 
    
    
    
    
    
    
    
    
#extracting audio from video  
import moviepy.editor as mp
#from moviepy.editor import *
my_clip = mp.VideoFileClip(r"D:\dev\projects\IBM skills build emotion recognition multi model\face emotion recognition\Emotion_Detection_CNN-main\Emotion_Detection_CNN-main\Make anyone angry in 10.38 seconds.mp4")
my_clip.audio.write_audiofile(r"D:\dev\projects\IBM skills build emotion recognition multi model\audio\my_result.mp3")
#audioclip = AudioFileClip("geeks.mp4")



#splitting audio into 1sec clips
from pydub import AudioSegment

audio = AudioSegment.from_file(r"D:\dev\projects\IBM skills build emotion recognition multi model\audio\my_result.mp3")
lengthaudio = len(audio)
print("Length of Audio File", lengthaudio)

start = 0
# # In Milliseconds, this will cut 10 Sec of audio
threshold = 1000
end = 0
counter = 0

while start < len(audio):

    end += threshold

    print(start , end)

    chunk = audio[start:end]

    filename = r'D:\dev\projects\IBM skills build emotion recognition multi model\audio\1sec audio clips\audio clips\chunk' + str(counter) + '.wav'

    chunk.export(filename, format="wav")

    counter +=1

    start += threshold






#Load the Dataset
paths = []
labels = []
for dirname, _, filenames in os.walk(r"D:\dev\projects\IBM skills build emotion recognition multi model\audio\1sec audio clips"):
    for filename in filenames:
        paths.append(os.path.join(dirname, filename))
        label = filename
        labels.append(label.lower())
  
print('Dataset is Loaded')


## Create a dataframe
df = pd.DataFrame()
df['speech'] = paths
df['label'] = labels
    
 
    
n = len(paths)

sound_prediction_list = []
for i in range(n):
    df1 = df.loc[[i],['speech']]
    model = load_model(r"D:\dev\projects\IBM skills build emotion recognition multi model\audio\model.h5")
    prediction_sound = final(df1,model)
    sound_prediction_list.append(prediction_sound)
    

for i in range(len(sound_prediction_list)):
    print("\n")
    print(i)
    k = "audio : " + str(sound_prediction_list[i]) 
    print(k)
    

    k1 = "video : " + str(keymax_list[i]) 
    print(k1)       
        



