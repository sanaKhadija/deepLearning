import os
import librosa
import numpy as np
from keras.models import load_model
import IPython.display as ipd

train_audio_path = r'D:\network_programming\deepLearning\audio'
classes = os.listdir(train_audio_path)
model = load_model(r'D:\network_programming\deepLearning\model\best_model12.hdf5')

def predict(audio):
    prob = model.predict(audio.reshape(1, 8000, 1))
    index = np.argmax(prob[0])
    return classes[index]

test, test_rate = librosa.load(r'D:\network_programming\deepLearning\testAudio\test_down.wav',
                                    sr = 16000)
test_sample = librosa.resample(test, test_rate, 8000)
ipd.Audio(test_sample, rate=8000)
test_sample.shape
print("##############")
print("[#]Text from test_down.wav : ",predict(test_sample))
print("--------------")
########################
test, test_rate = librosa.load(r'D:\network_programming\deepLearning\testAudio\test_go.wav',
                                    sr = 16000)
test_sample = librosa.resample(test, test_rate, 8000)
ipd.Audio(test_sample, rate=8000)
test_sample.shape
print("##############")
print("[#]Text from test_go.wav : ",predict(test_sample))
print("--------------")
#######################
test, test_rate = librosa.load(r'D:\network_programming\deepLearning\testAudio\test_up.wav',
                                    sr = 16000)
test_sample = librosa.resample(test, test_rate, 8000)
ipd.Audio(test_sample, rate=8000)
test_sample.shape
print("##############")
print("[#]Text form test_up.wav : ",predict(test_sample))
print("--------------")
