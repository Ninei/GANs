
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.io.wavfile
import librosa
import pywt
from TransSound import *

ts = TransSound()

#### Define DWT(Discrete Wavelet Transform) Constant Variables
sampleRate=22050 # 22.050kHz
sampleRate, sampleList, sampleData = ts.getSource(targetFile=ts.getSourceFile(), targeWavelet='db2', targetLevel=3);
# originalMatrix = pywt.wavedec(data=targetData, wavelet='db2', level=max_level)
cA3, cD3, cD2, cD1 = sampleList
print("< Discrete Wavelet Transform >\n" + "  cD1: {0}\n  cD2: {1}\n  cD3: {2}\n  cA3: {3}\n".format(cD1,cD2,cD3,cA3))

# sample_cD1 = (np.random.random((sampleCount*4,1))-0.5)*2 # 1 by 2000, -1.0 ~ +1.0
# sample_cD2 = (np.random.random((sampleCount*2,1))-0.5)*2
# sample_cD3 = (np.random.random((sampleCount,1))-0.5)*2
# sample_cA3 = (np.random.random((sampleCount,1))-0.5)*2

sample_cD1 = (np.random.random(cD1.size)-0.5)*2 # 1 by 2000, -1.0 ~ +1.0
sample_cD2 = (np.random.random(cD2.size)-0.5)*2
sample_cD3 = (np.random.random(cD3.size)-0.5)*2
sample_cA3 = (np.random.random(cA3.size)-0.5)*2

# 으음....
# train_tree = np.append(sample_cA3, sample_cD3, axis=1)
# print(sample_tree.size);
# print(sample_cD2.size);
# train_tree = np.append(train_tree, sample_cD2, axis=1)
# train_tree = np.append(train_tree, sample_cD1, axis=1)
# print(train_tree)

# trainList = [cA3, cD3, cD2, cD1]
trainList = [sample_cA3, sample_cD3, sample_cD2, sample_cD1]

# Trace Figure
ts.traceFigure(targetList=trainList, targetRate=sampleRate, targetFile=ts.getOutputFile(), targetWavelet='db2')