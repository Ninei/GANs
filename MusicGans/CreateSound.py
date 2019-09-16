
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.io.wavfile
import librosa
import pywt

#### Check Dataset & Output Directory
def checkPath(target) :
    if not os.path.exists(target): os.makedirs(target)

ROOT_INPUT_PATH = os.path.join(os.path.abspath(__file__+ "../../"), '.dataset/')
ROOT_OUT_PATH = os.path.join(os.path.abspath(__file__+ "../../"), '.output/')
ROOT_FIGURE_PATH = ROOT_OUT_PATH+".test/"
fileName = "randomSound"
fileExt = ".wav"
inputFile = ROOT_INPUT_PATH+fileName+  fileExt
transFile = ROOT_INPUT_PATH+fileName+"_32" + fileExt

checkPath(ROOT_OUT_PATH)
checkPath(ROOT_INPUT_PATH)
checkPath(ROOT_FIGURE_PATH)

#### Trnasform original file into 32bit-wave file
if not os.path.exists(transFile): 
    data, samplerate = librosa.load(inputFile, dtype='float32')
    librosa.output.write_wav(transFile, data, samplerate)

#### Define DWT(Discrete Wavelet Transform) Constant Variables
DWT_LEVEL=3
DWT_SAMPLE_RATE=22050 # 22.050kHz, 1초당 추출되는 샘플개수
DWT_NAME='db2'
DWT_MODE='smooth'

# Wave 사운드 파일로부터 데이터 추출
#### Load
# Return the sample rate (in samples/sec), data from a WAV file, Wave Format PCM
fs, samples_murmur = sio.wavfile.read(transFile)
print("Wave Info\n  Sample Rate={0}, ".format(fs)) # 22.050kHz, 1초당 추출되는 샘플개수
print("  Data Length={0}\n  Data={1}".format(len(samples_murmur), samples_murmur))

# 추출된 데이터로 이산웨이블릿트랜스폼 적용(레벨3)
# tree(cA, cD3, cD2, cD1) 추출
### Discrete Wavelet Info
# pywt.Wavelet: Describes properties of a discrete wavelet identified by the specified wavelet name, must be a valid wavelet name from the pywt.wavelist() list.
# wavelist: 'haar', 'db', 'sym', 'coif', 'bior', 'rbio', 'dmey', 'gaus', 'mexh', 'morl', 'cgau', 'shan', 'fbsp', 'cmor'
discrete_wavelet = pywt.Wavelet('db2')
print(discrete_wavelet)
max_level = pywt.dwt_max_level(len(samples_murmur), discrete_wavelet)
print('MAXIMUM DECOMPOSE LEVEL = ', max_level)

targetData = samples_murmur.copy() # NO read only

#### Discrete Wavelet Transform
# pywt.wavedec: Multilevel 1D Discrete Wavelet Transform of data. 
# Parameters: data, wavelet, mode='symmetric', level=None, axis=-1
# Returns: [cA_n, cD_n, cD_n-1, …, cD2, cD1] : list
originalMatrix = pywt.wavedec(data=targetData, wavelet='db2', level=3)
cA3, cD3, cD2, cD1 = originalMatrix
print("< Discrete Wavelet Transform >\n" + "  cD1: {0}\n  cD2: {1}\n  cD3: {2}\n  cA3: {3}\n".format(cD1,cD2,cD3,cA3))

# 랜덤변수 데이터 생성(레벨3, sD1, sD2, sD3)하여 학습 >> cD1, cD2, cD3와 유사하도록
# 학습된 sD1, sD2, sD3 신경망을 통해 나온 결과를 reconstruct_sample = pywt.waverec(tree, 'db2')하여 하나의 

sample_cD1 = (np.random.random((20000,1))-0.5)*2 # 1 by 2000, -1.0 ~ +1.0
sample_cD2 = (np.random.random((20000,1))-0.5)*2
sample_cD3 = (np.random.random((20000,1))-0.5)*2
sample_tree = np.append(sample_cD1, sample_cD2, axis=1)
sample_tree = np.append(sample_tree, sample_cD3, axis=1)
print(sample_tree)

sample_rate = DWT_SAMPLE_RATE
outputFile = ROOT_FIGURE_PATH + fileName+fileExt

sio.wavfile.write(outputFile, sample_rate, sample_tree)
rate, data = sio.wavfile.read(outputFile)

print("sample_rate: ", rate)
print("shape of data: ", data.shape)

# Visualize
plt.figure(figsize=(15,10))
plt.subplot(6,1,1)
plt.title('cD1')
plt.plot(np.linspace(0.0, len(sample_cD1), len(sample_cD1)), sample_cD1)
plt.grid()

plt.subplot(6,1,2)
plt.title('cD2')
plt.plot(np.linspace(0.0, len(sample_cD2), len(sample_cD2)), sample_cD2)
plt.grid()

plt.subplot(6,1,3)
plt.title('cD3')
plt.plot(np.linspace(0.0, len(sample_cD3), len(sample_cD3)), sample_cD3)
plt.grid()

plt.subplot(6,1,4)
plt.title('Construct Tree')
plt.plot(np.linspace(0.0, len(sample_tree), len(sample_tree)), sample_tree)
plt.grid()

plt.tight_layout()
plt.show()
plt.close()