import os
import pywt
import matplotlib.pyplot as plt
import numpy as np
import librosa
import scipy

from scipy import signal
from scipy.io import wavfile

def checkPath(target) :
    if not os.path.exists(target): os.makedirs(target)

#### Check Dataset & Output Directory
ROOT_INPUT_PATH = os.path.join(os.path.abspath(__file__+ "../../"), '.dataset/')
ROOT_OUT_PATH = os.path.join(os.path.abspath(__file__+ "../../"), '.output/')
ROOT_FIGURE_PATH = ROOT_OUT_PATH+".figureList/cwt/"
fileName = "Loop_0"
fileExt = ".wav"
inputFile = ROOT_INPUT_PATH+fileName+  fileExt
transFile = ROOT_INPUT_PATH+fileName+"_32" + fileExt

checkPath(ROOT_OUT_PATH)
checkPath(ROOT_INPUT_PATH)
checkPath(ROOT_FIGURE_PATH)

if not os.path.exists(transFile): 
    data, samplerate = librosa.load(inputFile, dtype='float32')
    librosa.output.write_wav(transFile, data, samplerate)

fs, samples_murmur = wavfile.read(transFile)
print("Wave Info\n  Sample Rate={0}, ".format(fs)) # 22.050kHz, 1초당 추출되는 샘플개수
print("  Data length={0}, Data List={1}".format(len(samples_murmur), samples_murmur))

continuous_wavelet = pywt.ContinuousWavelet('mexh') # Mexican Hat Wavelet
print(continuous_wavelet)

max_scale = 20
scales = np.arange(1, max_scale + 1)
cwtmatr, freqs = pywt.cwt(samples_murmur, scales, continuous_wavelet, 44100)
print("CWT INfo\n  Frequence List ={0}, ".format(freqs))
print("  Data length={0}\n  Data List={1}".format(len(cwtmatr), cwtmatr))
    
#### visualize
# plt.figure(figsize=(4,4))
# (phi, psi) = continuous_wavelet.wavefun()
# plt.plot(psi,phi)
# plt.title("CWT Info.png")
# plt.savefig(ROOT_FIGURE_PATH+fileName+"_Info_CWT.png")
# plt.show()

plt.figure(figsize=(20,10))
plt.subplot(2,1,1) # 2행, 1열, 1번째
plt.title(fileName+fileExt + ' Sample')
plt.plot(np.linspace(0.0, len(samples_murmur),len(samples_murmur)), samples_murmur)
plt.xlim(xmin=0)
plt.grid()

plt.subplot(2,1,2)
plt.title(fileName+fileExt +' CWT Figure')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (sec)')
plt.imshow(cwtmatr, extent=[0, int(len(samples_murmur)), 1, max_scale + 1], cmap='PRGn', aspect='auto', vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
plt.colorbar()
plt.savefig(ROOT_FIGURE_PATH+fileName+"_Figure_CWT.png")
plt.show()