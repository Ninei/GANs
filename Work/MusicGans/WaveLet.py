import os
import pywt
import matplotlib.pyplot as plt
import numpy as np
import librosa

from scipy import signal
from scipy.io import wavfile

def checkPath(target) :
    if not os.path.exists(target): os.makedirs(target)

# Check Dataset & Output Directory
ROOT_DATA_PATH = os.path.join(os.path.abspath(__file__+ "../../"), '.dateset/')
ROOT_RESULT_PATH = os.path.join(os.path.abspath(__file__+ "../../"), '.output/')
inputFile = ROOT_DATA_PATH+"Loop_0.wav"
outputFile = ROOT_DATA_PATH+"Loop_0_32.wav"

checkPath(ROOT_RESULT_PATH)
checkPath(ROOT_DATA_PATH)

if not os.path.exists(outputFile): 
    data, samplerate = librosa.load(inputFile, dtype='float32')
    librosa.output.write_wav(outputFile, data, samplerate)

fs, samples_murmur = wavfile.read(outputFile)

continuous_wavelet = pywt.ContinuousWavelet('mexh')
print(continuous_wavelet)

max_scale = 20
scales = np.arange(1, max_scale + 1)
cwtmatr, freqs = pywt.cwt(samples_murmur, scales, continuous_wavelet, 44100)

# visualize
plt.figure(figsize=(4,4))
(phi, psi) = continuous_wavelet.wavefun()
plt.plot(psi,phi)
plt.show()
plt.figure(figsize=(15,10))
plt.subplot(2,1,1)
plt.title('Murmur Heart Sound')
plt.xlabel('Samples')
plt.plot(np.linspace(0.0, len(samples_murmur),len(samples_murmur)), samples_murmur)
plt.xlim(xmin=0)
plt.grid()

plt.figure(figsize=(20,10))
plt.subplot(2,1,2)
plt.imshow(cwtmatr, extent=[0, int(len(samples_murmur)), 1, max_scale + 1],cmap='PRGn', aspect='auto', 
           vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
plt.colorbar()

plt.show()