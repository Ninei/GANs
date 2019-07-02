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

# Check Dataset & Output Directory
ROOT_INPUT_PATH = os.path.join(os.path.abspath(__file__+ "../../"), '.dataset/')
ROOT_OUT_PATH = os.path.join(os.path.abspath(__file__+ "../../"), '.output/')
ROOT_FIGURE_PATH = ROOT_OUT_PATH+".figureList/dwt/"
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

discrete_wavelet = pywt.Wavelet('db2')
print(discrete_wavelet)

max_level = pywt.dwt_max_level(len(samples_murmur), discrete_wavelet)
print('MAXIMUM DECOMPOSE LEVEL = ',max_level)

targetData = samples_murmur.copy() # NO read only

# decompose
tree = pywt.wavedec(data=targetData, wavelet='db2', level=3)
cA3, cD3, cD2, cD1 = tree
# print(len(cD1),len(cD2),len(cD3),len(cA3))

# reconstruct
rec_sample = pywt.waverec(tree, 'db2')
rec_to_orig = pywt.idwt(None, cD1, 'db2', 'smooth')  #
rec_to_level1 = pywt.idwt(None, cD2, 'db2', 'smooth')
rec_to_level2_from_detail = pywt.idwt(None, cD3, 'db2', 'smooth')
rec_to_level2_from_approx = pywt.idwt(cA3, None, 'db2', 'smooth')
#print(len(rec_to_orig),len(rec_to_level1),len(rec_to_level2_from_detail),len(rec_to_level2_from_approx))

# visualize
plt.figure(figsize=(4,4))
(phi, psi, x) = discrete_wavelet.wavefun()
plt.plot(x, phi)
plt.savefig(ROOT_FIGURE_PATH+fileName+"_figure_1.png")
plt.show()
plt.figure(figsize=(15,10))
plt.subplot(5,1,1)
plt.title(fileName+fileExt)
plt.xlabel('Samples')
plt.plot(np.linspace(0.0, len(samples_murmur),len(samples_murmur)), samples_murmur)
plt.xlim(xmin=0)
plt.grid()

plt.subplot(5,1,2)
plt.title('cD1')
plt.plot(np.linspace(0.0, len(rec_to_orig),len(rec_to_orig)), rec_to_orig)
plt.xlim(xmin=0)
plt.grid()
plt.subplot(5,1,3)
plt.title('cD2')
plt.plot(np.linspace(0.0, len(rec_to_level1),len(rec_to_level1)), rec_to_level1)
plt.xlim(xmin=0)
plt.grid()
plt.subplot(5,1,4)
plt.title('cD3')
plt.plot(np.linspace(0.0, len(rec_to_level2_from_detail),len(rec_to_level2_from_detail)), rec_to_level2_from_detail)
plt.xlim(xmin=0)
plt.grid()
plt.subplot(5,1,5)
plt.title('cA3')
plt.plot(np.linspace(0.0, len(rec_to_level2_from_approx),len(rec_to_level2_from_approx)), rec_to_level2_from_approx)
plt.xlim(xmin=0)
plt.grid()
plt.tight_layout()
plt.savefig(ROOT_FIGURE_PATH+fileName+"_figure_2.png")
plt.show()