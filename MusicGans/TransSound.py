import os
import librosa
import scipy.io as sio
import pywt
import matplotlib.pyplot as plt
import numpy as np


class TransSound:
    
    def __init__(self):
        self.checkPath(ROOT_INPUT_PATH)
        self.checkPath(ROOT_OUT_PATH)
        if not os.path.exists(transFile): 
            data, samplerate = librosa.load(inputFile, dtype='float32')
            librosa.output.write_wav(transFile, data, samplerate)
        super().__init__()

    def __new__(cls):
        return super().__new__(cls)

    def checkPath(self, target):
        if not os.path.exists(target): os.makedirs(target)

    def getSourceFile(self):
        return transFile

    def getOutputFile(self):
        return outputFile

    def getMaxLevel(self, target, targeWavelet):
        discrete_wavelet = pywt.Wavelet(targeWavelet)
        max_level = pywt.dwt_max_level(len(target), discrete_wavelet)
        return max_level

    def getSource(self, targetFile, targeWavelet, targetLevel):
        sampleRate, sampleData = sio.wavfile.read(targetFile)
        targetData = sampleData.copy()
        # pywt.wavedec: https://pywavelets.readthedocs.io/en/latest/ref/dwt-discrete-wavelet-transform.html
        sampleList = pywt.wavedec(data=targetData, wavelet=targeWavelet, level=targetLevel)
        return sampleRate, sampleList, targetData
    
    def traceFigure(self, targetList, targetRate, targetFile, targetWavelet):
        sampleList = pywt.waverec(targetList, 'db2')
        sio.wavfile.write(targetFile, targetRate, sampleList)

        cA3, cD3, cD2, cD1 = targetList
        rec_to_orig = pywt.idwt(None, cD1, targetWavelet, 'smooth')
        rec_to_level1 = pywt.idwt(None, cD2, targetWavelet, 'smooth')
        rec_to_level2_from_detail = pywt.idwt(None, cD3, targetWavelet, 'smooth')
        rec_to_level2_from_approx = pywt.idwt(cA3, None, targetWavelet, 'smooth')

        plt.figure(figsize=(15,10))
        plt.subplot(6,1,1)
        plt.title('Create Sample')
        plt.plot(np.linspace(0.0, len(sampleList),len(sampleList)), sampleList)
        plt.grid()

        plt.subplot(6,1,2)
        plt.title('cD1')
        plt.plot(np.linspace(0.0, len(rec_to_orig),len(rec_to_orig)), rec_to_orig)
        plt.grid()

        plt.subplot(6,1,3)
        plt.title('cD2')
        plt.plot(np.linspace(0.0, len(rec_to_level1),len(rec_to_level1)), rec_to_level1)
        plt.grid()

        plt.subplot(6,1,4)
        plt.title('cD3')
        plt.plot(np.linspace(0.0, len(rec_to_level2_from_detail),len(rec_to_level2_from_detail)), rec_to_level2_from_detail)
        plt.grid()

        plt.subplot(6,1,5)
        plt.title('cA3')
        plt.plot(np.linspace(0.0, len(rec_to_level2_from_approx),len(rec_to_level2_from_approx)), rec_to_level2_from_approx)
        plt.grid()

        plt.tight_layout()
        plt.show()
        plt.close()


ROOT_INPUT_PATH = os.path.join(os.path.abspath(__file__+ "../../"), '.dataset/')
ROOT_OUT_PATH = os.path.join(os.path.abspath(__file__+ "../../"), '.output/.createGAN/')
inputFileName = "Loop_0"
outputFileName =  "randomSound"
fileExt = ".wav"
inputFile = ROOT_INPUT_PATH + inputFileName + fileExt
transFile = ROOT_INPUT_PATH + inputFileName + "_32" + fileExt
outputFile = ROOT_OUT_PATH + outputFileName +fileExt