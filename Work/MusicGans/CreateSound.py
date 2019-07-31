#numpy_matplotlib_example.py
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.io.wavfile
import os

def checkPath(target) :
    if not os.path.exists(target): os.makedirs(target)

# Check Dataset & Output Directory
ROOT_INPUT_PATH = os.path.join(os.path.abspath(__file__+ "../../"), '.dataset/')
ROOT_OUT_PATH = os.path.join(os.path.abspath(__file__+ "../../"), '.output/')
ROOT_FIGURE_PATH = ROOT_OUT_PATH+".test/"
fileName = "saveTest"
fileExt = ".wav"
inputFile = ROOT_INPUT_PATH+fileName+  fileExt
transFile = ROOT_INPUT_PATH+fileName+"_32" + fileExt

checkPath(ROOT_OUT_PATH)
checkPath(ROOT_INPUT_PATH)
checkPath(ROOT_FIGURE_PATH)

y1 = np.random.randn(2000,1)
y2 = np.random.randn(2000,1)
yy = np.append(y1, y2, axis=1)

sample_rate = 16000

outputFile = ROOT_FIGURE_PATH + fileName+fileExt

sio.wavfile.write(outputFile, sample_rate, yy)
rate, data = sio.wavfile.read(outputFile)

print("sample_rate: ", rate)
print("shape of data: ", data.shape)

plt.plot(yy)
plt.show()