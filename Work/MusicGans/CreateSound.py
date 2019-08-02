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


# Wave 사운드 파일로 부터 데이터 추출
# 추출된 데이터로 이산웨이블릿트랜스폼 적용(레벨3)
# tree(cA, cD3, cD2, cD1) 추출
# 랜덤변수 데이터 생성(레벨3, sD1, sD2, sD3)하여 학습 >> cD1, cD2, cD3와 유사하도록
# 학습된 sD1, sD2, sD3 신경망을 통해 나온 결과를 reconstruct_sample = pywt.waverec(tree, 'db2')하여 하나의 

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