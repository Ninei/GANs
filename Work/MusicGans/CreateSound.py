
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.io.wavfile
import os

# Check Dataset & Output Directory
def checkPath(target) :
    if not os.path.exists(target): os.makedirs(target)

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

# Define DWT(Discrete Wavelet Transform) Constant Variables
DWT_LEVEL=3
DWT_SAMPLE_RATE=22050 # 22.050kHz, 1초당 추출되는 샘플개수
DWT_NAME='db2'
DWT_MODE='smooth'


# Wave 사운드 파일로 부터 데이터 추출
# 추출된 데이터로 이산웨이블릿트랜스폼 적용(레벨3)
# tree(cA, cD3, cD2, cD1) 추출
# 랜덤변수 데이터 생성(레벨3, sD1, sD2, sD3)하여 학습 >> cD1, cD2, cD3와 유사하도록
# 학습된 sD1, sD2, sD3 신경망을 통해 나온 결과를 reconstruct_sample = pywt.waverec(tree, 'db2')하여 하나의 

sample_cD1 = (np.random.random((1,2000))-0.5)*2 # 1 by 2000, -1.0 ~ +1.0
sample_cD2 = (np.random.random((1,2000))-0.5)*2
sample_cD3 = (np.random.random((1,2000))-0.5)*2
sample_tree = np.append(sample_cD1, sample_cD2, axis=0)
sample_tree = np.append(sample_tree, sample_cD3, axis=0)
print(sample_tree)

sample_rate = DWT_SAMPLE_RATE
outputFile = ROOT_FIGURE_PATH + fileName+fileExt

sio.wavfile.write(outputFile, sample_rate, sample_tree)
rate, data = sio.wavfile.read(outputFile)

print("sample_rate: ", rate)
print("shape of data: ", data.shape)

# 함수로 정리 필요...
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