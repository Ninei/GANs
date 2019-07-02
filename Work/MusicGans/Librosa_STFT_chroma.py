import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display


# Check Dataset & Output Directory
ROOT_DATA_PATH = os.path.join(os.path.abspath(__file__+ "../../"), '.dataset/')
ROOT_RESULT_PATH = os.path.join(os.path.abspath(__file__+ "../../"), '.output/')
def checkPath(target) :
    if not os.path.exists(target): os.makedirs(target)
checkPath(ROOT_RESULT_PATH)
checkPath(ROOT_DATA_PATH)
# Defina Scopre ?????
GENERATOR_SCOPE = "GAN/Generator"
DISCRIMINATOR_SCOPE = "GAN/Discriminator"


data, samplerate = librosa.load(ROOT_DATA_PATH+"Loop_0.wav", dtype='float32')
librosa.feature.chroma_stft(y=data, sr=samplerate)

S = np.abs(librosa.stft(data, n_fft=4096))**2
chroma = librosa.feature.chroma_stft(S=S, sr=samplerate)
print(chroma)

plt.figure(figsize=(10, 4))
librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
plt.colorbar()
plt.title('Chromagram')
plt.tight_layout()
plt.show()


