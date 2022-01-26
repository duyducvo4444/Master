import pysepm
from pystoi import stoi
from Utils import loadAudioToEval, SNR_dB

cleanPath = 'temp2.wav'
noisyPath = 'temp.wav'
ref, deg, fs = loadAudioToEval(cleanPath, noisyPath, sr=16000)

print('PESQ: ' + str(pysepm.pesq(ref, deg, fs)))
print('SNR(dB):' + str(SNR_dB(ref, deg, mode='log')))
print('STOI: ' + str(stoi(ref, deg, fs, extended=False)))

################################################################

# import matplotlib.pyplot as plt
# import librosa.display
#
# sr = 16000
# y, sr = librosa.load('wav/Clean/female.wav', sr=sr)
# y1, sr1 = librosa.load('wav/female_syn.wav', sr=sr)
# y2, sr2 = librosa.load('wav/female_airCon_10.wav', sr=sr)
# plt.figure(figsize=(15, 4))
# plt.subplot(1, 3, 1)
# librosa.display.waveshow(y, sr=sr)
# plt.xlabel('Time')
# plt.title('Clean speech')
# plt.subplot(1, 3, 2)
# librosa.display.waveshow(y1, sr=sr)
# plt.xlabel('Time')
# plt.title('Estimated speech')
# plt.subplot(1, 3, 3)
# librosa.display.waveshow(y2, sr=sr)
# plt.xlabel('Time')
# plt.title('Noisy speech')
# plt.show()