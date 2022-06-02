import librosa.display
import matplotlib.pyplot as plt

x, sr = librosa.load('wav/test/ex/clnsp16.wav', sr=16000)
y, _ = librosa.load('wav/test/ex/clnsp16_s.wav', sr=16000)
plt.figure(figsize=(18, 5))
plt.subplot(1, 2, 1)
librosa.display.waveplot(x, sr)
plt.xlabel('Time')
plt.title('Original signal')

plt.subplot(1, 2, 2)
librosa.display.waveplot(y, sr)
plt.xlabel('Time')
plt.title('Quantized signal')
plt.show()