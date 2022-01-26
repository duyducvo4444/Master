import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import get_window
import math


def waveplot(sig, sr=16000, title=''):
    plt.figure(figsize=(9, 5))
    librosa.display.waveplot(sig, sr)
    plt.xlabel('Time')
    plt.title(title)


# Plot spectrogram
def plot_spectrogram(sigf, tit=''):
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(np.log10(sigf + 1e-12), aspect="auto", origin="lower")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.title(tit)


# Load audio to calculate score
def loadAudioToEval(cleanPath, noisyPath, sr=16000):
    ref, fs = librosa.load(cleanPath, sr=sr)
    deg, fs = librosa.load(noisyPath, sr=sr)
    if len(ref) > len(deg):
        dif = len(ref) - len(deg)
        deg = np.pad(deg, (0, dif))
    else:
        deg = deg[:len(ref)]
    return ref, deg, fs


# Calculate mean squared error
def mse(x1, x2):
    dif = np.subtract(x1, x2)
    dif = np.square(dif)
    return dif.mean()


# Calculate power spectrum of a signal
def powSpec(x):
    return np.sum(np.square(x)) / len(x)


# Calculate log spectral distance
def lsd(x1, x2):
    const = 1 / (2 * np.pi)
    dis = np.sum(np.square(10 * (np.log10(powSpec(x1)) - np.log10(powSpec(x2)))))
    return np.sqrt(const * dis)


def SNR_dB(clean, noisy, mode='log'):
    """"
    Calculate SNR
    mode is either 'linear' or 'log'
    clean and noisy must have the same length
    """
    length = min(len(clean), len(noisy))
    clean = clean[:length]
    noisy = noisy[:length]
    sigmaClean = powSpec(clean)
    sigmaNoise = powSpec(noisy - clean)
    if mode == 'linear':
        return sigmaClean / sigmaNoise
    elif mode == 'log':
        return 10 * np.log10(sigmaClean / sigmaNoise)
    else:
        raise TypeError("mode must be either linear or log.")


def hyper_norm(x, C=0.1, K=10):
    return K * (1 - np.exp(-C * x)) / (1 + np.exp(-C * x))


def hyper_denorm(y, C=0.1, K=10):
    return (-1 / C) * np.log((K - y) / (K + y))


def loadaudio(path, sr=22050, mono=True, offset=0.0, duration=None, dtype=np.float32, res_type="kaiser_best"):
    x, _ = librosa.load(path, sr=sr, mono=mono, offset=offset, duration=duration, dtype=dtype, res_type=res_type)
    if duration is not None:
        frame_length = int(sr * duration)
        if frame_length > x.size:
            diff = frame_length - x.size
            x = np.pad(x, (0, diff))
    return x, sr


def SNR_gain(clean, noise, snr):
    return np.sqrt(powSpec(clean) / powSpec(noise) / 10 ** (snr / 10))


def segment(A, frame_length=512, hop_len=256, wintype=None):
    """Framing a 2d matrix into a 3d matrix"""
    if wintype is None:
        window = np.ones(frame_length)
    else:
        window = get_window(wintype, frame_length, fftbins=True)
    if A.ndim == 1:
        A = np.expand_dims(A, axis=0)
    sig_len = A.shape[1]
    num_seg = math.ceil((sig_len - frame_length) / hop_len) + 1  # Calculates number of segments
    diff = frame_length + (num_seg - 1) * hop_len - sig_len  # Calculates how many to pad
    A = np.pad(A, ((0, 0), (0, diff)))
    result = np.empty((num_seg, A.shape[0], frame_length), dtype=A.dtype)
    for i in range(num_seg):
        start_idx = hop_len * i
        result[i, :, :] = window * A[:, start_idx:(start_idx + frame_length)]
    return result


def unify(A, hop_len=256):
    """Revert the 3d matrix back to 2d matrix"""
    num_seg = A.shape[0]
    framelen = A.shape[-1]
    result = np.zeros((A.shape[1], (num_seg - 1) * hop_len + framelen), dtype=np.float32)
    result[:, :framelen] += A[0, :, :]
    gap = framelen - hop_len
    for i in range(1, A.shape[0]):
        start_idx = hop_len * i
        result[:, start_idx:(start_idx + framelen)] += A[i, :, :]
        result[:, start_idx:(start_idx + gap)] /= 2
    return result


def minmaxnorm(x):
    xmin = x.min()
    xmax = x.max()
    return (x - xmin) / (xmax - xmin)
