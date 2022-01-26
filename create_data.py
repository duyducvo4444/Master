import random
import numpy as np
from torch.utils.data import Dataset
import glob
from Utils import SNR_gain
import torch
import librosa


class CreateDataForSE(Dataset):
    def __init__(self, clean_url,
                 noise_url,
                 samplerate=16000,
                 augmentation=False,
                 duration=None):
        self.clean_url = clean_url  # '/*.wav'
        self.noise_url = noise_url  # '/*.wav'
        self.noise_snr = [-5, 20]
        self.sample_rate = samplerate
        self.noise_list = [f for f in glob.glob(noise_url)]
        self.clean_list = [f for f in glob.glob(clean_url)]
        self.duration = duration
        self.augmentation = augmentation

    def __len__(self):
        return len(self.clean_list)

    def __getitem__(self, idx):
        clean_file = self.clean_list[idx]
        clean_speech = self.loadaudio(clean_file, self.sample_rate, duration=self.duration)
        audio_clean, _, _ = self.scale_db(clean_speech, np.random.uniform(-35, -20, 1))
        if self.augmentation:
            noisy_speech = self.additive_noise(clean_speech)
            return clean_speech, noisy_speech
        return clean_speech

    @staticmethod
    def loadaudio(path, sr, duration):
        x, _ = librosa.load(path, sr)
        if duration is not None:
            frame_length = int(sr * duration)
            audio_size = len(x)
            if frame_length > audio_size:
                diff = frame_length - audio_size
                x = np.pad(array=x, pad_width=(0, diff), mode='wrap')
            start = random.randint(0, len(x) - frame_length)
            x = x[start: start + frame_length]
            x = x.astype(np.float32)
        return x

    @staticmethod
    def scale_db(y, target_dB_FS=-25, eps=1e-6):
        rms = np.sqrt(np.mean(y ** 2))
        scalar = 10 ** (target_dB_FS / 20) / (rms + eps)
        y *= scalar
        return y, rms, scalar

    def additive_noise(self, audio_clean):
        noise_name = random.choice(self.noise_list)
        noise, _ = librosa.load(noise_name, sr=self.sample_rate)
        if len(noise) < len(audio_clean):
            noise = np.pad(noise, (0, len(audio_clean) - len(noise)), mode='wrap')
        else:
            noise = noise[:len(audio_clean)]
        snr = random.uniform(self.noise_snr[0], self.noise_snr[1])
        noisy = audio_clean + noise * SNR_gain(audio_clean, noise, snr)
        noisy = noisy.astype(np.float32)
        return noisy


class ToTensor(object):
    def __call__(self, signal):
        return torch.from_numpy(signal)


class AudioSegment(object):
    def __init__(self, win_len, hop_len):
        self.win_len = win_len
        self.hop_len = hop_len
        return

    def __call__(self, batch):
        tf = ToTensor()
        batch = librosa.util.frame(batch, frame_length=self.win_len, hop_length=self.hop_len, axis=0)
        return tf(batch)
