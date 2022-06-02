import json
import numpy as np
import soundfile as sf
from models.ComplexVQ2 import ComplexVQ2
import torch
import librosa


device = "cuda" if torch.cuda.is_available() else "cpu"
temp = torch.load('Trained/NaN.pt')
fname = 'clnsp654'

with open('Configuration/train_config.json') as f:
    config = json.load(f)
global model_config
model_config = config['model_configs']

x, sr = librosa.load('wav/test/' + fname + '.wav', sr=16000, duration=2.0)
sf.write('wav/test/ex/' + fname + '.wav', x, samplerate=sr)
x = torch.from_numpy(x).float()
x = torch.unsqueeze(x, 0).to(device)

ana = ComplexVQ2(**model_config).to(device)
ana.copy_state_dict(temp)

xout, _, _, _ = ana.inference(x, pretrain=False)

xout = np.squeeze(xout.detach().cpu().numpy())

sf.write('wav/test/ex/' + fname + '_s.wav', xout, samplerate=sr)
print('done')
