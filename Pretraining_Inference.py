import argparse
import json
import numpy as np
import soundfile as sf
from models.ComplexVQ2 import ComplexVQ2
from models.InvGammatone_Origin import InvConvGamma
import torch
import librosa
import matplotlib.pyplot as plt


def get_loss(loadedTorch):
    return loadedTorch['loss'], \
           loadedTorch['test_loss'], \
           loadedTorch['perplex_top'], \
           loadedTorch['perplex_bot']


def plot_sth(sth, xtit='', ytit='', tit=''):
    xaxis = np.arange(0, len(sth))
    plt.figure()
    plt.plot(xaxis, sth)
    plt.xlabel(xtit)
    plt.ylabel(ytit)
    plt.title(tit)


def plot_loss(torchpt):
    loaded = torch.load(torchpt)
    loss, tloss, ptop, pbot = get_loss(loaded)
    plot_sth(loss, tit="Training loss (log)", xtit="Iteration")
    plot_sth(tloss, tit="Testing loss (log)", xtit="Iteration")
    plt.show()


# plot_loss('Trained/SEModel_VQ2_80.pt')
device = "cuda" if torch.cuda.is_available() else "cpu"
temp = torch.load('Trained/SEModel_VQ2_180.pt')
fname = 'clnsp16'

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, required=True,
                    help='JSON file for configuration')
args = parser.parse_args()
with open(args.config) as f:
    config = json.load(f)
global model_config
model_config = config['model_configs']

x, sr = librosa.load('wav/test/' + fname + '.wav', sr=16000, duration=2.0)
sf.write('wav/test/ex/' + fname + '.wav', x, samplerate=sr)
x = torch.from_numpy(x).float()
x = torch.unsqueeze(x, 0).to(device)

ana = ComplexVQ2(**model_config).to(device)
ana.copy_state_dict(temp['state_dict'])
invmodel = InvConvGamma(NumCh=32, f0=600, fs=16000).to(device)

log_var_speech, s_mix_r, s_mix_i = ana.inference(x)
phase = torch.atan2(s_mix_i, s_mix_r)
var_speech = torch.exp(log_var_speech)

mag_mix = torch.sqrt(var_speech)
xout = invmodel(mag_mix*torch.cos(phase))

xout = np.squeeze(xout.detach().cpu().numpy())

sf.write('wav/test/ex/' + fname + '_s.wav', xout, samplerate=sr)
print('done')
