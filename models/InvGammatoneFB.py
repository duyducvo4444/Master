import numpy as np
import torch
import torch.nn as nn
import math


class InvConvGamma(nn.Module):

    def __init__(self, NumCh=64, f0=600, fs=16000):
        super(InvConvGamma, self).__init__()
        self.NumCh = NumCh + 1
        self.f0 = f0
        self.sampling_rate = fs

        self.a = 10 ** (2 / NumCh)
        self.ERBw = 24.7 * (4.37 * f0 / 1000 + 1)
        self.bCoeff = 1.019
        self.bERBw = self.bCoeff * self.ERBw
        self.Kmax = NumCh / 2
        self.Kmin = -NumCh / 2
        self.ctrCh = NumCh / 2 + 1

        self.kth = np.transpose(np.linspace(self.Kmin, self.Kmax, NumCh + 1))
        self.invAmp = self.a ** self.kth
        self.alineLen = np.ceil(3 * fs * (self.a ** (-self.kth)) / (2 * math.pi * self.bERBw)).astype(int)
        self.tauMax = np.max(self.alineLen)
        self.tau = np.arange(0, self.tauMax / fs / (self.a ** self.Kmin), 1 / fs)

        self.gtLen = np.array([len(np.arange(0, self.tauMax / fs / self.a ** (nch + self.Kmin), 1 / fs))
                               for nch in range(NumCh + 1)])

        gt = self.InvGammaIR(self.invAmp, self.tau, self.f0, self.bCoeff, self.ERBw, self.gtLen)
        fil_len = gt.shape[-1]
        self.pad_amount = fil_len - 1

        self.inv_gamma = nn.Conv1d(in_channels=self.NumCh, out_channels=self.NumCh, padding=self.pad_amount,
                                   kernel_size=fil_len, bias=False, groups=self.NumCh)

        init_inv_gamma = torch.FloatTensor(gt[:, None, :])
        self.inv_gamma.weight = torch.nn.Parameter(init_inv_gamma, requires_grad=False)

    def forward(self, inp):
        device = inp.device

        real_part = inp[:, :self.NumCh, :]
        imag_part = inp[:, self.NumCh:, :]
        real_sig = torch.sqrt(real_part**2 + imag_part**2) * torch.cos(torch.atan2(imag_part, real_part))

        num_batches, _, num_samples = real_sig.shape
        y_inv = self.inv_gamma(real_sig) / self.sampling_rate

        align_index = np.linspace(start=self.gtLen - self.alineLen, stop=self.gtLen - self.alineLen + num_samples,
                                  num=num_samples, endpoint=False, axis=-1, dtype=int)
        aline_ind = torch.tensor(align_index, dtype=torch.long).expand(num_batches, -1, num_samples).to(device)
        output = torch.gather(input=y_inv, dim=-1, index=aline_ind)
        output = torch.sum(output, -2)
        return output

    @staticmethod
    def GammaTone(t, f0, bCoef=1.019, ERBw=None, Ftype='real'):
        n = 4
        bERB = bCoef * ERBw
        N = math.factorial(n - 1)
        amp = 2 * (2 * math.pi * bERB) ** n / N
        if Ftype == 'real':
            gt = amp * (t ** (n - 1)) * np.exp(-2 * math.pi * bERB * t) * np.cos(2 * math.pi * f0 * t)
        elif Ftype == 'imag':
            gt = amp * (t ** (n - 1)) * np.exp(-2 * math.pi * bERB * t) * np.sin(2 * math.pi * f0 * t)
        elif Ftype == 'complex':
            gt = amp * (t ** (n - 1)) * np.exp(-2 * math.pi * bERB * t) * np.exp(2 * math.pi * f0 * t)
        else:
            raise ValueError('Ftype should be "real", "imag", or "complex".')
        return gt

    def InvGammaIR(self, invAmp, tau, f0, bCoeff, ERBw, gtLen):
        gt = (np.expand_dims(invAmp, axis=1) @ np.ones((1, len(tau)))) * \
             self.GammaTone(np.expand_dims(invAmp, axis=1) * np.transpose(np.expand_dims(tau, axis=1)),
                            f0, bCoeff, ERBw, 'real')
        out = np.zeros(gt.shape)
        samples = gt.shape[-1]
        for i in range(out.shape[0]):
            temp = gt[i, :self.gtLen[i]]
            out[i] = np.pad(array=temp, pad_width=(samples - gtLen[i], 0))
        return out
