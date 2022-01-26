import numpy as np
import torch
import torch.nn as nn
import math


class ConvGamma(nn.Module):

    def __init__(self, NumCh=64, f0=600, fs=16000):
        super(ConvGamma, self).__init__()
        self.NumCh = NumCh
        self.f0 = f0
        self.sampling_rate = fs

        self.a = 10 ** (2 / NumCh)
        self.ERBw = 24.7 * (4.37 * f0 / 1000 + 1)
        self.bCoeff = 1.019
        self.bERBw = self.bCoeff * self.ERBw
        self.upperK = NumCh / 2
        self.lowerK = -NumCh / 2
        self.kth = np.transpose(np.linspace(self.lowerK, self.upperK, NumCh + 1))
        self.invAmp = self.a ** self.kth
        self.alineLen = np.ceil(3 * fs * (self.a ** (-self.kth)) / (2 * math.pi * self.bERBw)).astype(int)
        self.tauMax = np.max(self.alineLen)
        self.tau = np.arange(0, self.tauMax / fs / (self.a ** self.lowerK), 1 / fs)

        gtRe, gtIm = self.GammaIR(self.invAmp, self.tau, self.f0, self.bCoeff, self.ERBw)
        fil_len = gtRe.shape[-1]
        self.pad_amount = fil_len - 1

        # Filterbanks to generate gamatone (real part, imag part)
        self.gamma_filt_r = nn.Conv1d(in_channels=1, out_channels=self.NumCh + 1, padding=self.pad_amount,
                                      kernel_size=fil_len, bias=False)
        self.gamma_filt_i = nn.Conv1d(in_channels=1, out_channels=self.NumCh + 1, padding=self.pad_amount,
                                      kernel_size=fil_len, bias=False)

        # initialize the kernels of the Gamatone FilterBank
        init_gamma_filts_r = torch.FloatTensor(gtRe[:, None, :])
        init_gamma_filts_i = torch.FloatTensor(gtIm[:, None, :])

        self.gamma_filt_r.weight = torch.nn.Parameter(torch.flip(init_gamma_filts_r, dims=[-1]), requires_grad=False)
        self.gamma_filt_i.weight = torch.nn.Parameter(torch.flip(init_gamma_filts_i, dims=[-1]), requires_grad=False)

    def forward(self, x, typex='complex'):
        device = x.device
        if len(x.shape) == 2:
            num_batches = x.shape[0]
            num_samples = x.shape[1]
        else:
            num_batches = 1
            num_samples = x.shape[0]
        x = x.view(num_batches, 1, num_samples)

        # run samples thru analysis banks
        y_r = self.gamma_filt_r(x) / self.sampling_rate
        y_i = self.gamma_filt_i(x) / self.sampling_rate

        align_index = np.linspace(start=self.alineLen, stop=self.alineLen + num_samples,
                                  num=num_samples, endpoint=False, axis=-1, dtype=int)

        aline_ind = torch.tensor(align_index, dtype=torch.long).expand(num_batches, -1, num_samples).to(device)

        y_real = torch.gather(input=y_r, dim=-1, index=aline_ind)
        y_imag = torch.gather(input=y_i, dim=-1, index=aline_ind)
        cfs = self.Ch2Freq(self.kth, self.f0, self.NumCh)
        cf = torch.tensor(cfs.reshape((-1, 1))).to(device)

        if typex == 'complex':
            return y_real, y_imag, cf.float()
        elif typex == 'ana':
            time = torch.arange(0, num_samples / self.sampling_rate, 1 / self.sampling_rate).to(device)
            amp = torch.sqrt(y_real ** 2 + y_imag ** 2)
            phase = torch.atan2(y_imag, y_real) - torch.unsqueeze(2 * torch.tensor(math.pi).to(device) * cf * time, 0)
            return amp, phase.float(), cf.float()

    @staticmethod
    def GammaTone(t, f0, bCoef=1.019, ERBw=None, Ftype='real'):
        n = 4
        bERB = bCoef * ERBw
        amp = (2 * math.pi * bERB) ** n / (n - 1)
        if Ftype == 'real':
            gt = amp * (t ** (n - 1)) * np.exp(-2 * math.pi * bERB * t) * np.cos(2 * math.pi * f0 * t)
        elif Ftype == 'imag':
            gt = amp * (t ** (n - 1)) * np.exp(-2 * math.pi * bERB * t) * np.sin(2 * math.pi * f0 * t)
        elif Ftype == 'complex':
            gt = amp * (t ** (n - 1)) * np.exp(-2 * math.pi * bERB * t) * np.exp(2 * math.pi * f0 * t)
        else:
            raise ValueError('Ftype should be "real", "imag", or "complex".')
        return gt

    @staticmethod
    def Ch2Freq(Ch, f0, NumCh):
        a = 10 ** (2 / NumCh)
        Frs = f0 * a ** Ch
        return Frs

    def GammaIR(self, invAmp, tau, f0, bCoeff, ERBw):
        gtRe = (np.expand_dims(invAmp, axis=1) @ np.ones((1, len(tau)))) * \
               self.GammaTone(np.expand_dims(invAmp, axis=1) * np.transpose(np.expand_dims(tau, axis=1)),
                              f0, bCoeff, ERBw, 'real')
        gtIm = (np.expand_dims(invAmp, axis=1) @ np.ones((1, len(tau)))) * \
               self.GammaTone(np.expand_dims(invAmp, axis=1) * np.transpose(np.expand_dims(tau, axis=1)),
                              f0, bCoeff, ERBw, 'imag')
        return gtRe, gtIm
