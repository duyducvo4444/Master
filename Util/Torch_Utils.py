import torch
from scipy.signal import get_window
import math
import torch.nn.functional as F


def segment(A, frame_length=512, hop_len=256, wintype=None):
    if wintype is None:
        window = torch.ones(frame_length, device=A.device)
    else:
        window = get_window(wintype, frame_length, fftbins=True)
    A = torch.squeeze(A)
    sig_len = A.shape[-1]
    num_seg = math.ceil((sig_len - frame_length) / hop_len) + 1  # Calculates number of segments
    diff = frame_length + (num_seg - 1) * hop_len - sig_len  # Calculates how many to pad
    A = F.pad(A, (0, diff))
    result = torch.empty((num_seg, A.shape[0], frame_length), dtype=A.dtype, device=A.device)
    for i in range(num_seg):
        start_idx = hop_len * i
        result[i, :, :] = window * A[:, start_idx:(start_idx + frame_length)]
    return result


def unify(A, hop_len=256):
    num_seg = A.shape[0]
    framelen = A.shape[-1]
    result = torch.zeros((A.shape[1], (num_seg - 1) * hop_len + framelen), dtype=A.dtype, device=A.device)
    result[:, :framelen] += A[0, :, :]
    gap = framelen - hop_len
    for i in range(1, A.shape[0]):
        start_idx = hop_len * i
        result[:, start_idx:(start_idx + framelen)] += A[i, :, :]
        result[:, start_idx:(start_idx + gap)] = result[:, start_idx:(i * (framelen - 1) + 1)] / 2.0
    return result


def pi():
    return torch.Tensor([math.pi])


def tensor2numpy(x, squeeze=False):
    if squeeze:
        x = torch.squeeze(x)
    return x.detach().cpu().numpy()
