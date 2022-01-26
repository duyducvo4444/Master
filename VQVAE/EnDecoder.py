import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_size, num_hiddens):
        super(Encoder, self).__init__()

        self._conv_1 = nn.Conv1d(in_channels=input_size,
                                 out_channels=num_hiddens // 2,
                                 kernel_size=1024,
                                 stride=2, padding=1, dilation=1)
        self._conv_2 = nn.Conv1d(in_channels=num_hiddens // 2,
                                 out_channels=num_hiddens,
                                 kernel_size=1024,
                                 stride=2, padding=1, dilation=1)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(x)

        x = self._conv_2(x)
        x = F.relu(x)
        return x


class Decoder(nn.Module):
    def __init__(self, input_size, num_hiddens):
        super(Decoder, self).__init__()

        self._conv_trans_1 = nn.ConvTranspose1d(in_channels=num_hiddens,
                                                out_channels=num_hiddens // 2,
                                                kernel_size=1024,
                                                stride=2, padding=1, dilation=1)

        self._conv_trans_2 = nn.ConvTranspose1d(in_channels=num_hiddens // 2,
                                                out_channels=input_size,
                                                kernel_size=1024,
                                                stride=2, padding=1, dilation=1)

    def forward(self, inputs):
        x = self._conv_trans_1(inputs)
        x = F.relu(x)
        return F.relu(self._conv_trans_2(x))
