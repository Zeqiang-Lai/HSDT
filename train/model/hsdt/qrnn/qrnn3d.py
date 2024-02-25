from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from functools import partial


from .combinations import *


"""F pooling"""


class ChannelEnhancedFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, bias=False):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff, bias=bias)
        self.w_2 = nn.Linear(d_ff, d_model, bias=bias)
        self.w_3 = nn.Linear(d_model, d_ff, bias=bias)

    def forward(self, input):
        x = self.w_1(input)
        x = F.gelu(x)
        x1 = self.w_2(x)

        x = self.w_3(input)
        x, w = torch.chunk(x, 2, dim=-1)
        x2 = x * torch.sigmoid(w)

        return x1 + x2


class QRNN3DLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels, conv_layer, act='tanh'):
        super(QRNN3DLayer, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        # quasi_conv_layer
        self.conv = conv_layer
        self.act = act
        self.ffn = ChannelEnhancedFeedForward(hidden_channels, hidden_channels * 2)

    def _conv_step(self, inputs):
        gates = self.conv(inputs)
        Z, F = gates.split(split_size=self.hidden_channels, dim=1)
        if self.act == 'tanh':
            return Z.tanh(), F.sigmoid()
        elif self.act == 'relu':
            return Z.relu(), F.sigmoid()
        elif self.act == 'none':
            return Z, F.sigmoid
        else:
            raise NotImplementedError

    def _rnn_step(self, z, f, h):
        # uses 'f pooling' at each time step
        h_ = (1 - f) * z if h is None else f * h + (1 - f) * z
        return h_

    def forward(self, inputs, reverse=False):
        h = None
        Z, F = self._conv_step(inputs)
        h_time = []

        if not reverse:
            for time, (z, f) in enumerate(zip(Z.split(1, 2), F.split(1, 2))):  # split along timestep
                h = self._rnn_step(z, f, h)
                h_time.append(h)
        else:
            for time, (z, f) in enumerate((zip(
                reversed(Z.split(1, 2)), reversed(F.split(1, 2))
            ))):  # split along timestep
                h = self._rnn_step(z, f, h)
                h_time.insert(0, h)

        # return concatenated hidden states
        out = torch.cat(h_time, dim=2)
        out = rearrange(out, 'b c d h w -> b d h w c')
        out = self.ffn(out)
        out = rearrange(out, 'b d h w c -> b c d h w')
        return out

    def extra_repr(self):
        return 'act={}'.format(self.act)


class BiQRNN3DLayer(QRNN3DLayer):
    def _conv_step(self, inputs):
        gates = self.conv(inputs)
        Z, F1, F2 = gates.split(split_size=self.hidden_channels, dim=1)
        if self.act == 'tanh':
            return Z.tanh(), F1.sigmoid(), F2.sigmoid()
        elif self.act == 'relu':
            return Z.relu(), F1.sigmoid(), F2.sigmoid()
        elif self.act == 'none':
            return Z, F1.sigmoid(), F2.sigmoid()
        else:
            raise NotImplementedError

    def forward(self, inputs, fname=None):
        h = None
        Z, F1, F2 = self._conv_step(inputs)
        hsl = []
        hsr = []
        zs = Z.split(1, 2)

        for time, (z, f) in enumerate(zip(zs, F1.split(1, 2))):  # split along timestep
            h = self._rnn_step(z, f, h)
            hsl.append(h)

        h = None
        for time, (z, f) in enumerate((zip(
            reversed(zs), reversed(F2.split(1, 2))
        ))):  # split along timestep
            h = self._rnn_step(z, f, h)
            hsr.insert(0, h)

        # return concatenated hidden states
        hsl = torch.cat(hsl, dim=2)
        hsr = torch.cat(hsr, dim=2)

        if fname is not None:
            stats_dict = {'z': Z, 'fl': F1, 'fr': F2, 'hsl': hsl, 'hsr': hsr}
            torch.save(stats_dict, fname)
        out = hsl + hsr
        out = rearrange(out, 'b c d h w -> b d h w c')
        out = self.ffn(out)
        out = rearrange(out, 'b d h w c -> b c d h w')
        return out


class BiQRNNConv3D(BiQRNN3DLayer):
    def __init__(self, in_channels, hidden_channels, k=3, s=1, p=1, bn=True, act='tanh'):
        super(BiQRNNConv3D, self).__init__(
            in_channels, hidden_channels, BasicConv3d(in_channels, hidden_channels * 3, k, s, p, bn=bn), act=act)


class QRNNConv3D(QRNN3DLayer):
    def __init__(self, in_channels, hidden_channels, k=3, s=1, p=1, bn=True, act='tanh'):
        super(QRNNConv3D, self).__init__(
            in_channels, hidden_channels, BasicConv3d(in_channels, hidden_channels * 2, k, s, p, bn=bn), act=act)

class QRNNDownConv3D(QRNN3DLayer):
    def __init__(self, in_channels, hidden_channels, k=3, s=1, p=1, bn=True, act='tanh'):
        super(QRNNDownConv3D, self).__init__(
            in_channels, hidden_channels, BasicDownConv3d(in_channels, hidden_channels * 2, k, s, p, bn=bn), act=act)


class QRNNUpsampleConv3d(QRNN3DLayer):
    def __init__(self, in_channels, hidden_channels, k=3, s=1, p=1, upsample=(1, 2, 2), bn=True, act='tanh'):
        super(QRNNUpsampleConv3d, self).__init__(
            in_channels, hidden_channels, BasicUpsampleConv3d(in_channels, hidden_channels * 2, k, s, p, upsample, bn=bn), act=act)
