from collections import OrderedDict

import torch
import torch.nn as nn
from sync_batchnorm import SynchronizedBatchNorm3d

from .qrnn3d import QRNNConv3D, BiQRNNConv3D, QRNNUpsampleConv3d,QRNNDownConv3D


def PlainConv(in_ch, out_ch):
    return QRNNConv3D(in_ch, out_ch)


def DownConv(in_ch, out_ch):
    return QRNNDownConv3D(in_ch, out_ch, 3, (1, 2, 2), 1)


def UpConv(in_ch, out_ch):
    return QRNNUpsampleConv3d(in_ch, out_ch)


class Encoder(nn.Module):
    def __init__(self, channels, num_half_layer, sample_idx):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_half_layer):
            if i not in sample_idx:
                encoder_layer = PlainConv(channels, channels)
            else:
                encoder_layer = DownConv(channels, 2 * channels)
                channels *= 2
            self.layers.append(encoder_layer)

    def forward(self, x, xs, reverse):
        num_half_layer = len(self.layers)
        for i in range(num_half_layer - 1):
            x = self.layers[i](x, reverse)
            reverse = not reverse
            xs.append(x)
        x = self.layers[-1](x, reverse)
        reverse = not reverse
        return x


class Decoder(nn.Module):
    def __init__(self, channels, num_half_layer, sample_idx, Fusion=None):
        super(Decoder, self).__init__()
        # Decoder
        self.layers = nn.ModuleList()
        self.enable_fusion = Fusion is not None

        if self.enable_fusion:
            self.fusions = nn.ModuleList()
            ch = channels
            for i in reversed(range(num_half_layer)):
                fusion_layer = Fusion(ch)
                if i in sample_idx:
                    ch //= 2
                self.fusions.append(fusion_layer)

        for i in reversed(range(num_half_layer)):
            if i not in sample_idx:
                decoder_layer = PlainConv(channels, channels)
            else:
                decoder_layer = UpConv(channels, channels // 2)
                channels //= 2
            self.layers.append(decoder_layer)

    def forward(self, x, xs, reverse):
        num_half_layer = len(self.layers)
        x = self.layers[0](x, reverse)
        reverse = not reverse
        for i in range(1, num_half_layer):
            if self.enable_fusion:
                x = self.fusions[i](x, xs.pop())
            else:
                x = x + xs.pop()
            x = self.layers[i](x, reverse)
            reverse = not reverse
        return x


class HSDTQRNN(nn.Module):
    def __init__(self, in_channels, channels, num_half_layer, sample_idx, Fusion=None):
        super(HSDTQRNN, self).__init__()
        self.head = PlainConv(in_channels, channels)
        self.encoder = Encoder(channels, num_half_layer, sample_idx)
        self.decoder = Decoder(channels * (2**len(sample_idx)), num_half_layer, sample_idx, Fusion=Fusion)
        self.tail = nn.Conv3d(channels, in_channels, 3, 1, 1, bias=True)

    def forward(self, x):
        xs = [x]
        out = self.head(xs[0], reverse=False)
        xs.append(out)
        reverse = True
        out = self.encoder(out, xs, reverse)
        out = self.decoder(out, xs, reverse)
        out = out + xs.pop()
        out = self.tail(out)
        out = out + xs.pop()
        return out
