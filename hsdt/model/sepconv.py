from functools import partial
import torch.nn as nn
import torch


def repeat(x):
    if isinstance(x, (tuple, list)):
        return x
    return [x] * 3


class SepConv_PD(nn.Module):
    def __init__(self, BASECONV, in_ch, out_ch, k, s=1, p=1, bias=False):
        super().__init__()
        k, s, p = repeat(k), repeat(s), repeat(p)

        padding_mode = 'zeros'
        self.pw_conv = BASECONV(in_ch, out_ch, (k[0], 1, 1), (s[0], 1, 1), (p[0], 0, 0), bias=bias, padding_mode=padding_mode)
        self.dw_conv = BASECONV(out_ch, out_ch, (1, k[1], k[2]), (1, s[1], s[2]), (0, p[1], p[2]), bias=bias, padding_mode=padding_mode)

    def forward(self, x):
        x = self.pw_conv(x)
        x = self.dw_conv(x)
        return x

    @staticmethod
    def of(base_conv):
        return partial(SepConv_PD, base_conv)


class SepConv_DP(nn.Module):
    def __init__(self, BASECONV, in_ch, out_ch, k, s=1, p=1, bias=False):
        super().__init__()
        k, s, p = repeat(k), repeat(s), repeat(p)

        padding_mode = 'zeros'
        self.dw_conv = BASECONV(in_ch, out_ch, (1, k[1], k[2]), (1, s[1], s[2]), (0, p[1], p[2]), bias=bias, padding_mode=padding_mode)
        self.pw_conv = BASECONV(out_ch, out_ch, (k[0], 1, 1), (s[0], 1, 1), (p[0], 0, 0), bias=bias, padding_mode=padding_mode)

    def forward(self, x):
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x

    @staticmethod
    def of(base_conv):
        return partial(SepConv_DP, base_conv)


class SepConv_DP_CA(nn.Module):
    def __init__(self, BASECONV, in_ch, out_ch, k, s=1, p=1, bias=False):
        super().__init__()
        k, s, p = repeat(k), repeat(s), repeat(p)

        padding_mode = 'zeros'
        self.dw_conv = BASECONV(in_ch, out_ch, (1, k[1], k[2]), (1, s[1], s[2]), (0, p[1], p[2]), bias=bias, padding_mode=padding_mode)
        self.pw_conv = BASECONV(out_ch, out_ch * 2, (k[0], 1, 1), (s[0], 1, 1), (p[0], 0, 0), bias=bias, padding_mode=padding_mode)

    def forward(self, x):
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        x, w = torch.chunk(x, 2, dim=1)
        x = x * w.sigmoid()
        return x

    @staticmethod
    def of(base_conv):
        return partial(SepConv_DP_CA, base_conv)


class S3Conv(nn.Module):
    # deep wise then point wise
    def __init__(self, BASECONV, in_ch, out_ch, k, s=1, p=1, bias=False):
        super().__init__()
        k, s, p = repeat(k), repeat(s), repeat(p)

        padding_mode = 'zeros'
        self.dw_conv = nn.Sequential(
            BASECONV(in_ch, out_ch, (1, k[1], k[2]), (1, s[1], s[2]), (0, p[1], p[2]), bias=bias, padding_mode=padding_mode),
            nn.LeakyReLU(),
            BASECONV(out_ch, out_ch, (1, k[1], k[2]), 1, (0, p[1], p[2]), bias=bias, padding_mode=padding_mode),
            nn.LeakyReLU(),
        )
        self.pw_conv = BASECONV(in_ch, out_ch, (k[0], 1, 1), (s[0], 1, 1), (p[0], 0, 0), bias=bias, padding_mode=padding_mode)

    def forward(self, x):
        x1 = self.dw_conv(x)
        x2 = self.pw_conv(x)
        return x1 + x2

    @staticmethod
    def of(base_conv):
        return partial(S3Conv, base_conv)


class S3Conv_Seq(nn.Module):
    def __init__(self, BASECONV, in_ch, out_ch, k, s=1, p=1, bias=False):
        super().__init__()
        k, s, p = repeat(k), repeat(s), repeat(p)

        padding_mode = 'zeros'
        self.dw_conv = nn.Sequential(
            BASECONV(in_ch, out_ch, (1, k[1], k[2]), (1, s[1], s[2]), (0, p[1], p[2]), bias=bias, padding_mode=padding_mode),
            nn.LeakyReLU(),
            BASECONV(out_ch, out_ch, (1, k[1], k[2]), 1, (0, p[1], p[2]), bias=bias, padding_mode=padding_mode),
            nn.LeakyReLU(),
        )
        self.pw_conv = BASECONV(out_ch, out_ch, (k[0], 1, 1), (s[0], 1, 1), (p[0], 0, 0), bias=bias, padding_mode=padding_mode)

    def forward(self, x):
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x

    @staticmethod
    def of(base_conv):
        return partial(S3Conv_Seq, base_conv)


class S3Conv1(nn.Module):
    def __init__(self, BASECONV, in_ch, out_ch, k, s=1, p=1, bias=False):
        super().__init__()
        k, s, p = repeat(k), repeat(s), repeat(p)

        padding_mode = 'zeros'
        self.dw_conv = nn.Sequential(
            BASECONV(in_ch, out_ch, (1, k[1], k[2]), (1, s[1], s[2]), (0, p[1], p[2]), bias=bias, padding_mode=padding_mode),
            nn.LeakyReLU(),
        )
        self.pw_conv = BASECONV(in_ch, out_ch, (k[0], 1, 1), (s[0], 1, 1), (p[0], 0, 0), bias=bias, padding_mode=padding_mode)

    def forward(self, x):
        x1 = self.dw_conv(x)
        x2 = self.pw_conv(x)
        return x1 + x2

    @staticmethod
    def of(base_conv):
        return partial(S3Conv1, base_conv)
