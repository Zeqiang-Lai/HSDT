from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import TransformerBlock
from .sepconv import SepConv_DP, SepConv_DP_CA, S3Conv

BatchNorm3d = nn.BatchNorm3d
Conv3d = S3Conv.of(nn.Conv3d)
TransformerBlock = TransformerBlock
IsConvImpl = False
UseBN = True


def PlainConv(in_ch, out_ch):
    return nn.Sequential(OrderedDict([
        ('conv', Conv3d(in_ch, out_ch, 3, 1, 1, bias=False)),
        ('bn', BatchNorm3d(out_ch) if UseBN else nn.Identity()),
        ('attn', TransformerBlock(out_ch, bias=True))
    ]))


def DownConv(in_ch, out_ch):
    return nn.Sequential(OrderedDict([
        ('conv', nn. Conv3d(in_ch, out_ch, 3, (1, 2, 2), 1, bias=False)),
        ('bn', BatchNorm3d(out_ch)if UseBN else nn.Identity()),
        ('attn', TransformerBlock(out_ch, bias=True))
    ]))


def UpConv(in_ch, out_ch):
    return nn.Sequential(OrderedDict([
        ('up', nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=True)),
        ('conv', nn.Conv3d(in_ch, out_ch, 3, 1, 1, bias=False)),
        ('bn', BatchNorm3d(out_ch) if UseBN else nn.Identity()),
        ('attn', TransformerBlock(out_ch, bias=True))
    ]))


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

    def forward(self, x, xs):
        num_half_layer = len(self.layers)
        for i in range(num_half_layer - 1):
            x = self.layers[i](x)
            xs.append(x)
        x = self.layers[-1](x)
        return x


class Decoder(nn.Module):
    count = 1
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

    def forward(self, x, xs):
        num_half_layer = len(self.layers)
        x = self.layers[0](x)
        for i in range(1, num_half_layer):
            if self.enable_fusion:
                x = self.fusions[i](x, xs.pop())
            else:
                x = x + xs.pop()
            x = self.layers[i](x)
        return x


class HSDT(nn.Module):
    def __init__(self, in_channels, channels, num_half_layer, sample_idx, Fusion=None):
        super(HSDT, self).__init__()
        self.head = PlainConv(in_channels, channels)
        self.encoder = Encoder(channels, num_half_layer, sample_idx)
        self.decoder = Decoder(channels * (2**len(sample_idx)), num_half_layer, sample_idx, Fusion=Fusion)
        self.tail = nn.Conv3d(channels, 1, 3, 1, 1, bias=True)

    def forward(self, x):
        xs = [x]
        out = self.head(xs[0])
        xs.append(out)
        out = self.encoder(out, xs)
        out = self.decoder(out, xs)
        out = out + xs.pop()
        out = self.tail(out)
        out = out + xs.pop()[:, 0:1, :, :, :]
        return out

    def load_state_dict(self, state_dict, strict: bool = True):
        if IsConvImpl:
            new_state_dict = {}
            for k, v in state_dict.items():
                if ('attn.attn' in k) and 'weight' in k and 'attn_proj' not in k:
                    new_state_dict[k] = v.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                else:
                    new_state_dict[k] = v
            state_dict = new_state_dict
        return super().load_state_dict(state_dict, strict)


class HSDTSSR(nn.Module):
    def __init__(self, in_channels, channels, num_half_layer, sample_idx, Fusion=None):
        super(HSDTSSR, self).__init__()
        self.proj = nn.Conv2d(3, 31, 1, bias=False)
        self.head = PlainConv(in_channels, channels)
        self.encoder = Encoder(channels, num_half_layer, sample_idx)
        self.decoder = Decoder(channels * (2**len(sample_idx)), num_half_layer, sample_idx, Fusion=Fusion)
        self.tail = nn.Conv3d(channels, 1, 3, 1, 1, bias=True)

    def forward(self, x):
        if self.training:
            return self.forward_train(x)
        return self.forward_test(x)

    def forward_train(self, x):
        x = F.leaky_relu(self.proj(x)).unsqueeze(1)
        xs = [x]
        out = self.head(xs[0])
        xs.append(out)
        out = self.encoder(out, xs)
        out = self.decoder(out, xs)
        out = out + xs.pop()
        out = self.tail(out)
        out = out + xs.pop()[:, 0:1, :, :, :]
        out = out.squeeze(1)
        return out

    def forward_test(self, x):
        pad_x, H, W = pad_mod(x, 8)
        output = self.forward_train(pad_x)[..., :H, :W]
        return output

    def load_state_dict(self, state_dict, strict: bool = True):
        if IsConvImpl:
            new_state_dict = {}
            for k, v in state_dict.items():
                if ('attn.attn' in k) and 'weight' in k and 'attn_proj' not in k:
                    new_state_dict[k] = v.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                else:
                    new_state_dict[k] = v
            state_dict = new_state_dict
        return super().load_state_dict(state_dict, strict)


def pad_mod(x, mod):
    h, w = x.shape[-2:]
    h_out = (h // mod + 1) * mod
    w_out = (w // mod + 1) * mod
    out = torch.zeros(*x.shape[:-2], h_out, w_out).type_as(x)
    out[..., :h, :w] = x
    return out.to(x.device), h, w
