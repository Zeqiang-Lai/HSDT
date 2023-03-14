import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


class SSA(nn.Module):
    """ Spectral Self Attention (SSA) in "Hybrid Spectral Denoising Transformer with Learnable Query" 
        GSSA without gudiance.
    """
    def __init__(self, channel, num_bands, flex=False):
        super().__init__()
        self.channel = channel
        self.num_bands = num_bands

        self.value_proj = nn.Linear(channel, channel, bias=False)
        self.fc = nn.Linear(channel, channel, bias=False)

    def forward(self, x):
        B, C, D, H, W = x.shape

        residual = x

        tmp = x.reshape(B, C, D, H * W).mean(-1).permute(0, 2, 1)
        attn = tmp @ tmp.transpose(1, 2)

        attn = attn.reshape(B, self.num_bands, self.num_bands)
        attn = F.softmax(attn, dim=-1)  # B, band, band
        attn = attn.unsqueeze(1).unsqueeze(1)

        v = self.value_proj(rearrange(x, 'b c d h w -> b h w d c'))

        q = torch.matmul(attn, v)

        q = self.fc(q)
        q = rearrange(q, 'b h w d c -> b c d h w')

        q += residual

        return q, attn


class GSSA(nn.Module):
    """ Guided Spectral Self Attention (GSSA) in "Hybrid Spectral Denoising Transformer with Learnable Query" 
    """
    
    def __init__(self, channel, num_bands, flex=False):
        super().__init__()
        self.channel = channel
        self.num_bands = num_bands
        self.flex = flex

        # learnable query
        self.attn_proj = nn.Linear(channel, num_bands)  
        self.value_proj = nn.Linear(channel, channel, bias=False)
        self.fc = nn.Linear(channel, channel, bias=False)

    def forward(self, x):
        B, C, D, H, W = x.shape

        residual = x

        tmp = x.reshape(B, C, D, H * W).mean(-1).permute(0, 2, 1)

        if self.training:
            if random.random() > 0.5:
                attn = tmp @ tmp.transpose(1, 2)
            else:
                attn = self.attn_proj(tmp)
        else:
            if self.flex:
                attn = tmp @ tmp.transpose(1, 2)
            else:
                attn = self.attn_proj(tmp)

        attn = attn.reshape(B, self.num_bands, self.num_bands)
        attn = F.softmax(attn, dim=-1)  # B, band, band
        attn = attn.unsqueeze(1).unsqueeze(1)

        v = self.value_proj(rearrange(x, 'b c d h w -> b h w d c'))

        q = torch.matmul(attn, v)

        q = self.fc(q)
        q = rearrange(q, 'b h w d c -> b c d h w')

        q += residual

        return q, attn


class PixelwiseGSSA(GSSA):
    """ Pixelwise GSSA 
    """
    def __init__(self, channel, num_bands):
        super().__init__(channel, num_bands)

    def forward(self, x):
        B, C, D, H, W = x.shape
        x = rearrange(x, 'b c d h w -> b (h w) d c')

        residual = x

        if self.training:
            if random.random() > 0.5:
                attn = x @ x.transpose(-1, -2)
            else:
                attn = self.attn_proj(x)
        else:
            attn = x @ x.transpose(-1, -2)

        attn = F.softmax(attn, dim=-1)

        v = self.value_proj(x)
        q = torch.matmul(attn, v)

        q = self.fc(q)
        q += residual

        q = rearrange(q, 'b (h w) d c -> b c d h w', d=D, h=H, w=W)

        return q, attn


class GSSAConvImpl(GSSA):
    """ GSSA fast convolutional implementation
    """
    def __init__(self, channel, num_bands, flex=False):
        super().__init__(channel, num_bands)

    def forward(self, x):
        B, C, D, H, W = x.shape

        x = rearrange(x, 'b c d h w -> b h w d c')
        residual = x

        tmp = rearrange(x, 'b h w d c -> b (h w) d c').mean(1)
        attn = self.attn_proj(tmp)  # B,band,band
        attn = F.softmax(attn, dim=-1)  # b,band,band

        v = self.value_proj(x)  # b c band w h

        if B > 1:
            input = rearrange(v, 'b h w d c -> (b d) c h w').unsqueeze(0)
            weight = attn.reshape(B * D, D, 1, 1, 1)
            q = F.conv3d(input, weight, groups=B)  # 1, b*d, c, w, h
            q = rearrange(q.squeeze(0), '(b d) c h w -> b h w d c', b=B)
        else:
            input = rearrange(v, 'b h w d c -> c (b d) h w')
            weight = attn.reshape(D, D, 1, 1)
            q = F.conv2d(input, weight)  # 1, b*d, c, w, h
            q = rearrange(q, 'c (b d) w h -> b h w d c', b=B)

        q = self.fc(q)
        q += residual

        q = rearrange(q, 'b h w d c -> b c d h w')
        return q, attn


""" Feedforwrd
"""


class SMFFN(nn.Module):
    """ Self Modulated Feed Forward Network (SM-FFN) in "Hybrid Spectral Denoising Transformer with Learnable Query" 
    """
    
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


class SMFFNBranch(nn.Module):
    def __init__(self, d_model, d_ff, bias=False):
        super().__init__()
        self.w_3 = nn.Linear(d_model, d_ff, bias=bias)

    def forward(self, input):
        x = self.w_3(input)
        x, w = torch.chunk(x, 2, dim=-1)
        x2 = x * torch.sigmoid(w)
        return x2


class GDFN(nn.Module):
    """ 3D version of GDFN from Restormer.
    """
    def __init__(self, d_model, d_ff, bias=False):
        super(GDFN, self).__init__()
        self.project_in = nn.Conv3d(d_model, d_ff, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv3d(d_ff, d_ff, kernel_size=3, stride=1, padding=1, groups=d_ff, bias=bias)
        self.project_out = nn.Conv3d(d_model, d_model, kernel_size=1, bias=bias)

    def forward(self, input):
        input = rearrange(input, 'b d h w c -> b c d h w')
        x = self.project_in(input)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        output = x
        output = rearrange(output, 'b c d h w -> b d h w c')
        return output


class FFN(nn.Module):
    def __init__(self, d_model, d_ff, bias=False):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff, bias=bias)
        self.w_2 = nn.Linear(d_ff, d_model, bias=bias)
        self.act = nn.GELU()

    def forward(self, input):
        output= self.w_2(self.act(self.w_1(input)))
        return output


""" Transformer Block
"""


class TransformerBlock(nn.Module):
    def __init__(self, channels, num_bands=31, bias=False, flex=False):
        super().__init__()
        self.channels = channels
        self.attn = GSSA(channels, num_bands, flex=flex)
        self.ffn = SMFFN(channels, channels * 2, bias=bias)

    def forward(self, inputs):
        r, _ = self.attn(inputs)
        r = rearrange(r, 'b c d h w -> b d h w c')
        r = self.ffn(r)
        r = rearrange(r, 'b d h w c -> b c d h w')
        return r


class DummyTransformerBlock(nn.Module):
    def __init__(self, channels, num_bands=31, bias=False, flex=False):
        super().__init__()
        self.channels = channels

    def forward(self, inputs):
        return inputs.tanh()


""" Ablation
"""


class PixelwiseTransformerBlock(TransformerBlock):
    def __init__(self, channels, num_bands=31, bias=False, flex=False):
        super().__init__(channels, num_bands, bias, flex)
        self.channels = channels
        self.attn = PixelwiseGSSA(channels, num_bands, flex=flex)
        self.ffn = SMFFN(channels, channels * 2, bias=bias)


class FFNTransformerBlock(TransformerBlock):
    def __init__(self, channels, num_bands=31, bias=False, flex=False):
        super().__init__(channels, num_bands, bias, flex)
        self.channels = channels
        self.attn = GSSA(channels, num_bands, flex=flex)
        self.ffn = FFN(channels, channels * 2, bias=bias)


class GDFNTransformerBlock(TransformerBlock):
    def __init__(self, channels, num_bands=31, bias=False, flex=False):
        super().__init__(channels, num_bands, bias, flex)
        self.channels = channels
        self.attn = GSSA(channels, num_bands, flex=flex)
        self.ffn = GDFN(channels, channels * 2, bias=bias)


class GFNTransformerBlock(TransformerBlock):
    def __init__(self, channels, num_bands=31, bias=False, flex=False):
        super().__init__(channels, num_bands, bias, flex)
        self.channels = channels
        self.attn = GSSA(channels, num_bands, flex=flex)
        self.ffn = SMFFNBranch(channels, channels * 2, bias=bias)


class SSATransformerBlock(TransformerBlock):
    def __init__(self, channels, num_bands=31, bias=False, flex=False):
        super().__init__(channels, num_bands, bias, flex)
        self.channels = channels
        self.attn = SSA(channels, num_bands, flex=flex)
        self.ffn = SMFFN(channels, channels * 2, bias=bias)


class SSAFFNTransformerBlock(TransformerBlock):
    def __init__(self, channels, num_bands=31, bias=False, flex=False):
        super().__init__(channels, num_bands, bias, flex)
        self.channels = channels
        self.attn = SSA(channels, num_bands, flex=flex)
        self.ffn = FFN(channels, channels * 2, bias=bias)
