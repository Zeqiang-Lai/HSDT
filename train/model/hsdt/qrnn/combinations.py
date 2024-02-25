from functools import partial
import torch
import torch.nn as nn
from torch.nn import functional
from models.sync_batchnorm import SynchronizedBatchNorm2d, SynchronizedBatchNorm3d

BatchNorm3d = SynchronizedBatchNorm3d
from ..sepconv import SepConv3D_DW, SepConv3D_CA, SepConv3D_M


class UpsampleConv3d(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, upsample=None):
        super(UpsampleConv3d, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample_layer = torch.nn.Upsample(scale_factor=upsample, mode='trilinear', align_corners=True)

        self.conv3d = SepConv3D_M.of(nn.Conv3d)(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = self.upsample_layer(x_in)
        out = self.conv3d(x_in)
        return out


class BasicConv3d(nn.Sequential):
    def __init__(self, in_channels, channels, k=3, s=1, p=1, bias=False, bn=True):
        super(BasicConv3d, self).__init__()
        if bn:
            self.add_module('bn', BatchNorm3d(in_channels))
        self.add_module('conv', SepConv3D_M.of(nn.Conv3d)(in_channels, channels, k, s, p, bias=bias))

class BasicDownConv3d(nn.Sequential):
    def __init__(self, in_channels, channels, k=3, s=1, p=1, bias=False, bn=True):
        super(BasicDownConv3d, self).__init__()
        if bn:
            self.add_module('bn', BatchNorm3d(in_channels))
        self.add_module('conv', nn.Conv3d(in_channels, channels, k, s, p, bias=bias))


class BasicUpsampleConv3d(nn.Sequential):
    def __init__(self, in_channels, channels, k=3, s=1, p=1, upsample=(1, 2, 2), bn=True):
        super(BasicUpsampleConv3d, self).__init__()
        if bn:
            self.add_module('bn', BatchNorm3d(in_channels))
        self.add_module('upsample_conv', UpsampleConv3d(in_channels, channels, k, s, p, bias=False, upsample=upsample))
