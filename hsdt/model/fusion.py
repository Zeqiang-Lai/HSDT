import torch.nn as nn
import torch

# B,C,D,W,H


class AdaptiveFusion(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.conv_weight = nn.Sequential(
            nn.Conv3d(channel * 2, channel, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        w = self.conv_weight(torch.cat([x, y], dim=1))
        return (1 - w) * x + w * y


class ConcatFusion(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(channel * 2, channel, 3, 1, 1),
        )

    def forward(self, x, y):
        return self.conv(torch.cat([x, y], dim=1))


class SAdaptiveFusion(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.conv_weight = nn.Sequential(
            nn.Conv3d(channel * 2, channel, 1),
            nn.Tanh(),
            nn.Conv3d(channel, channel, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        w = self.conv_weight(torch.cat([x, y], dim=1))
        return (1 - w) * x + w * y


class SSAdaptiveFusion(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.conv_weight = nn.Sequential(
            nn.Conv3d(channel * 2, channel, 1),
            nn.Tanh(),
            nn.Conv3d(channel, channel, 1),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        w = self.conv_weight(torch.cat([x, y], dim=1))
        return (1 - w) * x + w * y


class ResidualAdaptiveFusion(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.conv_weight = nn.Sequential(
            nn.Conv3d(channel * 2, channel, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, value, augment):
        w = self.conv_weight(torch.cat([value, augment], dim=1))
        residual = (1 - w) * value + w * augment
        return value + augment + residual
