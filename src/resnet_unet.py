import torch
import torch.nn.functional as F
from torch import nn


# 因为ResNet34包含重复的单元，故用ResidualBlock类来简化代码
class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.basic = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1,
                      bias=False),  # 要采样的话在这里改变stride
            nn.BatchNorm2d(outchannel),  # 批处理正则化
            nn.ReLU(inplace=True),  # 激活
            nn.Conv2d(outchannel, outchannel, 3, 1, 1,
                      bias=False),  # 采样之后注意保持feature map的大小不变
            nn.BatchNorm2d(outchannel),
        )
        self.shortcut = shortcut

    def forward(self, x):
        out = self.basic(x)
        residual = x if self.shortcut is None else self.shortcut(x)  # 计算残差
        out += residual
        return nn.ReLU(inplace=True)(out)  # 注意激活


class Conv2dReLU(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
    ):
        super(Conv2dReLU, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return x


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)

        x = self.conv1(x)
        x = self.conv2(x)

        return x


class SegmentationHead(nn.Sequential):
    def __init__(self,
                 in_channels=16,
                 out_channels=1,
                 kernel_size=3,
                 upsampling=1):
        conv2d = nn.Conv2d(in_channels,
                           out_channels,
                           kernel_size=kernel_size,
                           padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(
            scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


# ResNet类
class Resnet34(nn.Module):
    def __init__(self, inchannels, out_c):
        super(Resnet34, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(inchannels, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
        )  # 开始的部分
        self.body = self.makelayers([3, 4, 6, 3])  # 具有重复模块的部分
        in_channels = [512, 256, 128, 128, 32]
        skip_channels = [256, 128, 64, 0, 0]
        out_channels = [256, 128, 64, 32, 16]
        blocks = [
            DecoderBlock(in_ch, skip_ch,
                         out_ch) for in_ch, skip_ch, out_ch in zip(
                in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)
        self.seg = SegmentationHead(16, out_c)

    def makelayers(self, blocklist):  # 注意传入列表而不是解列表
        self.layers = []
        for index, blocknum in enumerate(blocklist):
            if index != 0:
                shortcut = nn.Sequential(
                    nn.Conv2d(64 * 2 ** (index - 1),
                              64 * 2 ** index,
                              1,
                              2,
                              bias=False),
                    nn.BatchNorm2d(64 * 2 ** index))  # 使得输入输出通道数调整为一致
                self.layers.append(
                    ResidualBlock(64 * 2 ** (index - 1), 64 * 2 ** index, 2,
                                  shortcut))  # 每次变化通道数时进行下采样
            for i in range(0 if index == 0 else 1, blocknum):
                self.layers.append(
                    ResidualBlock(64 * 2 ** index, 64 * 2 ** index, 1))
        return nn.Sequential(*self.layers)

    def forward(self, x):
        self.features = []
        # 下采样
        # x = self.pre(x)
        for i, l in enumerate(self.pre):
            x = l(x)
            if i == 2:
                self.features.append(x)

        # print("y=", len(self.features))
        for i, l in enumerate(self.body):
            if i == 3 or i == 7 or i == 13:
                self.features.append(x)
            x = l(x)
        skips = self.features[::-1]

        # skips = self.features[1:]

        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        x = self.seg(x)
        return x
