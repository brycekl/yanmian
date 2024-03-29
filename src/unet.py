from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.MultiResUnet import Respath


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        # 转置卷积特征图变大一倍，通道数-2，而双线性插值没有改变通道数
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )


class Fusion(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Fusion, self).__init__()
        self.conv = DoubleConv(in_channel*2, in_channel)
        self.up = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2)

    def forward(self, x1, x2):
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        x = self.up(x)
        return x


class FusionUp(nn.Module):
    def __init__(self, in_channel, out_channel, bilinear):
        super(FusionUp, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channel // 2 * 3, out_channel, in_channel // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channel, in_channel // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channel // 2 * 3, out_channel)

    def forward(self, x1, x2, x3):
        x1 = self.up(x1)
        x = torch.cat([x3, x2, x1], dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 num_classes_2: int = 0,
                 bilinear: bool = False,
                 base_c: int = 64,
                 model_name='unet'):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_classes_2 = num_classes_2
        self.bilinear = bilinear
        self.model_name = model_name

        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)

        # 两个decoder
        if model_name.find('up1') != -1 and num_classes_2 != 0:
            self.up4_2 = Up(base_c * 2, base_c, bilinear)
            print('creating up1 unet model')
        elif model_name.find('up2') != -1 and num_classes_2 != 0:
            self.up3_2 = Up(base_c * 4, base_c * 2 // factor, bilinear)
            self.up4_2 = Up(base_c * 2, base_c, bilinear)
            print('creating up2 unet model')
        elif model_name.find('up3') != -1 and num_classes_2 != 0:
            self.up2_2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
            self.up3_2 = Up(base_c * 4, base_c * 2 // factor, bilinear)
            self.up4_2 = Up(base_c * 2, base_c, bilinear)
            print('creating up3 unet model')
        elif model_name.find('up4') != -1 and num_classes_2 != 0:
            self.up1_2 = Up(base_c * 16, base_c * 8 // factor, bilinear)
            self.up2_2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
            self.up3_2 = Up(base_c * 4, base_c * 2 // factor, bilinear)
            self.up4_2 = Up(base_c * 2, base_c, bilinear)
            print('creating up4 unet model')
        if num_classes_2 != 0:
            self.out_conv2 = OutConv(base_c, num_classes_2)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        ux1 = self.up1(x5, x4)
        ux2 = self.up2(ux1, x3)
        ux3 = self.up3(ux2, x2)
        ux4 = self.up4(ux3, x1)
        logits = self.out_conv(ux4)
        if self.model_name.find('up1') != -1 and self.num_classes_2 != 0:
            up4_2 = self.up4_2(ux3, x1)
            logits_2 = self.out_conv2(up4_2)
        elif self.model_name.find('up2') != -1 and self.num_classes_2 != 0:
            ux3_2 = self.up3_2(ux2, x2)
            ux4_2 = self.up4_2(ux3_2, x1)
            logits_2 = self.out_conv2(ux4_2)
        elif self.model_name.find('up3') != -1 and self.num_classes_2 != 0:
            up2_2 = self.up2_2(ux1, x3)
            ux3_2 = self.up3_2(up2_2, x2)
            ux4_2 = self.up4_2(ux3_2, x1)
            logits_2 = self.out_conv2(ux4_2)
        elif self.model_name.find('up4') != -1 and self.num_classes_2 != 0:
            up1_2 = self.up1_2(x5, x4)
            up2_2 = self.up2_2(up1_2, x3)
            ux3_2 = self.up3_2(up2_2, x2)
            ux4_2 = self.up4_2(ux3_2, x1)
            logits_2 = self.out_conv2(ux4_2)
        elif self.num_classes_2 != 0:
            logits_2 = self.out_conv2(ux4)
        if self.num_classes_2 != 0:
            logits = torch.cat([logits, logits_2], dim=1)

        return logits


class UnetFusion(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 num_classes_2: int = 0,
                 bilinear: bool = False,
                 base_c: int = 64,
                 model_name='unet'):
        super(UnetFusion, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_classes_2 = num_classes_2
        self.bilinear = bilinear
        self.model_name = model_name

        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = FusionUp(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = FusionUp(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = FusionUp(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)

        self.up1_2 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2_2 = FusionUp(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3_2 = FusionUp(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4_2 = FusionUp(base_c * 2, base_c, bilinear)
        self.out_conv2 = OutConv(base_c, num_classes_2)

        # 特征融合模块
        self.fusion2 = Fusion(base_c * 8, base_c * 4 // factor)
        self.fusion3 = Fusion(base_c * 4, base_c * 2 // factor)
        self.fusion4 = Fusion(base_c * 2, base_c)

    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        ux1 = self.up1(x5, x4)
        ux1_2 = self.up1_2(x5, x4)
        fusion2 = self.fusion2(ux1, ux1_2)

        ux2 = self.up2(ux1, x3, fusion2)
        ux2_2 = self.up2_2(ux1_2, x3, fusion2)
        fusion3 = self.fusion3(ux2, ux2_2)

        ux3 = self.up3(ux2, x2, fusion3)
        ux3_2 = self.up3_2(ux2_2, x2, fusion3)
        fusion4 = self.fusion4(ux3, ux3_2)

        ux4 = self.up4(ux3, x1, fusion4)
        ux4_2 = self.up4_2(ux3_2, x1, fusion4)
        logits1 = self.out_conv(ux4)
        logits_2 = self.out_conv2(ux4_2)
        logits = torch.cat([logits1, logits_2], dim=1)

        return logits

