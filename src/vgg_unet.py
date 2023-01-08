from collections import OrderedDict
from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models import vgg16_bn

from .unet import Up, OutConv


class DM_Module(nn.Module):
    def __init__(self, in_channel, in_spatial, cha_ratio=1, spa_ratio=1, down_ratio=1):
        """
        input data dimensions: b, c, h, w
        output data dimensions: b, c, h, w
        Args:
            in_channel: the channel of input data (x)
            in_spatial: equal to 2*h*w
            cha_ratio: to reduce amount of computation, first reduce the number of channels of the input data from
            in_channel to in_chennel // cha_ratio
            spa_ratio: reduce the number of channels from 2*h*w to h*w//spa_ratio
            down_ratio: control the channel of sigmoid operate
        """
        super(DM_Module, self).__init__()

        self.in_channel = in_channel
        self.in_spatial = in_spatial

        self.inter_channel = in_channel // cha_ratio
        self.inter_spatial = in_spatial // spa_ratio

        self.sigmoid = nn.Sigmoid()
        self.is_skip = True
        self.gx_spatial = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.inter_channel),
            nn.ReLU()
        )

        self.gg_spatial = nn.Sequential(
            nn.Conv2d(in_channels=self.in_spatial * 2, out_channels=self.inter_spatial,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.inter_spatial),
            nn.ReLU()
        )

        num_channel_s = 1 + self.inter_spatial
        self.W_spatial = nn.Sequential(
            nn.Conv2d(in_channels=num_channel_s, out_channels=num_channel_s // down_ratio,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_channel_s // down_ratio),
            nn.ReLU(),
            nn.Conv2d(in_channels=num_channel_s // down_ratio, out_channels=1,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1)
        )

        self.theta_spatial = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.inter_channel),
            nn.ReLU()
        )
        self.phi_spatial = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.inter_channel),
            nn.ReLU()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        # compute the long_range dependency
        theta_xs = self.theta_spatial(x)
        phi_xs = self.phi_spatial(x)
        theta_xs = theta_xs.view(b, self.inter_channel, -1)
        theta_xs = theta_xs.permute(0, 2, 1)  # b, h*w, c
        phi_xs = phi_xs.view(b, self.inter_channel, -1)  # b, c, h*w

        Gs = torch.matmul(theta_xs, phi_xs)  # batch 维度上矩阵相乘， 输出 b, h*w, h*w
        Gs_in = Gs.permute(0, 2, 1).view(b, h * w, h, w)  # transpose and reshape
        Gs_out = Gs.view(b, h * w, h, w)
        Gs_joint = torch.cat((Gs_in, Gs_out), 1)  # b, 2*h*w, h, w

        Gs_joint = self.gg_spatial(Gs_joint)  # reduce the amount of computation, we reduce the channel 2*h*w

        # compute the short_range dependency
        g_xs = self.gx_spatial(x)
        g_xs = torch.mean(g_xs, dim=1, keepdim=True)  # b, 1, h, w

        # compute the final dependency result
        ys = torch.cat((g_xs, Gs_joint), 1)
        W_ys = self.W_spatial(ys)

        out = self.sigmoid(W_ys.expand_as(x)) * x  # sigmoid: 在channel方向卷积，输出维度为1
        return out


class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Args:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
    """
    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}

        # 重新构建backbone，将没有使用到的模块全部删掉
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class VGG16UNet(nn.Module):
    def __init__(self, num_classes, pretrain_backbone: bool = False):
        super(VGG16UNet, self).__init__()
        backbone = vgg16_bn(pretrained=pretrain_backbone)

        # if pretrain_backbone:
        #     # 载入vgg16_bn预训练权重
        #     # https://download.pytorch.org/models/vgg16_bn-6c64b313.pth
        #     backbone.load_state_dict(torch.load("vgg16_bn.pth", map_location='cpu'))

        backbone = backbone.features

        stage_indices = [5, 12, 22, 32, 42]
        self.stage_out_channels = [64, 128, 256, 512, 512]
        return_layers = dict([(str(j), f"stage{i}") for i, j in enumerate(stage_indices)])
        self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

        c = self.stage_out_channels[4] + self.stage_out_channels[3]
        self.up1 = Up(c, self.stage_out_channels[3])
        c = self.stage_out_channels[3] + self.stage_out_channels[2]
        self.up2 = Up(c, self.stage_out_channels[2])
        c = self.stage_out_channels[2] + self.stage_out_channels[1]
        self.up3 = Up(c, self.stage_out_channels[1])
        c = self.stage_out_channels[1] + self.stage_out_channels[0]
        self.up4 = Up(c, self.stage_out_channels[0])
        self.conv = OutConv(self.stage_out_channels[0], num_classes=num_classes)
        # self.dm_module = DM_Module(512, 16*16)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        backbone_out = self.backbone(x)
        # x = self.dm_module(x) + x
        x = self.up1(backbone_out['stage4'], backbone_out['stage3'])
        x = self.up2(x, backbone_out['stage2'])
        x = self.up3(x, backbone_out['stage1'])
        x = self.up4(x, backbone_out['stage0'])
        x = self.conv(x)

        return x
