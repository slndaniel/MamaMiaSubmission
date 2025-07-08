# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import torch.nn as nn

from monai.networks.blocks.convolutions import Convolution
from monai.networks.blocks.upsample import UpSample
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.utils import InterpolateMode, UpsampleMode


def get_conv_layer(
    spatial_dims: int, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, bias: bool = False
):
    return Convolution(
        spatial_dims, in_channels, out_channels, strides=stride, kernel_size=kernel_size, bias=bias, conv_only=True
    )


def get_upsample_layer(
    spatial_dims: int, in_channels: int, upsample_mode: UpsampleMode | str = "nontrainable", scale_factor: int = 2, size: tuple = (7, 24, 24), scale_use: bool = True
):
    if scale_use:
      return UpSample(
          spatial_dims=spatial_dims,
          in_channels=in_channels,
          out_channels=in_channels,
          scale_factor=scale_factor,
          mode=upsample_mode,
          interp_mode=InterpolateMode.LINEAR,
          align_corners=False,
      )
    else:
      return UpSample(
          spatial_dims=spatial_dims,
          in_channels=in_channels,
          out_channels=in_channels,
          size=size,
          mode=upsample_mode,
          interp_mode=InterpolateMode.LINEAR,
          align_corners=False,
      )


class ResBlock(nn.Module):
    """
    ResBlock employs skip connection and two convolution blocks and is used
    in SegResNet based on `3D MRI brain tumor segmentation using autoencoder regularization
    <https://arxiv.org/pdf/1810.11654.pdf>`_.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        norm: tuple | str,
        kernel_size: int = 3,
        act: tuple | str = ("RELU", {"inplace": True}),
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions, could be 1, 2 or 3.
            in_channels: number of input channels.
            norm: feature normalization type and arguments.
            kernel_size: convolution kernel size, the value should be an odd number. Defaults to 3.
            act: activation type and arguments. Defaults to ``RELU``.
        """

        super().__init__()

        if kernel_size % 2 != 1:
            raise AssertionError("kernel_size should be an odd number.")

        self.norm1 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.norm2 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.act = get_act_layer(act)
        self.conv1 = get_conv_layer(
            spatial_dims, in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size
        )
        self.conv2 = get_conv_layer(
            spatial_dims, in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size
        )

    def forward(self, x):
        identity = x

        x = self.norm1(x)
        x = self.act(x)
        x = self.conv1(x)

        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)

        x += identity

        return x
     
        
class LightweightResBlock(nn.Module):
    """
    ResBlock employs skip connection and two convolution blocks and is used
    in SegResNet based on `3D MRI brain tumor segmentation using autoencoder regularization
    <https://arxiv.org/pdf/1810.11654.pdf>`_.
    """
    def __init__(self, in_channels: int, expanded_channels: int):
            super().__init__()
    
            # 1󪻑 Conv to expand channels
            self.conv1 = nn.Conv3d(in_channels, expanded_channels, kernel_size=1, stride=1)
            self.norm1 = nn.InstanceNorm3d(expanded_channels)
            self.act1 = nn.LeakyReLU()
    
            # 3󫢫 Depthwise Conv with stride=2 (DWConv3D)
            self.dwconv = nn.Conv3d(
                expanded_channels,
                expanded_channels,
                kernel_size=3,
                padding=1,
                groups=expanded_channels  # Depthwise
            )
            self.norm2 = nn.InstanceNorm3d(expanded_channels)
            self.act2 = nn.LeakyReLU()
    
            # 1󪻑 Conv to reduce back to original channels
            self.conv2 = nn.Conv3d(expanded_channels, in_channels, kernel_size=1, stride=1)
            self.norm3 = nn.InstanceNorm3d(in_channels)
            self.act3 = nn.LeakyReLU()
    
    
    def forward(self, x):
        #print(f'x shape: {x.shape}')
        identity = x
        #print(f'identity shape: {identity.shape}')

        out = self.conv1(x)
        #print(f'out1 shape: {out.shape}')
        out = self.norm1(out)
        out = self.act1(out)
        #print(f'out2 shape: {out.shape}')

        out = self.dwconv(out)
        out = self.norm2(out)
        out = self.act2(out)
        #print(f'out3 shape: {out.shape}')

        out = self.conv2(out)
        out = self.norm3(out)
        out = self.act3(out)
        #print(f'out4 shape: {out.shape}')

        out += identity
        return out
        
import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention3D(nn.Module):
    def __init__(self, in_planes, reduction_ratio=16):
        super(ChannelAttention3D, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_planes, in_planes // reduction_ratio, bias=False),
            nn.ReLU(),
            nn.Linear(in_planes // reduction_ratio, in_planes, bias=False)
        )

    def forward(self, x):
        B, C, D, H, W = x.size()
        avg_pool = F.adaptive_avg_pool3d(x, 1).view(B, C)
        max_pool = F.adaptive_max_pool3d(x, 1).view(B, C)

        avg_out = self.mlp(avg_pool)
        max_out = self.mlp(max_pool)

        attention = torch.sigmoid(avg_out + max_out).view(B, C, 1, 1, 1)
        return x * attention.expand_as(x)

class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention3D, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        attention = torch.sigmoid(self.conv(concat))
        return x * attention

class CBAM3D(nn.Module):
    def __init__(self, channels, reduction_ratio=16, kernel_size=7):
        super(CBAM3D, self).__init__()
        self.channel_attention = ChannelAttention3D(channels, reduction_ratio)
        self.spatial_attention = SpatialAttention3D(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class AttentionResBlock3D(nn.Module):
    def __init__(self,
        spatial_dims: int,
        in_channels: int,
        norm: tuple | str,
        kernel_size: int = 3,
        act: tuple | str = ("RELU", {"inplace": True}),):
        super().__init__()
        self.resblock = ResBlock(spatial_dims=spatial_dims, in_channels=in_channels, norm=norm, act=act)
        self.cbam = CBAM3D(in_channels)

    def forward(self, x):
        x = self.resblock(x)
        x = self.cbam(x)
        return x
        
class AttentionLightweightResBlock3D(nn.Module):
    def __init__(self,
        spatial_dims: int,
        in_channels: int,
        norm: tuple | str,
        kernel_size: int = 3,
        act: tuple | str = ("RELU", {"inplace": True}),):
        super().__init__()
        self.resblock = LightweightResBlock(in_channels=in_channels, expanded_channels=2*in_channels)
        self.cbam = CBAM3D(in_channels)

    def forward(self, x):
        x = self.resblock(x)
        x = self.cbam(x)
        return x

class LightweightResBlock3D(nn.Module):
    def __init__(self,
        spatial_dims: int,
        in_channels: int,
        norm: tuple | str,
        kernel_size: int = 3,
        act: tuple | str = ("RELU", {"inplace": True}),):
        super().__init__()
        self.resblock = LightweightResBlock(in_channels=in_channels, expanded_channels=2*in_channels)

    def forward(self, x):
        x = self.resblock(x)
        return x
