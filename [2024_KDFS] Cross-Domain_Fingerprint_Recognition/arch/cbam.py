import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.reduction_ratio = reduction_ratio
        self.mlp = nn.Sequential(
            nn.Linear(gate_channels * len(pool_types), gate_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        pool_outputs = []

        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (height, width), stride=(height, width)).view(batch_size, channels, 1, 1)
                pool_outputs.append(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (height, width), stride=(height, width)).view(batch_size, channels, 1, 1)
                pool_outputs.append(max_pool)
            elif pool_type == 'stochastic':
                stochastic_pool, _ = torch.max(x.view(batch_size, channels, -1), dim=2, keepdim=True)
                pool_outputs.append(stochastic_pool)

        pool_outputs = torch.cat(pool_outputs, dim=1)

        mlp = nn.Sequential(
            nn.Linear(channels * 2, channels // self.reduction_ratio).to(x.device),
            nn.ReLU(inplace=True),
            nn.Linear(channels // self.reduction_ratio, channels).to(x.device)
        )
        channel_att_sum = mlp(pool_outputs.view(batch_size, -1))
        # channel_att_sum = self.mlp(pool_outputs.view(batch_size, -1))
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale
    
class StochasticPool2d(nn.Module):
    def __init__(self, kernel_size, stride):
        super(StochasticPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
    
    def forward(self, x):
        if self.training:
            out = F.max_pool2d(x, kernel_size=self.kernel_size, stride=self.stride)
            out_size = out.size()
            noise = torch.randn(out_size).to(x.device)
            out = out + noise
            return out
        else:
            return F.avg_pool2d(x, kernel_size=self.kernel_size, stride=self.stride)

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out