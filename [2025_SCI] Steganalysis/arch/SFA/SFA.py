import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class StegoScore(nn.Module):
    def __init__(self, trainable_filter=True):
        super(StegoScore, self).__init__()

        self.hpdf = nn.Parameter(
            torch.tensor([
            -0.0544158422, 0.3128715909, -0.6756307363,
            0.5853546837, 0.0158291053, -0.2840155430,
            -0.0004724846, 0.1287474266, 0.0173693010,
            -0.0440882539, -0.0139810279, 0.0087460940,
            0.0048703530, -0.0003917404, -0.0006754494, -0.0001174768
        ], dtype=torch.float32),
        requires_grad=trainable_filter)


    def forward(self, x):
        # x: (batch_size, 3, H, W)
        if len(x.shape) == 4:
            _, channels, H, W = x.shape
        elif len(x.shape) == 3:
            channels, H, W = x.shape
        else:
            raise ValueError(f"Invalid shape of x: {x.shape}")
        
        # Define Daubechies 8 filters
        hpdf = self.hpdf
        lpdf = (-1) ** torch.arange(len(self.hpdf), device=x.device) * hpdf.flip(0)
        
        # Create 2D filters
        F_filters = []
        for i in range(3):
            if i == 0:
                # LH Filter(row: Low, col: High)
                F_filters.append(torch.ger(lpdf, hpdf))
            elif i == 1:
                # HL Filter(row: High, col: Low)
                F_filters.append(torch.ger(hpdf, lpdf))
            else:
                # HH Filter(row: High, col: High)
                F_filters.append(torch.ger(hpdf, hpdf))

        # Calculate padding size
        pad_size = max([f.shape[0] for f in F_filters]) -1
        
        # Apply padding
        x_padded = F.pad(x, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
        
        # Calculate Filter Results
        xi = []
        for F_filter in F_filters:
            F_filter = F_filter.unsqueeze(0).unsqueeze(0).to(x.device)  # (1, 1, k, k)
            F_filter_rot = F_filter.flip(-2, -1)
                
            R = F.conv2d(x_padded, F_filter.expand(channels, -1, -1, -1), groups=channels)
            abs_R = torch.abs(R)
            
            xi_i = F.conv2d(abs_R, torch.abs(F_filter_rot).expand(channels, -1, -1, -1), groups=channels)
            if F_filter.shape[-1] % 2 == 0:  # 짝수 크기 필터 (열 방향)
                xi_i = torch.roll(xi_i, shifts=1, dims=-1)
            if F_filter.shape[-2] % 2 == 0:  # 짝수 크기 필터 (행 방향)
                xi_i = torch.roll(xi_i, shifts=1, dims=-2)
            
            xi.append(xi_i)

        # Calculate rho(Harmonic Mean)
        xi_stack = torch.stack(xi, dim=0)  # (3, batch_size, channels, H, W)
        p = -1
        rho = (xi_stack ** p).sum(dim=0) ** (-1 / p)

        # Thresholding and NaN handling
        wetCost = 1e10
        rho = torch.clamp(rho, max=wetCost)
        rho[torch.isnan(rho)] = wetCost

        # Normalize rho
        min_rho = rho.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
        max_rho = rho.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        rho = (rho - min_rho) / (max_rho - min_rho + 1e-8)
        
        # Invert rho for attention
        score = 1 - rho
        score = F.sigmoid(score)
        
        return x * score

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
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
        
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale
    
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)
    
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

class SFA(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], attention_type='SFA', trainable_filter=True):
        super(SFA, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.attention_type = attention_type
        if self.attention_type == 'CBAM':
            self.SpatialGate = SpatialGate()
        elif self.attention_type == 'SFA':
            self.StegoScore = StegoScore(trainable_filter)

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if self.attention_type == 'CBAM':
            x_out = self.SpatialGate(x)
        elif self.attention_type == 'SFA':
            x_out = self.StegoScore(x)
        return x_out