import os
import argparse
import numpy as np
import cv2
from pathlib import Path
import copy
import logging
import random
import scipy.io as sio
import time
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn.parameter import Parameter
import torch.nn.functional as F

class StegoScore(nn.Module):
    def __init__(self, trainable_filter=False):
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
            _, H, W = x.shape
            channels = 1
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


# class ChannelWiseStegoScore(nn.Module):
#     def __init__(self, trainable_filter=True):
#         super(ChannelWiseStegoScore, self).__init__()
#         self.trainable_filter = trainable_filter
#         if self.trainable_filter:
#             self.stego_list = nn.ModuleList([StegoScore(trainable_filter=True) for _ in range(3)])
#         else:
#             self.stego = StegoScore(trainable_filter=False)
    
#     def forward(self, x):
#         outputs = []
#         B, C, H, W = x.shape
#         if self.trainable_filter:
#             for c in range(C):
#                 x_c = x[:, c:c+1, :, :]
#                 x_c_score = self.stego_list[c](x_c)
#                 outputs.append(x_c_score)
#         else:
#             for c in range(C):
#                 x_c = x[:, c:c+1, :, :]
#                 x_c_score = self.stego(x_c)
#                 outputs.append(x_c_score)
#         out = torch.cat(outputs, dim=1)
#         return out
    
class Type1a(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(Type1a, self).__init__()
        self.inchannel=inchannel
        self.outchannel=outchannel
        self.relu = nn.ReLU()

        self.basic=nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=3, padding=1),
                nn.BatchNorm2d(outchannel),
                nn.ReLU(),
                
                nn.Conv2d(outchannel, outchannel, kernel_size=3, padding=1),
                nn.BatchNorm2d(outchannel),
                nn.ReLU()
                )

    def forward(self,x):
        out=self.basic(x)
        return out
      
      
class Type1b(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(Type1b, self).__init__()
        self.inchannel=inchannel
        self.outchannel=outchannel
        self.relu = nn.ReLU()

        self.basic=nn.Sequential(
                nn.Conv2d(inchannel, inchannel, kernel_size=3, padding=1),
                nn.BatchNorm2d(inchannel),
                nn.ReLU(),
                
                nn.Conv2d(inchannel, outchannel, kernel_size=3, padding=1),
                nn.BatchNorm2d(outchannel),
                nn.ReLU()
                )

    def forward(self,x):
        out=self.basic(x)

        return out
      
    
class Type2(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(Type2, self).__init__()
        self.inchannel=inchannel
        self.outchannel=outchannel
        self.relu = nn.ReLU()

        self.basic=nn.Sequential(
                nn.Conv2d(inchannel, inchannel, kernel_size=3, padding=1),
                nn.BatchNorm2d(inchannel),
                nn.ReLU(),
                
                nn.Conv2d(inchannel, outchannel, kernel_size=3, padding=1),
                nn.BatchNorm2d(outchannel),
            
                nn.AvgPool2d(kernel_size=3, padding=1, stride=2),
                )
        self.shortcut=nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=2),
                nn.BatchNorm2d(outchannel),
                )
    def forward(self,x):
        out=self.basic(x)
        out+=self.shortcut(x)
        out=self.relu(out)

        return out
        
        
class Type3(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(Type3, self).__init__()
        self.inchannel=inchannel
        self.outchannel=outchannel
        self.relu=nn.ReLU()

        self.basic=nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1),
                nn.BatchNorm2d(outchannel),
                nn.ReLU(),
                nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=2, groups=32, padding=1),
                nn.BatchNorm2d(outchannel),
                nn.ReLU(),
                nn.Conv2d(outchannel, outchannel, kernel_size=1),
                nn.BatchNorm2d(outchannel),
                )
        self.shortcut=nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(outchannel),
                )
    def forward(self,x):
        out=self.basic(x)
        out+=self.shortcut(x)
        out=self.relu(out)
        return out
    
        
class UCNet_WOW(nn.Module):
  def __init__(self, num_classes=2, trainable_filter=True):
    super(UCNet_WOW, self).__init__()
    
    self.pre = StegoScore(trainable_filter=trainable_filter)

    self.group1 = Type1a(3,32)
    self.group2 = Type2(32,32)
    self.group3 = Type3(32,64)
    self.group4 = Type2(64,128)
    self.group5 = Type1b(128,256)

    self.avg = nn.AvgPool2d(kernel_size=32, stride=1)

    self.fc1 = nn.Linear(1 * 1 * 256, num_classes)

  def forward(self, input):
    # seperate color channels
    output = input
    
    output_c1 = output[:, 0, :, :]
    output_c2 = output[:, 1, :, :] 
    output_c3 = output[:, 2, :, :] 
    out_c1 = output_c1.unsqueeze(1)
    out_c2 = output_c2.unsqueeze(1)
    out_c3 = output_c3.unsqueeze(1)
    c1 = self.pre(out_c1)
    c2 = self.pre(out_c2)
    c3 = self.pre(out_c3)
    
    output = torch.cat([c1, c2, c3], dim=1)
    output = self.group1(output)
    output = self.group2(output)
    output = self.group3(output)
    output = self.group4(output)
    output = self.group5(output)
    output = self.avg(output)
    output = output.view(output.size(0), -1)

    output = self.fc1(output)

    return output