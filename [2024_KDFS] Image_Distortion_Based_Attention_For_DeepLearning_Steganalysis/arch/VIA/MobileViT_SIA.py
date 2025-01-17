import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
from .SIA import StegoImpactAttention
import os
from PIL import Image
import matplotlib.pyplot as plt

class MobileViT_SIA(nn.Module):
    def __init__(self, num_classes=2, rho_scaler=0.5):
        super(MobileViT_SIA, self).__init__()
        model = timm.create_model('mobilevit_s', num_classes=num_classes, pretrained=True)
        modules = list(model.children())
        
        self.stage0 = modules[0]

        stage1_modules = list(modules[1])
        self.stage1_0 = stage1_modules[0]
        self.stage1_1 = stage1_modules[1]
        self.stage1_2 = stage1_modules[2]
        self.stage1_3 = stage1_modules[3]
        self.stage1_4 = stage1_modules[4]
        
        self.stage2 = modules[2]
        self.stage3 = modules[3]
        
        self.sia = StegoImpactAttention()
        
        self.rho_scaler = nn.Parameter(torch.tensor(rho_scaler))
        # When Apply SIA before MobileViT Block
        # self.attention_proj_stage1_0 = nn.Conv2d(3, 16, kernel_size=1)
        # self.attention_proj_stage1_1 = nn.Conv2d(3, 32, kernel_size=1)
        # self.attention_proj_stage1_2 = nn.Conv2d(3, 64, kernel_size=1)
        # self.attention_proj_stage1_3 = nn.Conv2d(3, 96, kernel_size=1)
        # self.attention_proj_stage1_4 = nn.Conv2d(3, 128, kernel_size=1)
        
        # When Apply SIA after MobileViT Block
        # self.attention_proj_stage0 = nn.Conv2d(3, 16, kernel_size=1)
        # self.attention_proj_stage1_0 = nn.Conv2d(3, 32, kernel_size=1)
        # self.attention_proj_stage1_1 = nn.Conv2d(3, 64, kernel_size=1)
        self.attention_proj_stage1_2 = nn.Conv2d(3, 96, kernel_size=1)
        self.attention_proj_stage1_3 = nn.Conv2d(3, 128, kernel_size=1)
        self.attention_proj_stage1_4 = nn.Conv2d(3, 160, kernel_size=1)

    def forward(self, x, apply_sia=False, return_features=False, attention_type='add', attn_block=None):
        # x Shape: (batch_size, 3, 512, 512)
        x_original = x.clone()  # 원본 이미지를 저장
        
        if return_features:
            feature_maps = []

        # Attention Map 생성
        if apply_sia:
            if attention_type == 'add':
                x_sia = self.sia.get_rho(x) * self.rho_scaler  # (batch_size, channels, H, W)
            elif attention_type == 'mul':
                x_sia = 1+(self.sia.get_rho(x) * self.rho_scaler)  # (batch_size, channels, H, W)

        # Stage 0
        x = self.stage0(x)
        # if return_features:
        #     feature_maps.append(x)
        
        # Stage 1_0
        x = self.stage1_0(x)
        # if return_features:
        #     feature_maps.append(x)

        # Stage 1_1
        x = self.stage1_1(x)
        
        # if return_features:
        #     feature_maps.append(x)

        # Stage 1_2
        # if apply_sia and 2 in attn_block:
            # print('=======================Stage 1_2=======================')
            # print(f'Before Attention: mean={x.mean().item():.4f}, std={x.std().item():.4f}, min={x.min().item():.4f}, max={x.max().item():.4f}')
            # x_sia = F.interpolate(x_sia, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
            # x = self.apply_attention(x, x_sia, projector=self.attention_proj_stage1_2, attention_type=attention_type)
            # print(f'After Attention: mean={x.mean().item():.4f}, std={x.std().item():.4f}, min={x.min().item():.4f}, max={x.max().item():.4f}')
            # print(f'SIA: mean={x_sia.mean().item():.4f}, std={x_sia.std().item():.4f}, min={x_sia.min().item():.4f}, max={x_sia.max().item():.4f}')
        x = self.stage1_2(x)
        if apply_sia and 2 in attn_block:
            # print('=======================Stage 1_2=======================')
            # print(f'Before Attention: mean={x.mean().item()}, std={x.std().item()}, min={x.min().item()}, max={x.max().item()}')
            x_sia = F.interpolate(x_sia, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
            x = self.apply_attention(x, x_sia, projector=self.attention_proj_stage1_2, attention_type=attention_type)
            # print(f'After Attention: mean={x.mean().item()}, std={x.std().item()}, min={x.min().item()}, max={x.max().item()}')
            # print(f'SIA: mean={x_sia.mean().item()}, std={x_sia.std().item()}, min={x_sia.min().item()}, max={x_sia.max().item()}')
        # if return_features:
        #     feature_maps.append(x)

        # Stage 1_3
        # if apply_sia:
            # print('=======================Stage 1_3=======================')
            # print(f'Before Attention: mean={x.mean().item():.4f}, std={x.std().item():.4f}, min={x.min().item():.4f}, max={x.max().item():.4f}')
            # x_sia = F.interpolate(x_sia, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
            # x = self.apply_attention(x, x_sia, projector=self.attention_proj_stage1_3, attention_type=attention_type)
            # print(f'After Attention: mean={x.mean().item():.4f}, std={x.std().item():.4f}, min={x.min().item():.4f}, max={x.max().item():.4f}')
            # print(f'SIA: mean={x_sia.mean().item():.4f}, std={x_sia.std().item():.4f}, min={x_sia.min().item():.4f}, max={x_sia.max().item():.4f}')
        x = self.stage1_3(x)
        if apply_sia and 3 in attn_block:
            # print('=======================Stage 1_3=======================')
            # print(f'Before Attention: mean={x.mean().item()}, std={x.std().item()}, min={x.min().item()}, max={x.max().item()}')
            x_sia = F.interpolate(x_sia, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
            x = self.apply_attention(x, x_sia, projector=self.attention_proj_stage1_3, attention_type=attention_type)
            # print(f'After Attention: mean={x.mean().item()}, std={x.std().item()}, min={x.min().item()}, max={x.max().item()}')
            # print(f'SIA: mean={x_sia.mean().item()}, std={x_sia.std().item()}, min={x_sia.min().item()}, max={x_sia.max().item()}')
        # if return_features:
        #     feature_maps.append(x)

        # Stage 1_4
        # if apply_sia:
            # print('=======================Stage 1_4=======================')
            # print(f'Before Attention: mean={x.mean().item():.4f}, std={x.std().item():.4f}, min={x.min().item():.4f}, max={x.max().item():.4f}')
            # x_sia = F.interpolate(x_sia, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
            # x = self.apply_attention(x, x_sia, projector=self.attention_proj_stage1_4, attention_type=attention_type)
            # print(f'After Attention: mean={x.mean().item():.4f}, std={x.std().item():.4f}, min={x.min().item():.4f}, max={x.max().item():.4f}')
            # print(f'SIA: mean={x_sia.mean().item():.4f}, std={x_sia.std().item():.4f}, min={x_sia.min().item():.4f}, max={x_sia.max().item():.4f}')
        x = self.stage1_4(x)
        if apply_sia and 4 in attn_block:
            # print('=======================Stage 1_4=======================')
            # print(f'Before Attention: mean={x.mean().item()}, std={x.std().item()}, min={x.min().item()}, max={x.max().item()}')
            x_sia = F.interpolate(x_sia, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
            x = self.apply_attention(x, x_sia, projector=self.attention_proj_stage1_4, attention_type=attention_type)
            # print(f'After Attention: mean={x.mean().item()}, std={x.std().item()}, min={x.min().item()}, max={x.max().item()}')
            # print(f'SIA: mean={x_sia.mean().item()}, std={x_sia.std().item()}, min={x_sia.min().item()}, max={x_sia.max().item()}')
        # if return_features:
        #     feature_maps.append(x)

        # Stage 2
        x = self.stage2(x)
        if return_features:
            feature_maps.append(x)

        # Stage 3 - Classification Head
        x = self.stage3(x)

        if return_features:
            return x, feature_maps[0]
        else:
            return x

    def apply_attention(self, x, x_sia, projector, attention_type='add'):
        # x_sia_resized를 x의 채널에 맞게 투영
        x_sia = projector(x_sia)
        
        if attention_type == 'add':
            x = x + x_sia
        elif attention_type == 'mul':
            x = x * x_sia
        else:
            raise ValueError(f"Invalid attention type: {attention_type}")
        return x