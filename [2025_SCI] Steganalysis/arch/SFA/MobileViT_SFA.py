import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import create_model  # 수정된 임포트
import numpy as np
from .SFA import SFA
import os
from PIL import Image
import matplotlib.pyplot as plt

class MobileViT_SFA(nn.Module):
    def __init__(self, num_classes=2, apply_sfa=False, attention_type='add', 
                 rho_scaler=0.5, return_features=False, attention_after=None):
        """
        attention_after: stage1의 몇 번째 블럭 이후에 attention을 적용할지 지정하는 리스트.
                         예: [2,3,4] (기본값) -> stage1_2, stage1_3, stage1_4 이후에 attention 적용.
        """
        super(MobileViT_SFA, self).__init__()
        # timm.create_model 대신, timm.models.create_model을 사용
        model = create_model('mobilevit_s', num_classes=num_classes, pretrained=True)
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
        
        self.return_features = return_features
        self.apply_sfa = apply_sfa

        
        if self.apply_sfa:
            self.sfa = SFA()         
   
            self.attention_type = attention_type
            self.rho_scaler = nn.Parameter(torch.tensor(rho_scaler))
            self.attention_after = attention_after if attention_after is not None else [2, 3, 4]

            # stage1_1: x.shape =>  torch.Size([32, 64, 128, 128])
            # stage1_2: x.shape =>  torch.Size([32, 96, 64, 64])
            # stage1_3: x.shape =>  torch.Size([32, 128, 32, 32])
            # stage1_4: x.shape =>  torch.Size([32, 160, 16, 16])

            # 초기 세팅            
            # if 2 in self.attention_after:
            #     self.attention_proj_stage1_2 = nn.Conv2d(3, 96, kernel_size=1)
            # if 3 in self.attention_after:
            #     self.attention_proj_stage1_3 = nn.Conv2d(3, 128, kernel_size=1)
            # if 4 in self.attention_after:
            #     self.attention_proj_stage1_4 = nn.Conv2d(3, 160, kernel_size=1)
            
            if 2 in self.attention_after:
                self.attention_proj_stage1_2 = nn.Sequential(
                    nn.Conv2d(3, 96, kernel_size=1, stride=1),
                    nn.AdaptiveAvgPool2d((64, 64))
                )
            if 3 in self.attention_after:
                self.attention_proj_stage1_3 = nn.Sequential(
                    nn.Conv2d(3, 128, kernel_size=1, stride=1),
                    nn.AdaptiveAvgPool2d((32, 32))
                )
            if 4 in self.attention_after:
                self.attention_proj_stage1_4 = nn.Sequential(
                    nn.Conv2d(3, 160, kernel_size=1, stride=1),
                    nn.AdaptiveAvgPool2d((16, 16))
                )


    def forward(self, x):
        if self.return_features:
            feature_maps = []

        if self.apply_sfa:
            if self.attention_type == 'add':
                x_sfa = self.sfa.get_rho(x) * self.rho_scaler  # (batch_size, channels, H, W)
            elif self.attention_type == 'mul':
                x_sfa = 1 + (self.sfa.get_rho(x) * self.rho_scaler)  # (batch_size, channels, H, W)

        # Stage 0
        x = self.stage0(x)
        if self.return_features:
            feature_maps.append(x)
        
        # Stage 1_0
        x = self.stage1_0(x)
        if self.return_features:
            feature_maps.append(x)

        # Stage 1_1
        x = self.stage1_1(x)
        if self.return_features:
            feature_maps.append(x)

        # Stage 1_2
        x = self.stage1_2(x)
        if self.apply_sfa and (2 in self.attention_after):
            # x_sfa_resized = F.interpolate(x_sfa, size=(x.size(2), x.size(3)), 
            #                               mode='bilinear', align_corners=False)
            x = self.apply_attention(x, x_sfa, projector=self.attention_proj_stage1_2)
        if self.return_features:
            feature_maps.append(x)

        # Stage 1_3
        x = self.stage1_3(x)
        if self.apply_sfa and (3 in self.attention_after):
            # x_sfa_resized = F.interpolate(x_sfa, size=(x.size(2), x.size(3)), 
            #                               mode='bilinear', align_corners=False)
            x = self.apply_attention(x, x_sfa, projector=self.attention_proj_stage1_3)
        if self.return_features:
            feature_maps.append(x)

        # Stage 1_4
        x = self.stage1_4(x)
        if self.apply_sfa and (4 in self.attention_after):
            # x_sfa_resized = F.interpolate(x_sfa, size=(x.size(2), x.size(3)), 
            #                               mode='bilinear', align_corners=False)
            x = self.apply_attention(x, x_sfa, projector=self.attention_proj_stage1_4)
        if self.return_features:
            feature_maps.append(x)

        # Stage 2
        x = self.stage2(x)
        if self.return_features:
            feature_maps.append(x)

        # Stage 3 - Classification Head
        x = self.stage3(x)

        if self.return_features:
            return x, feature_maps
        else:
            return x

    def apply_attention(self, x, x_sfa, projector):
        x_sfa = projector(x_sfa)
        
        if self.attention_type == 'add':
            x = x + x_sfa
        elif self.attention_type == 'mul':
            x = x * (1+torch.sigmoid(x_sfa))
        else:
            raise ValueError(f"Invalid attention type: {self.attention_type}")
        return x
