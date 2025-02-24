import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import create_model  # 수정된 임포트
import numpy as np
from .SFA import SFA
import os
from PIL import Image
import matplotlib.pyplot as plt

class MobileViT_CBAM_based_SFA(nn.Module):
    def __init__(self, num_classes=2, apply_sfa=False, return_features=False, attention_after=[2,3,4], residual_connection=True, trainable_filter=True):
        """
        attention_after: stage1의 몇 번째 블럭 이후에 attention을 적용할지 지정하는 리스트.
                         예: [2,3,4] (기본값) -> stage1_2, stage1_3, stage1_4 이후에 attention 적용.
        """
        super(MobileViT_CBAM_based_SFA, self).__init__()
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
            self.attention_after = attention_after if attention_after is not None else [2, 3, 4]
            self.residual_connection = residual_connection
                
            # stage1_0: x.shape =>  torch.Size([32, 32, 256, 256])
            # stage1_1: x.shape =>  torch.Size([32, 64, 128, 128])
            # stage1_2: x.shape =>  torch.Size([32, 96, 64, 64])
            # stage1_3: x.shape =>  torch.Size([32, 128, 32, 32])
            # stage1_4: x.shape =>  torch.Size([32, 160, 16, 16])
            if 0 in self.attention_after:
                self.SFA_1_0 = SFA(gate_channels=32, reduction_ratio=16, pool_types=['max', 'avg'], attention_type='SFA', trainable_filter=trainable_filter)
            if 1 in self.attention_after:
                self.SFA_1_1 = SFA(gate_channels=64, reduction_ratio=16, pool_types=['max', 'avg'], attention_type='SFA', trainable_filter=trainable_filter)
            if 2 in self.attention_after:
                self.SFA_1_2 = SFA(gate_channels=96, reduction_ratio=16, pool_types=['max', 'avg'], attention_type='SFA', trainable_filter=trainable_filter)
            if 3 in self.attention_after:
                self.SFA_1_3 = SFA(gate_channels=128, reduction_ratio=16, pool_types=['max', 'avg'], attention_type='SFA', trainable_filter=trainable_filter)
            if 4 in self.attention_after:
                self.SFA_1_4 = SFA(gate_channels=160, reduction_ratio=16, pool_types=['max', 'avg'], attention_type='SFA', trainable_filter=trainable_filter)


    def forward(self, x):
        if self.return_features:
            feature_maps = []

        # Stage 0
        x = self.stage0(x)
        if self.return_features:
            feature_maps.append(x)
        
        # Stage 1_0
        x = self.stage1_0(x)
        if self.apply_sfa and (0 in self.attention_after):
            if self.residual_connection:
                x=x+self.SFA_1_0(x)
            else:
                x=self.SFA_1_0(x)
        if self.return_features:
            feature_maps.append(x)

        # Stage 1_1
        x = self.stage1_1(x)
        if self.apply_sfa and (1 in self.attention_after):
            if self.residual_connection:
                x=x+self.SFA_1_1(x)
            else:
                x=self.SFA_1_1(x)
        if self.return_features:
            feature_maps.append(x)

        # Stage 1_2
        x = self.stage1_2(x)
        if self.apply_sfa and (2 in self.attention_after):
            if self.residual_connection:
                x=x+self.SFA_1_2(x)
            else:
                x=self.SFA_1_2(x)
        if self.return_features:
            feature_maps.append(x)

        # Stage 1_3
        x = self.stage1_3(x)
        if self.apply_sfa and (3 in self.attention_after):
            if self.residual_connection:
                x=x+self.SFA_1_3(x)
            else:    
                x=self.SFA_1_3(x)
        if self.return_features:
            feature_maps.append(x)

        # Stage 1_4
        x = self.stage1_4(x)
        if self.apply_sfa and (4 in self.attention_after):
            if self.residual_connection:
                x=x+self.SFA_1_4(x)
            else:
                x=self.SFA_1_4(x)
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