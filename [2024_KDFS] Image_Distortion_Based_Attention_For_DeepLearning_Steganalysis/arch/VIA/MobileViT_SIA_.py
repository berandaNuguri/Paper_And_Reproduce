import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
from .SIA import StegoImpactAttention
import os
from PIL import Image
import matplotlib.pyplot as plt

class SIA_MobileViTBlock(nn.Module):
    def __init__(self, original_block):
        super(SIA_MobileViTBlock, self).__init__()
        self.original_block = original_block

    def forward(self, x, x_sia=None, projector=None, apply_sia=False, attention_type='add', collect_attention_features=False):
        x_local = x
        x_sia_clone = x_sia.clone() if x_sia is not None else None

        x = self.original_block.conv_kxk(x)
        x = self.original_block.conv_1x1(x)
        
        B, C, H, W = x.shape
        N = H * W
        x = x.view(B, C, N).permute(0, 2, 1)  # [B, N, C]

        # Initialize list to store attention features if required
        attention_features = []

        for i in range(len(self.original_block.transformer)):
            if x_sia_clone is not None:
                x_sia = x_sia_clone

            x = self.original_block.transformer[i].norm1(x)
            x = self.original_block.transformer[i].attn(x)

            if apply_sia and x_sia is not None:
                # Capture x before attention
                x_before_attention = x.clone() if collect_attention_features else None

                # Prepare x_sia
                B_sia, C_sia, H_sia, W_sia = x_sia.shape
                x_sia_resized = F.interpolate(
                    x_sia, 
                    size=(int(np.sqrt(x.size(1))), int(np.sqrt(x.size(1)))), 
                    mode='bilinear', 
                    align_corners=False
                )
                x_sia_view = x_sia_resized.view(B_sia, C_sia, -1).permute(0, 1, 2)

                x = self.apply_attention(x, x_sia_view, projector=projector, attention_type=attention_type)

                # Capture x after attention
                x_after_attention = x.clone() if collect_attention_features else None

                if collect_attention_features:
                    attention_features.append((x_before_attention, x_after_attention))

            x = self.original_block.transformer[i].ls1(x)
            x = self.original_block.transformer[i].drop_path1(x)
            x = self.original_block.transformer[i].norm2(x)
            x = self.original_block.transformer[i].mlp(x)
            x = self.original_block.transformer[i].ls2(x)
            x = self.original_block.transformer[i].drop_path2(x)

        # Transformer output
        x = x.permute(0, 2, 1).contiguous().view(B, C, H, W)

        x = self.original_block.conv_proj(x)

        # Concatenate x_local and x along channel dimension
        x = torch.cat((x_local, x), dim=1)

        x = self.original_block.conv_fusion(x)

        if collect_attention_features:
            for i in range(len(attention_features)):
                before_np = attention_features[i][0].detach().cpu().numpy()
                after_np = attention_features[i][1].detach().cpu().numpy()

                before_mean = np.mean(before_np)
                before_std = np.std(before_np)
                before_max = np.max(before_np)
                before_min = np.min(before_np)

                after_mean = np.mean(after_np)
                after_std = np.std(after_np)
                after_max = np.max(after_np)
                after_min = np.min(after_np)
                
                print(f'Before Attention - Mean: {before_mean:.4f}, Std: {before_std:.4f}, Max: {before_max:.4f}, Min: {before_min:.4f}')
                print(f'After Attention - Mean: {after_mean:.4f}, Std: {after_std:.4f}, Max: {after_max:.4f}, Min: {after_min:.4f}')
                # print(f'Before Attention: {attention_features[i][0]}')
                # print(f'After Attention: {attention_features[i][1]}')
            return x, attention_features
        else:
            return x
        
    def apply_attention(self, x, x_sia, projector, attention_type='add'):
        x_sia = projector(x_sia)
        x_sia = x_sia.permute(0, 2, 1)

        if attention_type == 'add':
            x = x + x_sia
        elif attention_type == 'mul':
            x = 1+(x * x_sia)
        else:
            raise ValueError(f"Invalid attention type: {attention_type}")
        return x

class MobileViT_SIA_(nn.Module):
    def __init__(self, num_classes=2, rho_scaler=0.5):
        super(MobileViT_SIA_, self).__init__()
        model = timm.create_model('mobilevit_s', num_classes=num_classes, pretrained=True)

        self.sia = StegoImpactAttention()
        self.rho_scaler = nn.Parameter(torch.tensor(rho_scaler))
        self.attention_proj_stage1_2 = nn.Conv1d(3, 144, kernel_size=1)
        self.attention_proj_stage1_3 = nn.Conv1d(3, 192, kernel_size=1)
        self.attention_proj_stage1_4 = nn.Conv1d(3, 240, kernel_size=1)

        modules = list(model.children())
        
        self.stage0 = modules[0]

        stage1_modules = list(modules[1])
        self.stage1_0 = stage1_modules[0]
        self.stage1_1 = stage1_modules[1]  
        
        self.stage1_2 = nn.Sequential(stage1_modules[2][0], SIA_MobileViTBlock(stage1_modules[2][1]))
        self.stage1_3 = nn.Sequential(stage1_modules[3][0], SIA_MobileViTBlock(stage1_modules[3][1]))   
        self.stage1_4 = nn.Sequential(stage1_modules[4][0], SIA_MobileViTBlock(stage1_modules[4][1]))

        self.stage2 = modules[2]
        self.stage3 = modules[3]

    def forward(self, x, apply_sia=False, return_features=False, attention_type='add', collect_attention_features=False):
        x_original = x.clone()  # Save the original image

        if return_features:
            feature_maps = []
        else:
            feature_maps = None

        if collect_attention_features:
            attention_features = []
        else:
            attention_features = None

        # Generate Attention Map
        if apply_sia:
            x_sia = self.sia.get_rho(x) * self.rho_scaler  # (batch_size, channels, H, W)

        # Stage 0
        x = self.stage0(x)
        if return_features:
            feature_maps.append(x)
        
        # Stage 1_0
        x = self.stage1_0(x)
        if return_features:
            feature_maps.append(x)

        # Stage 1_1
        x = self.stage1_1(x)
        if return_features:
            feature_maps.append(x)

        # Stage 1_2
        x = self.stage1_2[0](x)
        if apply_sia:
            if collect_attention_features:
                print('=======================Stage 1_2=======================')
                x, attn_feats = self.stage1_2[1](x, x_sia, self.attention_proj_stage1_2, apply_sia, attention_type, collect_attention_features)
                attention_features.append(attn_feats)
            else:
                x = self.stage1_2[1](x, x_sia, self.attention_proj_stage1_2, apply_sia, attention_type)
        else:
            x = self.stage1_2[1](x, None, None, apply_sia)
        if return_features:
            feature_maps.append(x)

        # Stage 1_3
        x = self.stage1_3[0](x)
        if apply_sia:
            if collect_attention_features:
                print('=======================Stage 1_3=======================')
                x, attn_feats = self.stage1_3[1](x, x_sia, self.attention_proj_stage1_3, apply_sia, attention_type, collect_attention_features)
                attention_features.append(attn_feats)
            else:
                x = self.stage1_3[1](x, x_sia, self.attention_proj_stage1_3, apply_sia, attention_type)
        else:
            x = self.stage1_3[1](x, None, None, apply_sia)
        if return_features:
            feature_maps.append(x)

        # Stage 1_4
        x = self.stage1_4[0](x)
        if apply_sia:
            if collect_attention_features:
                print('=======================Stage 1_4=======================')
                x, attn_feats = self.stage1_4[1](x, x_sia, self.attention_proj_stage1_4, apply_sia, attention_type, collect_attention_features)
                attention_features.append(attn_feats)
            else:
                x = self.stage1_4[1](x, x_sia, self.attention_proj_stage1_4, apply_sia, attention_type)
        else:
            x = self.stage1_4[1](x, None, None, apply_sia)
        if return_features:
            feature_maps.append(x)

        # Stage 2
        x = self.stage2(x)
        if return_features:
            feature_maps.append(x)

        # Stage 3 - Classification Head
        x = self.stage3(x)

        outputs = [x]
        if return_features:
            outputs.append(feature_maps)
        if collect_attention_features:
            outputs.append(attention_features)

        if len(outputs) == 1:
            return outputs[0]
        else:
            return tuple(outputs)
