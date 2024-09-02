import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from arch.cbam import ChannelGate, SpatialGate

class SiameseNetwork(nn.Module):
    def __init__(self, backbone, num_classes, global_pool):
        super(SiameseNetwork, self).__init__()
        # self.backbone = nn.Sequential(*list(backbone.children())[:])
        self.backbone = backbone
        
        self.num_classes = num_classes
        self.global_pool = global_pool

        if self.global_pool:
            self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            
        self.dummy_input = torch.randn((2,1,500,500))
        with torch.no_grad():
            dummy_output = self.backbone(self.dummy_input)
            if self.global_pool:
                dummy_output = self.global_avg_pool(dummy_output)

        self.n_features = dummy_output.size(1)//2

        self.classifier = nn.Sequential(
            nn.Linear(self.n_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.num_classes),
        )

        self.backbone.apply(self.init_weights)
        self.classifier.apply(self.init_weights)
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def forward_once(self, x, concat=False):
        cnn_features, attn_features = self.backbone(x, concat=False)
        return cnn_features, attn_features

    def forward(self, input1, input2):
        if self.global_pool:
            output1 = self.global_avg_pool(self.forward_once(input1))
            output2 = self.global_avg_pool(self.forward_once(input2))
        else:
            cnn_features1, attn_features1 = self.forward_once(input1)
            cnn_features2, attn_features2 = self.forward_once(input2)

        # features = torch.cat((output1, output2), 1)
        # features = features.view(features.size()[0], -1)
        cnn_features = 0.2 * (cnn_features1 * cnn_features2)
        attn_features = 0.8 * (attn_features1 * attn_features2)

        features = cnn_features + attn_features
        output = self.classifier(features)
        
        return output, features

class AFRNet(nn.Module):
    def __init__(self, backbone, in_chans=1, num_classes=1):
        super(AFRNet, self).__init__()
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.spatial_alignment = nn.Sequential(
            nn.Conv2d(self.in_chans, 16, kernel_size=7, padding=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 24, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(24, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 48, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(48, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(64*15*15, 32),
            nn.Linear(32, 4)
        )
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, padding=3, stride=2),
            self.backbone[1],
            self.backbone[2],
            self.backbone[3],
            self.backbone[4],
            self.backbone[5],
            self.backbone[6],
        )
        self.cnn_head = nn.Sequential(
            self.backbone[7],
            nn.Flatten(),
            nn.Linear(2048*16*16, 384, bias=True)
        )
        self.attn_head = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 384),
            nn.TransformerEncoderLayer(d_model=384, dim_feedforward=1536, nhead=12),
            nn.Linear(384, 384, bias=True)
        )

    def forward(self, x, concat=True):
        # alignment_params = self.spatial_alignment(x)
        # theta = torch.tensor([[
        #     [alignment_params[0], alignment_params[1], 0],
        #     [alignment_params[2], alignment_params[3], 0]
        # ]], dtype=torch.float32).to(x.device)
        # grid = F.affine_grid(theta, x.size(), align_corners=False)
        # x = F.grid_sample(x, grid, align_corners=False)
        
        x = self.feature_extractor(x)
        cnn_features = self.cnn_head(x)
        patches = x.flatten(2).permute(2, 0, 1)
        attn_features = self.attn_head(patches)
        attn_features = attn_features[0]
        
        if concat:
            features = torch.cat([cnn_features, attn_features], dim=1)
            return features
        else:
            return cnn_features, attn_features

class IAESNet(nn.Module):
    def __init__(self, num_classes=1):
        super(IAESNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)

        self.pool = nn.MaxPool2d(3, stride=2)

        self.relu = nn.ReLU()

        self.fc = nn.Sequential(nn.Linear(64 * 61 * 61, 64),
                                nn.ReLU(),
                                nn.Linear(64, num_classes))

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)

        x = torch.flatten(x, 1)

        x = self.fc(x)
        
        return x

class CASAResnet(nn.Module):
    def __init__(self, backbone, reduction_ratio=16):
        super(CASAResnet, self).__init__()
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        
        self.layer0 = self.backbone[:4]
        self.layer1 = self.backbone[4]
        self.layer2 = self.backbone[5]
        self.layer3 = self.backbone[6]
        self.layer4 = self.backbone[7]
            
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(832, 1)
    
    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)

        x = ChannelGate(x.size(2), reduction_ratio=16).to(x.device)(x)
        attentionmap_1 = SpatialGate().to(x.device)(x)

        x = self.layer2(attentionmap_1)
        x = self.layer3(x)

        x = ChannelGate(x.size(2), reduction_ratio=16).to(x.device)(x)
        attentionmap_2 = SpatialGate().to(x.device)(x)
        
        x = self.layer4(attentionmap_2)

        x = ChannelGate(x.size(2), reduction_ratio=16).to(x.device)(x)
        attentionmap_3 = SpatialGate().to(x.device)(x)

        attentionmap_1 = self.pool(attentionmap_1).flatten(1)
        attentionmap_2 = self.pool(attentionmap_2).flatten(1)
        attentionmap_3 = self.pool(attentionmap_3).flatten(1)
        
        combined_features = torch.cat([attentionmap_1, attentionmap_2, attentionmap_3], dim=1)
        output = self.fc(combined_features)
        
        return output

class ChannelAttentionModule(nn.Module):
    def __init__(self, reduction_ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.reduction_ratio = reduction_ratio
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        avg_feature = self.avg_pool(x)
        max_feature = self.max_pool(x)
        
        concat_feature = torch.cat([avg_feature, max_feature], dim=1)
        
        mlp = nn.Sequential(
            nn.Linear(channels * 2, channels // self.reduction_ratio).to(x.device),
            nn.ReLU(inplace=True),
            nn.Linear(channels // self.reduction_ratio, channels).to(x.device)
        )
        
        attention_map = mlp(concat_feature.view(batch_size, -1))
        attention_map = attention_map.view(batch_size, channels, 1, 1)
        
        scaled_feature = x * attention_map.expand_as(x)
        
        return scaled_feature

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        conv1 = nn.Conv2d(channels, 1, kernel_size=1, stride=1, padding=0).to(x.device)
        bn1 = nn.BatchNorm2d(1).to(x.device)
        relu1 = nn.ReLU(inplace=True)
        
        spatial_attention_map = relu1(bn1(conv1(x)))
        
        return x * spatial_attention_map.expand_as(x)