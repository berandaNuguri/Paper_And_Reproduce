import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Function

class DSDBNet(nn.Module):
    def __init__(self, spect_encoder, png_encoder, num_classes, feature_fusion='concat' ,global_pool=True, spect_input_size=(1, 3, 224, 224), png_input_size=(1, 3, 224, 224)):
        super(DSDBNet, self).__init__()
        self.spect_encoder = nn.Sequential(*list(spect_encoder.children())[:-1])
        self.png_encoder = nn.Sequential(*list(png_encoder.children())[:-1])
        self.feature_fusion = feature_fusion
        self.global_pool = global_pool

        if self.global_pool:
            self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.dummy_input = torch.randn(spect_input_size)
        with torch.no_grad():
            dummy_output = self.spect_encoder(self.dummy_input)
            if self.global_pool:
                dummy_output = self.global_avg_pool(dummy_output)

            self.spect_feature_dim = dummy_output.size(1)
            
        self.dummy_input = torch.randn(png_input_size)
        with torch.no_grad():
            dummy_output = self.png_encoder(self.dummy_input)
            if self.global_pool:
                dummy_output = self.global_avg_pool(dummy_output)

            self.png_feature_dim = dummy_output.size(1)
        
        if self.feature_fusion == 'concat':
            self.fc = nn.Linear(self.spect_feature_dim + self.png_feature_dim, num_classes)
        elif self.feature_fusion == 'sum':
            self.fc = nn.Linear(self.spect_feature_dim, num_classes)
        else:
            raise ValueError(f'Invalid feature fusion method: {self.feature_fusion}')

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, spect, png, return_features=False):
        if self.global_pool:
            spect_features = self.global_avg_pool(self.spect_encoder(spect))
            png_features = self.global_avg_pool(self.png_encoder(png))
        else:
            spect_features = self.spect_encoder(spect)
            png_features = self.png_encoder(png)

        spect_features = spect_features.view(spect_features.size(0), -1)
        png_features = png_features.view(png_features.size(0), -1)

        if self.feature_fusion == 'concat':
            fusion_features = torch.cat((spect_features, png_features), dim=1)
        elif self.feature_fusion == 'sum':
            fusion_features = spect_features + png_features

        outputs = self.fc(fusion_features)
        
        if return_features:
            return outputs, fusion_features
        else:
            return outputs

class SiameseNet(nn.Module):
    def __init__(self, encoder, num_classes, feature_fusion='concat' ,global_pool=True, input_size=(1, 30, 224, 224)):
        super(SiameseNet, self).__init__()
        self.encoder = nn.Sequential(*list(encoder.children())[:-1])
        self.feature_fusion = feature_fusion
        self.global_pool = global_pool

        if self.global_pool:
            self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.dummy_input = torch.randn(input_size)
        with torch.no_grad():
            dummy_output = self.encoder(self.dummy_input)
            if self.global_pool:
                dummy_output = self.global_avg_pool(dummy_output)

            self.feature_dim = dummy_output.size(1)
        
        if self.feature_fusion == 'concat':
            self.fc = nn.Linear(self.feature_dim * 2, num_classes)
        elif self.feature_fusion == 'sum':
            self.fc = nn.Linear(self.feature_dim, num_classes)
        else:
            raise ValueError(f'Invalid feature fusion method: {self.feature_fusion}')

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, spect, png, return_features=False):
        if self.global_pool:
            features_1 = self.global_avg_pool(self.encoder(spect))
            features_2 = self.global_avg_pool(self.encoder(png))
        else:
            features_1 = self.encoder(spect)
            features_2 = self.encoder(png)

        features_1 = features_1.view(features_1.size(0), -1)
        features_2 = features_2.view(features_2.size(0), -1)

        if self.feature_fusion == 'concat':
            concatenated_features = torch.cat((features_1, features_2), dim=1)
        elif self.feature_fusion == 'sum':
            concatenated_features = features_1 + features_2

        outputs = self.fc(concatenated_features)

        if return_features:
            return outputs, concatenated_features
        else:
            return outputs

