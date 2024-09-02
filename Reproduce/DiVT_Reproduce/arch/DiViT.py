import torch
import torch.nn as nn
import torch.nn.functional as F

class DiViT(nn.Module):
    def __init__(self, encoder, num_classes, input_size=(1, 3, 256, 256)):
        super(DiViT, self).__init__()
        self.encoder = nn.Sequential(*list(encoder.children())[:-1])
        # self.encoder = encoder
        
        dummy_input = torch.randn(input_size)
        with torch.no_grad():
            dummy_output = self.encoder(dummy_input)
        
        self.feature_dim = dummy_output.size(1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc = nn.Linear(self.feature_dim, num_classes)
        
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.fc.weight, mode='fan_out', nonlinearity='relu')
        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias, 0)
            
    def forward(self, x):
        features = self.encoder(x)
    
        pooled_features = self.pool(features)
        flattened_features = torch.flatten(pooled_features, 1)
        
        logits = self.fc(flattened_features)
        
        return logits, features
