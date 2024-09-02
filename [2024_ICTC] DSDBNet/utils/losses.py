import torch
import torch.nn as nn

class ChannelWeightedLoss(nn.Module):
    def __init__(self, weights, mode='L1'):
        super(ChannelWeightedLoss, self).__init__()
        self.weights = torch.tensor(weights, dtype=torch.float32)
        self.mode = mode

    def forward(self, output, target):
        if self.mode == 'L1':
            loss = torch.mean(torch.abs(output - target) * self.weights.view(1, -1, 1, 1))
        elif self.mode == 'L2':
            squared_diff = (output - target) ** 2
            weighted_squared_diff = squared_diff * self.weights.view(1, -1, 1, 1)
            loss = torch.mean(weighted_squared_diff)
        else:
            raise ValueError(f"Unsupported loss mode: {self.mode}")
        
        return loss