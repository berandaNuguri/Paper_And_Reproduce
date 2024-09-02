import torch
import torch.nn as nn
import torch.nn.functional as F

class DiCLoss(nn.Module):
    def __init__(self):
        super(DiCLoss, self).__init__()

    def forward(self, features, labels):
        if features.size(0) != labels.size(0):
            raise ValueError('Feature size and label size are different!!')

        features = features[labels == 0]
        
        if features.numel() == 0:
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        # abs_features = torch.abs(features)
        # l1_norms = torch.sum(abs_features, dim=1)
        # dic_loss = torch.mean(l1_norms)
        
        # calc l1-norm using torch.norm
        dic_loss = torch.norm(features, p=1, dim=1).mean()

        # calc diff between two methods using sum
        # diff = torch.sum(torch_dic_loss - dic_loss)
        # print(f"torch_dic_loss: {torch_dic_loss}, dic_loss: {dic_loss}, diff: {diff}")

        return dic_loss