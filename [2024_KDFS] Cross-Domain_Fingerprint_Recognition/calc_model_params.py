import torch
import torch.nn as nn
import timm

from arch.architectures import SiameseNetwork, AFRNet

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params / 1000000:.1f}M")
    print(f"Trainable Parameters: {trainable_params / 1000000:.1f}M")

if __name__ == "__main__":
    count_parameters(AFRNet(backbone=timm.create_model('resnet50', pretrained=True, in_chans=1), num_classes=2))
    count_parameters(timm.create_model('efficientnet_b0', pretrained=True, in_chans=1))
    # count_parameters(SiameseNetwork(AFRNet(backbone=timm.create_model('resnet50', pretrained=True, in_chans=1)), num_classes=2, global_pool=False))
    # count_parameters(SiameseNetwork(timm.create_model('efficientnet_b0', pretrained=True, in_chans=1), num_classes=2, global_pool=False))