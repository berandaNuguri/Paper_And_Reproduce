import os
import random
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from utils.dataset import DiViTDataset
from utils.sample_frames import sample_frames

from torch.utils.data import DataLoader, WeightedRandomSampler, ConcatDataset
import torch

def WeightedSampler(dataset):
    labels = [label for _, label in dataset]
    labels = torch.tensor(labels)
    
    real_count = (labels == 0).sum().item()
    fake_count = (labels != 0).sum().item()

    real_weight = 1. / real_count
    fake_weight = 1. / fake_count
    
    weights = [real_weight if label == 0 else fake_weight for label in labels]
    
    return WeightedRandomSampler(weights, len(weights), replacement=True)

"""
torch Weighted Sampler이용한 데이터로더(구 버전)
"""
# def get_dataset(src1_data, src1_train_num_frames, src2_data, src2_train_num_frames, src3_data, src3_train_num_frames,
#                 tgt_data, tgt_test_num_frames, batch_size):
#     print('Load Source Data')
#     src1_train_data = sample_frames(bench=src1_data, flag='train', num_frames=src1_train_num_frames)
#     src2_train_data = sample_frames(bench=src2_data, flag='train', num_frames=src2_train_num_frames)
#     src3_train_data = sample_frames(bench=src3_data, flag='train', num_frames=src3_train_num_frames)

#     print('Load Target Data')
#     tgt_test_data = sample_frames(bench=tgt_data, flag='test', num_frames=tgt_test_num_frames)

#     concatenated_dataset = ConcatDataset([
#         DiViTDataset(src1_train_data, train=True),
#         DiViTDataset(src2_train_data, train=True),
#         DiViTDataset(src3_train_data, train=True)
#     ])
    
#     num_classes = len(torch.unique(torch.tensor([data[1] for data in concatenated_dataset])))
#     print(f'Total Classes: {num_classes}')
#     print('Create Class Weighted Sampler')
#     weighted_sampler = WeightedSampler(concatenated_dataset)

#     train_dataloader = DataLoader(concatenated_dataset, batch_size=batch_size, sampler=weighted_sampler, num_workers=16)
#     # train_dataloader = DataLoader(concatenated_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
#     tgt_dataloader = DataLoader(DiViTDataset(tgt_test_data, train=False), batch_size=batch_size, shuffle=False, num_workers=16)
    
#     return train_dataloader, tgt_dataloader, num_classes

"""
SSDG에서 사용한 sample 별 데이터로더 생성하여 Balacned Sampling
"""
def get_dataset(src1_data, src1_train_num_frames, src2_data, src2_train_num_frames, src3_data, src3_train_num_frames,
                tgt_data, tgt_test_num_frames, batch_size):
    print('Load Source Data')
    src1_train_real_data = sample_frames(bench=src1_data, mode='train', flag=0, num_frames=src1_train_num_frames)
    src1_train_fake_data = sample_frames(bench=src1_data, mode='train', flag=1, num_frames=src1_train_num_frames)
    src2_train_real_data = sample_frames(bench=src2_data, mode='train', flag=0, num_frames=src2_train_num_frames)
    src2_train_fake_data = sample_frames(bench=src2_data, mode='train', flag=1, num_frames=src2_train_num_frames)
    src3_train_real_data = sample_frames(bench=src3_data, mode='train', flag=0, num_frames=src3_train_num_frames)
    src3_train_fake_data = sample_frames(bench=src3_data, mode='train', flag=1, num_frames=src3_train_num_frames)

    print('Load Target Data')
    tgt_test_data = sample_frames(bench=tgt_data, mode='test', flag=2, num_frames=tgt_test_num_frames)

    src1_train_dataloader_real = DataLoader(DiViTDataset(src1_train_real_data, train=True), batch_size=batch_size, shuffle=True, num_workers=4)
    src1_train_dataloader_fake = DataLoader(DiViTDataset(src1_train_fake_data, train=True), batch_size=batch_size, shuffle=True, num_workers=4)
    src2_train_dataloader_real = DataLoader(DiViTDataset(src2_train_real_data, train=True), batch_size=batch_size, shuffle=True, num_workers=4)
    src2_train_dataloader_fake = DataLoader(DiViTDataset(src2_train_fake_data, train=True), batch_size=batch_size, shuffle=True, num_workers=4)
    src3_train_dataloader_real = DataLoader(DiViTDataset(src3_train_real_data, train=True), batch_size=batch_size, shuffle=True, num_workers=4)
    src3_train_dataloader_fake = DataLoader(DiViTDataset(src3_train_fake_data, train=True), batch_size=batch_size, shuffle=True, num_workers=4)

    tgt_dataloader = DataLoader(DiViTDataset(tgt_test_data, train=False), batch_size=batch_size, shuffle=False, num_workers=16)
    
    return src1_train_dataloader_fake, src1_train_dataloader_real, \
           src2_train_dataloader_fake, src2_train_dataloader_real, \
           src3_train_dataloader_fake, src3_train_dataloader_real, \
           tgt_dataloader